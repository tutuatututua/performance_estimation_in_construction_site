import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler
from sklearn.model_selection import train_test_split, StratifiedKFold
import cv2
from tqdm import tqdm
import os
from pathlib import Path
from collections import defaultdict
import logging
import gc
import time
from typing import Optional, Tuple, Dict, Any

try:
    from utils.pose_extraction import (
        detect_upper_body_pose,
        normalize_joints,
        PoseMemory,
        calculate_displacement
    )
    from utils.model_loader import load_pose_model
except ImportError:
    print("ERROR: Failed to import from utils.pose_extraction or utils.model_loader. Check PYTHONPATH.")
    raise

logger = logging.getLogger(__name__)

class MovementSequenceDataset(Dataset):
    """Custom dataset for movement sequences."""
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray, # Store numeric labels (displacement-based scores)
        job_ids: np.ndarray,
        video_indices: Optional[np.ndarray] = None,
        transform=None
    ):
        self.sequences = torch.FloatTensor(np.array(sequences, dtype=np.float32))
        # Store the original numeric displacement-based score calculated during processing
        # This is NOT the reconstruction error used by the AE for anomaly detection,
        # but it might be useful for analysis or other tasks.
        self.displacement_scores = torch.FloatTensor(np.array(labels, dtype=np.float32))
        self.job_ids = torch.LongTensor(np.array(job_ids, dtype=np.int64))
        if video_indices is not None and len(video_indices) == len(self.sequences):
            self.video_indices = torch.LongTensor(np.array(video_indices, dtype=np.int64))
        else:
            self.video_indices = None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        # Return the displacement score calculated earlier
        label = self.displacement_scores[idx]
        job_id = self.job_ids[idx]
        video_idx = self.video_indices[idx] if self.video_indices is not None else -1

        if self.transform is not None and callable(self.transform):
            sequence = self.transform(sequence)

        # Return sequence, displacement score, job_id
        return sequence, label, job_id


class _TrainingBatchProcessor:
    """Helper class for batch processing frames during training data prep."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tracker_state = None

    def initialize_model(self):
        """Initialize YOLO model."""
        if self.model is None:
            try:
                self.model = load_pose_model(str(self.config.yolo_model_path), self.config)
                if torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                logger.info("YOLO model initialized for batch processing.")
            except Exception as e:
                 logger.error(f"Failed to initialize YOLO model: {e}")
                 raise

    def process_batch(self, frames):
        """Process a batch of frames using the YOLO model."""
        if self.model is None: self.initialize_model()

        results_list = []
        try:
            use_tracking = getattr(self.config, 'use_tracking', True)
            if not isinstance(frames, list): frames = [frames]
            if not frames: return [] # Handle empty input

            if use_tracking:
                tracker_args = {
                    'conf': self.config.yolo_conf_threshold, 'iou': self.config.yolo_iou_threshold,
                    'persist': True, 'tracker': self.config.tracker_config, 'verbose': False
                }
                for frame in frames:
                    if not isinstance(frame, np.ndarray):
                         logger.warning("Frame is not a NumPy array, skipping.")
                         results_list.append(None); continue
                    try:
                        yolo_results = self.model.track(frame, **tracker_args)
                        results_list.append(yolo_results[0] if yolo_results else None)
                    except Exception as track_err:
                         logger.error(f"Error during tracking for a frame: {track_err}")
                         results_list.append(None)
            else:
                 yolo_results_batch = self.model(
                     frames, conf=self.config.yolo_conf_threshold, iou=self.config.yolo_iou_threshold,
                     stream=False, verbose=False
                 )
                 results_list = yolo_results_batch if isinstance(yolo_results_batch, list) else [yolo_results_batch]

            if len(results_list) != len(frames):
                 logger.warning(f"Result count mismatch: {len(results_list)} vs {len(frames)}. Padding.")
                 results_list.extend([None] * (len(frames) - len(results_list)))
            return results_list

        except Exception as e:
            logger.error(f"Error processing frame batch: {str(e)}")
            import traceback; logger.error(traceback.format_exc())
            return [None] * len(frames)

    def reset_tracker(self):
        """Reset the tracker state."""
        self.tracker_state = None
        logger.info("Tracker state reset (handled internally by YOLOv8).")


class MovementDataProcessor:
    """Data processing class using percentile thresholds."""

    def __init__(self, config):
        self.config = config
        self.num_workers = getattr(config, 'dataloader_workers', min(os.cpu_count() or 1, 4))
        self.batch_processor = _TrainingBatchProcessor(config)
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = getattr(config, 'batch_size_frames', 32)
        self.max_chunk_size = getattr(config, 'max_memory_frames', 200)
        self.processed_videos = set()
        self.video_info = {}
        self.video_paths = []
        self.train_loader = None
        self.val_loader = None
        # Thresholds are no longer calculated or stored here. They come from evaluator.

    def _get_cache_config_params(self) -> Dict[str, Any]:
        """Returns config parameters relevant for cache validation."""
        return {
            'config_num_joints': self.config.num_joints,
            'config_sample_rate': self.config.sample_rate,
            'config_min_seq_len': self.config.min_sequence_length,
            # Removed disp_scale_factor as it's not used for AE thresholding
        }

    def _load_from_cache(self, cache_path: Path, video_path: Path) -> Optional[Dict[str, Any]]:
        """Attempt to load processed data (sequences AND displacement scores) from cache."""
        if not self.config.enable_cache or not cache_path.exists():
            return None
        try:
            cached_data = np.load(str(cache_path), allow_pickle=True)
            required_keys = ['sequences', 'labels', 'num_frames', 'num_people', 'video_name', 'config_params']
            if not all(key in cached_data for key in required_keys):
                logger.warning(f"Invalid cache (missing keys) for {video_path.name}. Reprocessing.")
                return None
            if cached_data['video_name'] != video_path.name:
                logger.warning(f"Cache name mismatch for {video_path.name}. Reprocessing.")
                return None
            cached_config = cached_data['config_params'].item()
            current_config_params = self._get_cache_config_params()
            mismatched_params = []
            for key, current_val in current_config_params.items():
                cached_val = cached_config.get(key)
                if isinstance(current_val, float):
                    if cached_val is None or not np.isclose(current_val, cached_val):
                        mismatched_params.append(f"{key} (current={current_val}, cached={cached_val})")
                elif current_val != cached_val:
                    mismatched_params.append(f"{key} (current={current_val}, cached={cached_val})")
            if mismatched_params:
                logger.warning(f"Cache config mismatch for {video_path.name}: {', '.join(mismatched_params)}. Reprocessing.")
                return None

            sequences = cached_data['sequences']
            labels = cached_data['labels'] # These are the displacement scores
            if not isinstance(sequences, np.ndarray) or not isinstance(labels, np.ndarray):
                 logger.warning(f"Invalid cache data types for {video_path.name}. Reprocessing.")
                 return None
            if len(sequences) != len(labels):
                 logger.warning(f"Cache sequence/label count mismatch for {video_path.name}. Reprocessing.")
                 return None
            if len(sequences) > 0 and isinstance(sequences[0], np.ndarray):
                 expected_features = current_config_params['config_num_joints']
                 first_seq_features = sequences[0].shape[1] if sequences[0].ndim == 2 else -1
                 if first_seq_features != expected_features:
                      logger.warning(f"Cache sequence feature mismatch for {video_path.name} ({first_seq_features} vs {expected_features}). Reprocessing.")
                      return None

            return {
                'sequences': sequences,
                'labels': labels, # Return displacement scores
                'num_frames': cached_data['num_frames'],
                'num_people': cached_data['num_people'],
                'video_name': video_path.name
            }
        except Exception as e:
            logger.warning(f"Cache load failed for {video_path.name}: {e}")
        return None

    def process_single_video(self, video_path: Path, job_type: str) -> Optional[Dict[str, Any]]:
        """Process a single video, extracting sequences and calculating displacement scores."""
        logger.info(f"Processing video: {video_path.name} (Job: {job_type})")
        cache_path = self.cache_dir / f"{video_path.stem}_cache.npz"
        cached_data = self._load_from_cache(cache_path, video_path)
        if cached_data:
            if 'labels' not in cached_data or cached_data['labels'] is None:
                 logger.warning(f"Cache for {video_path.name} missing 'labels'. Reprocessing.")
                 cached_data = None
            else:
                 if 'job_type' not in cached_data: cached_data['job_type'] = job_type
                 return cached_data

        self.batch_processor.reset_tracker()
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened(): logger.warning(f"Could not open video: {video_path.name}"); return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0; logger.warning(f"Invalid FPS for {video_path.name}, using 30.")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, int(round(fps / self.config.sample_rate)))
            target_width, target_height = self.config.frame_resize
            pose_memory = PoseMemory(memory_frames=15, min_keypoint_conf=self.config.yolo_min_keypoint_conf)
            person_data = defaultdict(lambda: {'poses': [], 'timestamps': []})
            frames_processed_count = 0; frames_analyzed_count = 0; frames_buffer = []

            with tqdm(total=total_frames, desc=f"Extracting {video_path.name}", unit="frame", leave=False) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frames_processed_count += 1; pbar.update(1)
                    if (frames_processed_count - 1) % frame_interval == 0:
                        frames_analyzed_count += 1
                        frame_resized = cv2.resize(frame, (target_width, target_height))
                        frames_buffer.append(frame_resized)
                        if len(frames_buffer) >= self.batch_size or frames_processed_count == total_frames:
                            results_batch = self.batch_processor.process_batch(frames_buffer)
                            for i, result in enumerate(results_batch):
                                if result is None: continue
                                frame_in_buffer = frames_buffer[i]
                                timestamp = (frames_processed_count - len(frames_buffer) + i) / fps
                                poses = detect_upper_body_pose(frame_in_buffer, result, pose_memory, self.config)
                                for person_id, pose in poses:
                                    normalized_pose = normalize_joints(pose)
                                    if normalized_pose is not None:
                                        person_data[person_id]['poses'].append(normalized_pose)
                                        person_data[person_id]['timestamps'].append(timestamp)
                            frames_buffer = []
                            if frames_analyzed_count % self.max_chunk_size == 0:
                                gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

            all_sequences = []; all_displacement_scores = [] # Renamed from all_numeric_labels
            min_len_for_sequence = max(2, getattr(self.config, 'min_sequence_length', 80) // 4)
            num_features = getattr(self.config, 'num_joints', 6)

            # Use a fixed scale factor for displacement score calculation here
            # Note: This score is primarily for potential analysis later, NOT for AE thresholding
            displacement_score_scale_factor = 5.0 # Example fixed scale factor

            for person_id, data in person_data.items():
                poses_list = data['poses']
                if len(poses_list) >= min_len_for_sequence:
                    try:
                        displacement_sequences = self._calculate_sequences(
                            poses_list, window_size=self.config.min_sequence_length
                        )
                        if displacement_sequences:
                             valid_sequences_person = [seq for seq in displacement_sequences if seq is not None and len(seq) > 0]
                             if not valid_sequences_person: continue
                             sequences_np_person = [np.array(seq) for seq in valid_sequences_person]

                             for seq in sequences_np_person:
                                 if not isinstance(seq, np.ndarray) or seq.size == 0: continue
                                 try:
                                     if seq.ndim != 2 or seq.shape[1] != num_features:
                                         logger.warning(f"Skipping label calculation for invalid sequence shape: {seq.shape}")
                                         continue
                                     all_sequences.append(seq)

                                     joint_rms = np.sqrt(np.nanmean(np.square(seq), axis=0))

                                     mean_rms = np.nanmean(joint_rms) if joint_rms.size > 0 else 0.0

                                     if not np.isfinite(mean_rms):
                                         mean_rms = 0.0

                                     movement_score = max(0.0, mean_rms * displacement_score_scale_factor)
                                     movement_score = min(100.0, movement_score)
                                     all_displacement_scores.append(movement_score)

                                 except Exception as label_err:
                                     logger.warning(f"Could not calculate displacement score for a sequence in {video_path.name}: {label_err}")
                                     if len(all_sequences) > len(all_displacement_scores): all_sequences.pop()
                    except Exception as e:
                        logger.error(f"Error generating sequences/scores for person {person_id} in {video_path.name}: {str(e)}", exc_info=True)

            valid_sequences_final = []; valid_labels_final = []
            num_features = getattr(self.config, 'num_joints', 6) # Get expected features again for validation

            for i in range(min(len(all_sequences), len(all_displacement_scores))):
                seq = all_sequences[i]; label = all_displacement_scores[i]
                is_valid_seq = ( isinstance(seq, np.ndarray) and np.issubdtype(seq.dtype, np.number) and
                                 seq.ndim == 2 and seq.shape[0] > 0 and seq.shape[1] == num_features )
                is_valid_label = np.isfinite(label)
                if is_valid_seq and is_valid_label:
                    valid_sequences_final.append(seq); valid_labels_final.append(label)
                else:
                    seq_info = f"dtype: {getattr(seq, 'dtype', 'N/A')}, shape: {getattr(seq, 'shape', 'N/A')}" if isinstance(seq, np.ndarray) else f"type: {type(seq)}"
                    logger.warning(f"Discarding invalid sequence/label pair idx {i} for {video_path.name}. Seq valid: {is_valid_seq} ({seq_info}), Label valid: {is_valid_label} (value: {label})")

            all_sequences_filtered = valid_sequences_final
            all_displacement_scores_filtered = valid_labels_final # Renamed

            if not all_sequences_filtered:
                 logger.warning(f"No valid sequences/labels remain after filtering for {video_path.name}.")
                 return None

            result_data = {
                'sequences': np.array(all_sequences_filtered, dtype=object),
                'labels': np.array(all_displacement_scores_filtered, dtype=np.float32), # Save displacement scores as 'labels'
                'num_frames': frames_processed_count,
                'num_people': len(person_data),
                'video_name': video_path.name,
                'job_type': job_type,
                'config_params': self._get_cache_config_params()
            }

            if self.config.enable_cache:
                try: np.savez_compressed(str(cache_path), **result_data); logger.debug(f"Saved cache for {video_path.name}")
                except Exception as e: logger.warning(f"Cache save failed for {video_path.name}: {e}")

            return {k: v for k, v in result_data.items() if k != 'config_params'}

        except Exception as e:
            logger.error(f"Error processing video {video_path.name}: {str(e)}", exc_info=True)
            return None
        finally:
            if cap is not None: cap.release()
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _collect_video_files(self):
        """Collect all video files organized by job type."""
        logger.info("Collecting video files...")
        video_files = []
        self.video_paths = []
        self.video_info = {}
        for job_idx, job_type in enumerate(self.config.job_categories):
            job_dir = Path(self.config.video_dir) / job_type
            if not job_dir.is_dir(): logger.warning(f"Job directory not found: {job_dir}"); continue
            job_video_paths = []
            for ext in ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']:
                job_video_paths.extend(list(job_dir.glob(f"*{ext}")))
            unique_paths = sorted(list(set(job_video_paths)))
            for path in unique_paths:
                video_files.append({'path': path, 'job_idx': job_idx, 'job_type': job_type})
                self.video_paths.append(path)
                self.video_info[str(path)] = {'job_type': job_type, 'job_idx': job_idx}
            logger.info(f"Found {len(unique_paths)} videos for '{job_type}'")
        if not video_files: logger.error(f"No video files found in job subdirs of {self.config.video_dir}"); raise FileNotFoundError("No training video files found.")
        logger.info(f"Total video files collected: {len(video_files)}")
        return video_files

    def _process_videos(self, video_files):
        """Process videos sequentially, collecting sequences and displacement scores."""
        all_sequences, all_labels, all_job_ids, video_indices = [], [], [], []
        problem_videos = []
        total_videos = len(video_files); processed_count = 0
        for video_idx, file_info in enumerate(video_files):
            video_path = file_info['path']; job_idx = file_info['job_idx']; job_type = file_info['job_type']
            processed_count += 1
            logger.info(f"--- Processing video {processed_count}/{total_videos}: {video_path.name} ---")
            if str(video_path) in self.processed_videos: continue
            result = self.process_single_video(video_path, job_type)
            self.processed_videos.add(str(video_path))
            if result and 'sequences' in result and 'labels' in result and \
               result['sequences'] is not None and result['labels'] is not None and \
               len(result['sequences']) > 0 and len(result['sequences']) == len(result['labels']):
                sequences = result['sequences']; numeric_labels = result['labels'] # Displacement scores
                all_sequences.extend(sequences); all_labels.extend(numeric_labels)
                all_job_ids.extend([job_idx] * len(numeric_labels)); video_indices.extend([video_idx] * len(numeric_labels))
            else:
                log_msg = f"Skipping {video_path.name}: "; reason = "Unknown reason."
                if not result: reason = "Processing failed."
                elif 'sequences' not in result or result['sequences'] is None or len(result['sequences']) == 0: reason = "No valid sequences."
                elif 'labels' not in result or result['labels'] is None: reason = "No labels returned."
                elif len(result.get('sequences',[])) != len(result.get('labels',[])): reason = "Seq/label count mismatch."
                logger.warning(log_msg + reason); problem_videos.append(video_path.name)
            if processed_count % 5 == 0: gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if problem_videos: logger.warning(f"Videos with issues: {', '.join(problem_videos)}")
        logger.info(f"Finished processing videos. Total sequences: {len(all_sequences)}")
        return {'sequences': all_sequences, 'labels': all_labels, 'job_ids': all_job_ids, 'video_indices': video_indices}

    def _calculate_sequences(self, poses_list, window_size=20):
        """Calculate sequences of displacement vectors."""
        if not poses_list or len(poses_list) < 2: return []
        displacements = []
        num_features = getattr(self.config, 'num_joints', 6)
        for i in range(len(poses_list) - 1):
            disp = calculate_displacement(poses_list[i+1], poses_list[i])
            if disp is not None: displacements.append(np.nan_to_num(np.array(disp, dtype=np.float32)))
            else: displacements.append(np.zeros(num_features, dtype=np.float32))
        sequences = []
        num_displacements = len(displacements)
        sequence_length = window_size
        if num_displacements < sequence_length: return []
        stride = max(1, sequence_length // 4)
        for start_idx in range(0, num_displacements - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            sequences.append(np.array(displacements[start_idx:end_idx], dtype=np.float32))
        return sequences

    def _standardize_sequences(self, data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Standardize sequences to a fixed length. Does NOT calculate thresholds anymore.
        Returns standardized sequences, labels (displacement scores), job_ids, video_indices.
        """
        all_sequences_list = data['sequences']
        all_labels_list = data['labels'] # Displacement scores
        all_job_ids_list = data['job_ids']
        video_indices_list = data.get('video_indices', [])

        if not all_sequences_list:
            logger.error("No sequences found to standardize.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        try:
            labels_np = np.array(all_labels_list, dtype=np.float32)
            labels_np = np.nan_to_num(labels_np)
            # Log stats about the displacement scores
            min_val, max_val = np.min(labels_np), np.max(labels_np)
            mean_val, median_val = np.mean(labels_np), np.median(labels_np)
            logger.info(f"Displacement Score stats (before standardization): Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}, Median={median_val:.2f}")
        except Exception as e:
            logger.error(f"Failed to convert labels to numpy array: {e}")
            raise ValueError("Invalid label data provided.") from e

        # --- Sequence Standardization ---
        num_features = getattr(self.config, 'num_joints', 6)
        seq_lengths = [seq.shape[0] for seq in all_sequences_list if isinstance(seq, np.ndarray) and seq.ndim == 2 and seq.shape[1] == num_features]
        if not seq_lengths:
            logger.error("No valid sequences found for standardization after initial filtering.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        target_length = getattr(self.config, 'min_sequence_length', int(np.median(seq_lengths)))
        target_length = max(target_length, 1)
        self.config.set_standardized_length(target_length) # Store the length used
        logger.info(f"Standardizing sequences to length={target_length}, features={num_features}")

        valid_std_seqs, valid_labels, valid_job_ids, valid_vid_indices = [], [], [], []
        skipped_count = 0
        for i, seq in enumerate(all_sequences_list):
            if not isinstance(seq, np.ndarray) or seq.ndim != 2 or seq.shape[1] != num_features:
                skipped_count += 1; continue
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            current_length = seq.shape[0]
            standardized_seq = np.zeros((target_length, num_features), dtype=np.float32)
            if current_length == target_length: standardized_seq = seq
            elif current_length < target_length: standardized_seq[:current_length, :] = seq
            else: standardized_seq = seq[-target_length:, :]

            valid_std_seqs.append(standardized_seq)
            valid_labels.append(all_labels_list[i]) # Keep displacement score
            valid_job_ids.append(all_job_ids_list[i])
            if video_indices_list and i < len(video_indices_list): valid_vid_indices.append(video_indices_list[i])
            elif video_indices_list: valid_vid_indices.append(-1)

        if skipped_count > 0: logger.warning(f"Skipped {skipped_count} invalid sequences during standardization.")
        if not valid_std_seqs:
            logger.error("No valid sequences remained after standardization.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        final_sequences = np.array(valid_std_seqs, dtype=np.float32)
        final_labels = np.array(valid_labels, dtype=np.float32) # Displacement scores
        final_job_ids = np.array(valid_job_ids, dtype=np.int64)
        final_video_indices = np.array(valid_vid_indices, dtype=np.int64) if valid_vid_indices else None

        logger.info(f"Standardization complete. Final shapes: X={final_sequences.shape}, y={final_labels.shape}")
        return final_sequences, final_labels, final_job_ids, final_video_indices


    def _create_data_loaders(self, data):
        """Create train and validation data loaders. No longer uses thresholds."""
        X, y, job_ids, video_indices = data # y contains displacement scores

        if len(X) == 0:
            logger.error("Cannot create data loaders with empty dataset (X).")
            raise ValueError("Cannot create data loaders from empty standardized data.")

        # --- Splitting Data ---
        try:
            stratify_on = job_ids # Stratify by job type
            unique_jobs, counts = np.unique(job_ids, return_counts=True)
            min_samples_per_job = np.min(counts) if len(counts)>0 else 0
            n_splits = 5 # Default k for StratifiedKFold or test_size=0.2
            if min_samples_per_job < n_splits and len(X) >= n_splits:
                logger.warning(f"Cannot stratify by job_id: Min samples per job ({min_samples_per_job}) < n_splits ({n_splits}). Using non-stratified split.")
                stratify_on = None # Fallback
            arrays_to_split = [X, y, job_ids]; indices_present = video_indices is not None and video_indices.size > 0
            if indices_present: arrays_to_split.append(video_indices)
            split_results = train_test_split(*arrays_to_split, test_size=0.2, stratify=stratify_on, random_state=42)
            X_train, X_val = split_results[0], split_results[1]; y_train, y_val = split_results[2], split_results[3]
            ids_train, ids_val = split_results[4], split_results[5]; vid_train, vid_val = (split_results[6], split_results[7]) if indices_present else (None, None)
        except ValueError as e:
            logger.error(f"Stratified split failed: {e}. Trying non-stratified split.")
            try:
                arrays_to_split_ns = [X, y, job_ids];
                if indices_present: arrays_to_split_ns.append(video_indices)
                split_results_ns = train_test_split(*arrays_to_split_ns, test_size=0.2, random_state=42)
                X_train, X_val = split_results_ns[0], split_results_ns[1]; y_train, y_val = split_results_ns[2], split_results_ns[3]
                ids_train, ids_val = split_results_ns[4], split_results_ns[5]; vid_train, vid_val = (split_results_ns[6], split_results_ns[7]) if indices_present else (None, None)
            except Exception as split_err:
                logger.error(f"Failed to split data even without stratification: {split_err}"); raise ValueError("Could not split data.") from split_err

        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        # Log split distribution based on job IDs
        self._log_split_distribution(y_train, ids_train, "Training (Before Aug)")
        self._log_split_distribution(y_val, ids_val, "Validation")

        # --- Data Augmentation (using displacement score y_train for potential balancing) ---
        if getattr(self.config, 'data_augmentation', False) and len(X_train) > 0: # Optional augmentation
            logger.info("Applying data augmentation to training set...")
            try:
                # Augmentation can still optionally balance based on the displacement score distribution
                X_aug, y_aug, ids_aug, vid_aug = self._augment_training_data(X_train, y_train, ids_train, vid_train)
                if X_aug is not None and len(X_aug) > 0:
                    X_train = np.vstack([X_train, X_aug]); y_train = np.concatenate([y_train, y_aug])
                    ids_train = np.concatenate([ids_train, ids_aug])
                    if vid_train is not None and vid_aug is not None: vid_train = np.concatenate([vid_train, vid_aug])
                    elif vid_aug is not None: vid_train = np.concatenate([np.full(len(X_train)-len(X_aug), -1, dtype=np.int64), vid_aug])
                    logger.info(f"After augmentation - Training samples: {len(X_train)}")
                    self._log_split_distribution(y_train, ids_train, "Training (Augmented)")
            except Exception as aug_err:
                logger.error(f"Error during data augmentation: {aug_err}. Proceeding without.")
                self._log_split_distribution(y_train, ids_train, "Training (Augmentation Failed)")

        # --- Create Datasets and DataLoaders ---
        try:
            train_dataset = MovementSequenceDataset(X_train, y_train, ids_train, vid_train if vid_train is not None else None)
            val_dataset = MovementSequenceDataset(X_val, y_val, ids_val, vid_val if vid_val is not None else None)
        except Exception as ds_err: logger.error(f"Error creating MovementSequenceDataset: {ds_err}"); raise ValueError("Failed to create PyTorch Datasets.") from ds_err

        batch_size = self.config.batch_size; num_workers = self.num_workers
        pin_memory = getattr(self.config, 'pin_memory', torch.cuda.is_available())
        persistent_workers = getattr(self.config, 'persistent_workers', True) and num_workers > 0

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=False, persistent_workers=persistent_workers)
        self.train_loader, self.val_loader = train_loader, val_loader
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return train_loader, val_loader

    def _log_split_distribution(self, labels, job_ids, split_name):
        """Helper function to log job distributions (label distribution removed)."""
        if len(labels) == 0: logger.info(f"{split_name} set is empty."); return
        job_ids_np = np.array(job_ids)
        total_count = len(labels)

        logger.info(f"{split_name} job distribution:")
        if len(job_ids_np) > 0:
            unique_jobs, counts = np.unique(job_ids_np, return_counts=True)
            for job_id, count in zip(unique_jobs, counts):
                job_name = self.config.job_categories[job_id] if 0 <= job_id < len(self.config.job_categories) else f"InvalidID({job_id})"
                logger.info(f"  Job {job_id} ({job_name}): {count} samples ({count/total_count*100:.1f}%)")
        else: logger.info(f"  No job IDs available for {split_name} set.")


    def _augment_training_data(self, X, y, job_ids, video_indices):
        """Augment training data. Can optionally balance based on displacement scores (y)."""
        if len(X) == 0 or not getattr(self.config, 'data_augmentation', False): # Defaulting augmentation to False
            return np.array([]), np.array([]), np.array([]), np.array([]) if video_indices is not None else None

        logger.info(f"Augmenting {len(X)} training samples...")
        noise_factor = getattr(self.config, 'noise_factor', 0.05)
        X_aug, y_aug, ids_aug = [], [], []
        vid_aug = [] if video_indices is not None else None

        # Example: Simple duplication or noise addition (not balancing based on y for now)
        num_to_add = int(len(X) * 0.5) # Augment 50%
        if num_to_add == 0: return np.array([]), np.array([]), np.array([]), np.array([]) if video_indices is not None else None

        np.random.seed(42)
        chosen_indices = np.random.choice(np.arange(len(X)), num_to_add, replace=True)

        for idx in chosen_indices:
            seq = X[idx].copy(); label = y[idx]; job_id = job_ids[idx]
            seq_std = np.std(seq); noise = np.random.normal(0, noise_factor * seq_std if seq_std > 1e-6 else noise_factor, seq.shape)
            aug_seq = seq + noise; aug_label = label # Keep original displacement score for augmented data
            aug_seq = np.nan_to_num(aug_seq)
            X_aug.append(aug_seq); y_aug.append(aug_label); ids_aug.append(job_id)
            if vid_aug is not None: vid_aug.append(-1) # Mark augmented

        if X_aug:
            X_aug_np = np.array(X_aug, dtype=X.dtype); y_aug_np = np.array(y_aug, dtype=y.dtype)
            ids_aug_np = np.array(ids_aug, dtype=job_ids.dtype); vid_aug_np = np.array(vid_aug, dtype=np.int64) if vid_aug is not None else None
            logger.info(f"Generated {len(X_aug_np)} augmented samples.")
            return X_aug_np, y_aug_np, ids_aug_np, vid_aug_np
        else:
            logger.info("No samples were augmented.")
            vid_return = np.array([], dtype=np.int64) if video_indices is not None else None
            return np.array([]), np.array([]), np.array([]), vid_return


    def prepare_data(self) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        Main function: collect, process, standardize, split. Returns loaders.
        """
        if self.train_loader and self.val_loader:
            logger.warning("Using existing data loaders.")
            return self.train_loader, self.val_loader

        logger.info("--- Starting Data Preparation Pipeline ---")
        train_loader, val_loader = None, None
        try:
            video_files = self._collect_video_files()
            if not video_files: return None, None

            processed_data = self._process_videos(video_files)
            if not processed_data or 'sequences' not in processed_data or not processed_data['sequences']:
                 logger.error("Failed to process videos or extract sequences/labels.")
                 return None, None

            # Standardize sequences (no threshold calculation here)
            standardized_data_tuple = self._standardize_sequences(processed_data)
            X, y, job_ids, video_indices = standardized_data_tuple # y contains displacement scores

            if X is None or y is None or len(X) == 0:
                 logger.error("Standardization failed or resulted in no data.")
                 return None, None

            # Create DataLoaders
            train_loader, val_loader = self._create_data_loaders((X, y, job_ids, video_indices))
            logger.info("--- Data Preparation Pipeline Complete ---")

        except (ValueError, FileNotFoundError) as data_err:
            logger.error(f"Data preparation failed: {data_err}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error during data preparation: {e}", exc_info=True)
            return None, None

        # Return loaders only
        return train_loader, val_loader