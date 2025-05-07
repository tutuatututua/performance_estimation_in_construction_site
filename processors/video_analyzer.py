# processors/video_analyzer.py

import torch
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
import time
import traceback
from typing import Optional, List, Dict, Tuple, Any, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import gc
import json # Added for loading model_info

# Import necessary project modules
from utils.model_loader import load_movement_classifier, load_pose_model
from utils.pose_extraction import (
    detect_upper_body_pose, normalize_joints,
    calculate_displacement, draw_poses, PoseMemory
)
# Import visualization functions, including the new direct score plot
from utils import visualization
from utils.visualization import create_direct_score_plot_image # Import the new function
# Import tracking utility if needed (currently not used directly here but might be relevant)
# from utils.tracking import filter_tracking_errors_and_correct

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """
    Analyzes videos using LSTM AE for reconstruction error (anomaly/familiarity)
    and direct displacement scores (magnitude), generating visualizations.
    """

    def __init__(self, config):
        """
        Initializes the VideoAnalyzer.

        Args:
            config: The configuration object containing paths and parameters.
        """
        self.config = config
        self.device = config.device
        logger.info(f"Initializing VideoAnalyzer on device: {self.device}")

        # --- Load Models and Configuration ---
        self.pose_model = self._load_pose_model()
        self.movement_model, self.model_info = self._load_movement_model()

        # Ensure models and info loaded correctly
        if self.movement_model is None or self.model_info is None:
            raise RuntimeError("Failed to load movement model or model info during VideoAnalyzer init.")

        # Extract model configuration details
        self.model_config = self.model_info.get('model_config', {})
        self.input_size = self.model_config.get('num_joints', self.model_config.get('input_size')) # Prefer num_joints if available
        self.sequence_length = self.model_info.get('standardized_sequence_length', self.model_config.get('sequence_length'))

        # --- Load Reconstruction Error Thresholds ---
        threshold_info = self.model_info.get('anomaly_thresholds')
        if threshold_info and threshold_info.get('source') not in ['calculation_failed', 'calculation_failed_or_no_data', 'missing (JSON load failed or file absent)', 'missing (key not in JSON)']:
            self.low_threshold = threshold_info.get('low_threshold')
            self.high_threshold = threshold_info.get('high_threshold')
            # Validate loaded thresholds
            if not isinstance(self.low_threshold, (int, float)) or not isinstance(self.high_threshold, (int, float)) or self.low_threshold >= self.high_threshold:
                logger.error(f"Invalid anomaly thresholds loaded from model_info.json: Low={self.low_threshold}, High={self.high_threshold}. Using defaults.")
                self.low_threshold = 0.2 # Default fallback
                self.high_threshold = 0.6 # Default fallback
                self.threshold_source = 'default_fallback_invalid_json'
            else:
                self.threshold_source = threshold_info.get('source', 'model_info')
                logger.info(f"Using RECONSTRUCTION ERROR thresholds from {self.threshold_source}: "
                            f"Low={self.low_threshold:.4f}, High={self.high_threshold:.4f}")
        else:
            logger.error("CRITICAL: Anomaly thresholds missing, invalid, or calculation failed in model_info.json.")
            self.low_threshold = 0.2 # Default fallback
            self.high_threshold = 0.6 # Default fallback
            self.threshold_source = 'default_fallback_missing'
            logger.warning(f"Using DEFAULT fallback anomaly thresholds: Low={self.low_threshold:.4f}, High={self.high_threshold:.4f}")

        # --- Load Direct Score Statistics ---
        direct_stats_info = self.model_info.get('direct_score_stats')
        if direct_stats_info and direct_stats_info.get('source') not in ['calculation_failed_or_no_data', 'calculation_failed']:
            self.direct_score_mean = direct_stats_info.get('mean')
            self.direct_score_q1 = direct_stats_info.get('q1')
            self.direct_score_q3 = direct_stats_info.get('q3')
            # Validate loaded stats
            if all(v is not None and np.isfinite(v) for v in [self.direct_score_mean, self.direct_score_q1, self.direct_score_q3]):
                logger.info(f"Using DIRECT SCORE stats from {direct_stats_info.get('source', 'model_info')}: "
                            f"Mean={self.direct_score_mean:.2f}, Q1={self.direct_score_q1:.2f}, Q3={self.direct_score_q3:.2f}")
            else:
                logger.warning("Direct score stats found in model_info.json but contain None/NaN values. Disabling direct score plot comparison.")
                self.direct_score_mean, self.direct_score_q1, self.direct_score_q3 = None, None, None
        else:
            logger.warning("Direct score statistics (mean, q1, q3) not found or calculation failed in model_info.json. Direct score plot comparison to training data disabled.")
            self.direct_score_mean, self.direct_score_q1, self.direct_score_q3 = None, None, None

        # --- Validate essential model parameters ---
        if self.input_size is None or self.sequence_length is None:
            missing_req = [k for k,v in {'input_size':self.input_size, 'sequence_length':self.sequence_length}.items() if v is None]
            logger.error(f"Essential model parameters missing after loading: {missing_req}")
            raise ValueError(f"Essential model parameters missing: {missing_req}")

        # --- Set up output directory paths ---
        self.output_dir = Path(config.output_dir)
        self.visualization_dir = self.output_dir / 'visualization' / 'movement_analysis'
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directories initialized: {self.output_dir}")

        # --- Initialize other components ---
        self.pose_memory = PoseMemory(memory_frames=15, min_keypoint_conf=self.config.yolo_min_keypoint_conf)
        self.current_analysis_data = None # Stores data for the video currently being processed
        # For storing last analysis results
        self.last_movement_data = None

        self._validate_model_config()

    def _load_pose_model(self):
        """Loads the YOLO pose estimation model using the utility function."""
        try:
            pose_model = load_pose_model(self.config.yolo_model_path, self.config)
            logger.info("YOLO pose detection model loaded successfully.")
            return pose_model
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load YOLO pose model: {e}", exc_info=True)
            raise # Re-raise the exception to halt initialization

    def _load_movement_model(self):
        """Loads the movement analysis model (LSTM Autoencoder) and its info."""
        try:
            movement_model, model_info = load_movement_classifier(self.config.model_save_path, self.config)
            if movement_model is None or model_info is None:
                raise RuntimeError("load_movement_classifier returned None for model or model_info.")
            logger.info(f"Movement analysis model loaded successfully from {self.config.model_save_path}")
            return movement_model, model_info
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load movement analysis model: {e}", exc_info=True)
            raise # Re-raise the exception

    def _validate_model_config(self):
        """Validates the loaded model configuration for essential AE parameters."""
        required_params = ['input_size', 'sequence_length', 'embedding_dim']
        # Use attributes already set in __init__
        loaded_params = {
            'input_size': self.input_size,
            'sequence_length': self.sequence_length,
            'embedding_dim': self.model_config.get('embedding_dim')
        }
        if any(loaded_params.get(param) is None for param in required_params):
            missing = [p for p in required_params if loaded_params.get(p) is None]
            raise ValueError(f"Incomplete model configuration for Autoencoder. Missing: {missing}")
        logger.info(f"Model parameters validated: input_size={self.input_size}, sequence_length={self.sequence_length}, embedding_dim={loaded_params['embedding_dim']}")

    def _prepare_sequence(self, sequence: np.ndarray) -> torch.Tensor:
        """
        Validates, standardizes (pads/truncates), and converts a sequence to a tensor.
        """
        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence, dtype=np.float32)

        # Ensure sequence is 2D and has the correct number of features
        if sequence.ndim != 2 or sequence.shape[1] != self.input_size:
            raise ValueError(f"Invalid sequence shape: {sequence.shape}. Expected (SeqLen, {self.input_size})")

        # Pad or truncate sequence to the model's required length
        current_length = sequence.shape[0]
        standardized_sequence = np.zeros((self.sequence_length, self.input_size), dtype=np.float32)
        if current_length == self.sequence_length:
            standardized_sequence = sequence
        elif current_length < self.sequence_length:
            # Pad at the end
            standardized_sequence[:current_length, :] = sequence
        else: # current_length > self.sequence_length
            # Truncate (take the last part)
            standardized_sequence = sequence[-self.sequence_length:, :]

        # Convert to tensor and add batch dimension
        return torch.from_numpy(standardized_sequence.astype(np.float32)).unsqueeze(0).to(self.device)

    def predict_batch(self, sequences: List[np.ndarray], job_type: Optional[str] = None) -> List[float]:
        """
        Predicts reconstruction errors for a batch of sequences using the AE model.
        Handles individual sequence errors gracefully.

        Args:
            sequences: List of numpy arrays, each representing a sequence (SeqLen, Features).
            job_type: Optional job type for conditional autoencoder.

        Returns:
            List of reconstruction errors (float), one per input sequence. Returns inf for failed sequences.
        """
        if not sequences:
            logger.debug("Empty sequence list passed to predict_batch")
            return []

        self.movement_model.eval() # Ensure model is in evaluation mode
        batch_errors = []

        # Process sequences individually to handle potential errors in preparation/prediction
        for i, seq_orig in enumerate(sequences):
            try:
                # Validate and prepare sequence (padding/truncation handled by _prepare_sequence)
                if not isinstance(seq_orig, np.ndarray):
                    seq = np.array(seq_orig, dtype=np.float32)
                else:
                    seq = seq_orig.astype(np.float32) # Ensure correct dtype

                # Check for non-finite values before standardization
                if not np.all(np.isfinite(seq)):
                    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
                    logger.debug(f"Replaced non-finite values in input sequence {i}")

                # Prepare sequence (standardize length, convert to tensor)
                # _prepare_sequence raises ValueError on shape mismatch
                x = self._prepare_sequence(seq)

                # Handle conditional input
                job_ids = self._get_job_ids_tensor(job_type) # Use helper method

                # Perform prediction
                with torch.no_grad():
                    reconstructed_seq = self.movement_model(x, job_ids)

                    # Validate output shape
                    if reconstructed_seq.shape != x.shape:
                        logger.error(f"Model output shape {reconstructed_seq.shape} doesn't match input shape {x.shape} for sequence {i}")
                        batch_errors.append(float('inf'))
                        continue

                    # Calculate reconstruction error (MSE)
                    error = torch.mean((reconstructed_seq - x) ** 2).item()

                    # Validate error value
                    if not np.isfinite(error):
                        logger.warning(f"Non-finite reconstruction error {error} calculated for sequence {i}")
                        batch_errors.append(float('inf'))
                    else:
                        batch_errors.append(error)

            except ValueError as ve: # Catch errors from _prepare_sequence
                 logger.error(f"Data error processing sequence {i}: {ve}")
                 batch_errors.append(float('inf'))
            except RuntimeError as rt_err: # Catch PyTorch runtime errors (like OOM)
                if "out of memory" in str(rt_err).lower():
                    logger.error(f"CUDA out of memory predicting sequence {i}. Trying to recover...")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    batch_errors.append(float('inf')) # Mark as failed
                else:
                    logger.error(f"Runtime error predicting sequence {i}: {rt_err}")
                    batch_errors.append(float('inf'))
            except Exception as e: # Catch any other unexpected errors
                logger.error(f"Unexpected error predicting sequence {i}: {e}", exc_info=True)
                batch_errors.append(float('inf'))

        # Ensure output list length matches input list length
        if len(batch_errors) != len(sequences):
            logger.warning(f"Batch error count mismatch: got {len(batch_errors)}, expected {len(sequences)}. Padding with inf.")
            while len(batch_errors) < len(sequences):
                batch_errors.append(float('inf'))
            batch_errors = batch_errors[:len(sequences)] # Truncate if somehow too many

        return batch_errors

    def predict_sequence(self, sequence: np.ndarray, job_type: Optional[str] = None) -> float:
        """
        Predicts the reconstruction error for a single sequence.
        Helper function, calls predict_batch internally.
        """
        results = self.predict_batch([sequence], job_type)
        return results[0] if results else float('inf')

    def _get_job_ids_tensor(self, job_type: Optional[str] = None) -> Optional[torch.Tensor]:
        """Gets the job IDs tensor for conditional Autoencoder, if applicable."""
        if not self.model_config.get('conditional'):
            return None # Not a conditional model
        if job_type is None:
             logger.debug("Job type not provided for conditional AE. Proceeding without condition.")
             return None
        if not hasattr(self.config, 'job_categories') or job_type not in self.config.job_categories:
            logger.warning(f"Job type '{job_type}' not found in config.job_categories. Proceeding without condition.")
            return None
        try:
            job_idx = self.config.job_categories.index(job_type)
            return torch.tensor([job_idx], device=self.device, dtype=torch.long)
        except ValueError: # Should not happen due to check above, but included for safety
            logger.warning(f"Value error getting index for job type '{job_type}'.")
            return None

    def analyze_video(self, video_path, job_type=None, output_path=None, skip_frames=0, fps_override=None):
        """
        Analyzes a complete video file, generates annotated video with plots.

        Args:
            video_path (str or Path): Path to the input video file.
            job_type (str, optional): Job category for context. Defaults to None.
            output_path (str or Path, optional): Path to save the output video. Defaults to None (saves in output_dir).
            skip_frames (int): Number of frames to skip between processing (if > 0, overrides sample_rate). Defaults to 0.
            fps_override (float, optional): Override the video's FPS for analysis and output. Defaults to None.

        Returns:
            str: Path to the generated analyzed video, or None if analysis failed.
        """
        video_path = Path(video_path)
        logger.info(f"Starting analysis for video: {video_path.name}")
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return None

        try:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_width == 0 or frame_height == 0:
                logger.error(f"Invalid frame dimensions obtained from video: {frame_width}x{frame_height}")
                cap.release()
                return None

            # --- STEP 1: Process frames, extract poses and raw displacements ---
            # Store result in self.current_analysis_data for use in _create_output_video
            self.current_analysis_data, frame_poses = self._process_video_frames(
                cap, video_path, job_type, skip_frames, fps_override
            )
            if not self.current_analysis_data:
                logger.warning(f"No valid movement data extracted from {video_path.name}. Output video will not be created.")
                cap.release()
                return None

            # --- STEP 2: Create output video using stored data ---
            # This step now calculates reconstruction errors and direct scores, interpolates, and generates frames with plots
            output_video_path = self._create_output_video(
                cap=cap, # Pass cap primarily for FPS info
                frame_width=frame_width,
                frame_height=frame_height,
                output_path=output_path,
                analysis_data=self.current_analysis_data, # Use self attribute
                frame_poses=frame_poses, # Pass stored raw poses
                job_type=job_type,
                fps_override=fps_override # Pass override for output FPS consistency
            )

            # --- STEP 3: Generate standalone sequence plot (reconstruction error) ---
            if output_video_path:
                # Use the final analysis_data (with interpolated values) for the plot
                self._generate_and_register_sequence_plot(video_path, self.current_analysis_data, job_type, frame_width)
                direct_score_plot_path = self._generate_direct_score_plot(video_path, self.current_analysis_data, job_type, frame_width)
                if direct_score_plot_path:
                    logger.info(f"Direct score visualization generated: {direct_score_plot_path}")

            return str(output_video_path) if output_video_path else None

        except Exception as e:
            logger.error(f"Error processing video {video_path.name}: {e}", exc_info=True)
            return None
        finally:
            # Ensure resources are released
            if cap.isOpened(): cap.release()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            # Reset current analysis data for the next video
            self.current_analysis_data = None

    def _get_output_path(self, video_path: Path, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Determines the output path for the analyzed video."""
        video_path = Path(video_path) # Ensure Path object
        if output_path is None:
            # Default output path construction
            output_path = self.output_dir / f"{video_path.stem}_analyzed.mp4"
        else:
            output_path = Path(output_path) # Convert string to Path if necessary
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output video will be saved to: {output_path}")
        return output_path

    def _process_video_frames(self, cap, video_path, job_type=None, skip_frames=0, fps_override=None):
        """
        Processes video frames to detect poses and calculate raw displacements.
        Initializes data structures to store results per person.

        Returns:
            Tuple[Dict, Dict]: (person_movement_data, frame_poses)
        """
        in_fps = cap.get(cv2.CAP_PROP_FPS)
        out_fps = fps_override if fps_override else in_fps
        if out_fps <= 0: logger.warning(f"Invalid input FPS ({in_fps}). Using 30."); out_fps = 30.0

        frame_interval = self._get_frame_interval(in_fps, skip_frames)
        logger.info(f"Analyzing every {frame_interval} frames (Input FPS: {in_fps:.2f}, Target Rate: ~{out_fps / frame_interval:.1f}Hz).")

        # Get original frame dimensions for pose scaling
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_frame_size = (orig_width, orig_height)
        # Get target dimensions for pose detection resizing
        target_width, target_height = self.config.frame_resize
        logger.info(f"Original frame size: {orig_width}x{orig_height}")
        logger.info(f"Detection frame size: {target_width}x{target_height}")

        # Data structures - include 'direct_scores' list
        person_movement_data = defaultdict(lambda: {'times': [], 'reconstruction_errors': [], 'anomaly_levels': [], 'poses': [], 'raw_displacements': [], 'direct_scores': []})
        frame_poses = defaultdict(list) # Stores raw poses per frame number for drawing later
        last_poses_normalized = {} # Stores last normalized pose per person_id for displacement calculation

        # Progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        processed_frame_count = 0
        pbar = tqdm(total=total_frames, desc=f"Processing Frames [{video_path.stem}]", unit="frame", leave=False)
        start_time = time.time()
        # Memory management
        frames_since_cleanup = 0
        cleanup_interval = 100 # Clean up GPU memory periodically

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break # End of video
                frame_count += 1
                pbar.update(1)

                # Skip frames based on interval
                if (frame_count - 1) % frame_interval != 0: continue

                # Resize frame for detection if needed
                try:
                    if frame.shape[1] != target_width or frame.shape[0] != target_height:
                        frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    else:
                        frame_resized = frame
                except Exception as resize_err:
                    logger.warning(f"Could not resize frame {frame_count}: {resize_err}. Skipping frame.")
                    continue

                processed_frame_count += 1
                frames_since_cleanup += 1
                current_time = frame_count / out_fps # Use output FPS for consistent time

                # Detect poses, passing original size for correct scaling if needed by pose extraction
                poses_raw = self._detect_poses(
                    frame_resized,
                    frame_count,
                    original_frame_size=original_frame_size
                )
                # Store raw poses detected in this frame (keyed by frame number)
                frame_poses[frame_count] = poses_raw

                # Analyze movement if poses were detected
                if poses_raw:
                    try:
                        # This function calculates displacements and stores placeholders
                        self._analyze_movement_in_frame(
                            poses_raw=poses_raw,
                            person_movement_data=person_movement_data,
                            last_poses_normalized=last_poses_normalized,
                            current_time=current_time,
                            job_type=job_type,
                            frame_count=frame_count
                        )
                    except Exception as analyze_err:
                        logger.warning(f"Error analyzing movement in frame {frame_count}: {analyze_err}")

                # Regular memory cleanup
                if frames_since_cleanup >= cleanup_interval:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    frames_since_cleanup = 0

        except Exception as e:
            logger.error(f"Error during frame processing loop: {str(e)}", exc_info=True)
            logger.info("Attempting to continue with partial data...")
        finally:
            pbar.close()
            elapsed_time = time.time() - start_time
            logger.info(f"Frame processing finished. Time: {elapsed_time:.2f}s. Processed {processed_frame_count}/{total_frames} frames.")
            # Final cleanup
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return person_movement_data, frame_poses

    def _generate_and_register_sequence_plot(self, video_path, analysis_data, job_type=None, plot_width=1200, custom_output_dir=None):
        """
        Generates and saves the standalone reconstruction error sequence plot WITHOUT dot markers.
        Now with option for custom output directory.
        """
        try:
            # --- Initial setup (figsize, data collection) ---
            plt.close('all')
            fig_width_inches = max(10, plot_width / 100); fig, ax = plt.subplots(figsize=(fig_width_inches, 6))
            has_plot_data = False; max_error_value = 0; min_error_value = float('inf'); all_times = []; all_errors = []
            for person_id, data in analysis_data.items():
                if 'times' in data and 'reconstruction_errors' in data and len(data['times']) > 0:
                    times = np.array(data['times']); errors = np.array(data['reconstruction_errors']); valid_mask = np.isfinite(errors) & (errors > 0)
                    if np.any(valid_mask):
                        valid_times = times[valid_mask]; valid_errors = errors[valid_mask]; all_times.extend(valid_times); all_errors.extend(valid_errors)
                        if len(valid_errors) > 0: max_error_value = max(max_error_value, np.max(valid_errors)); min_error_value = min(min_error_value, np.min(valid_errors))
            if not all_errors: logger.warning(f"No valid error data for plot {video_path.stem}."); plt.close(fig); return None
            has_plot_data = True

            # --- Y-axis scaling logic ---
            consistent_scale_baseline = 0.6
            if self.high_threshold is not None and np.isfinite(self.high_threshold): # Check threshold validity
                if max_error_value <= self.high_threshold * 1.5: y_max = max(self.high_threshold * 1.5, consistent_scale_baseline)
                else: y_max = max(max_error_value * 1.2, self.high_threshold * 1.5)
            else: # Fallback if threshold is missing or invalid
                logger.warning(f"High threshold ({self.high_threshold}) is invalid or missing for scaling plot {video_path.stem}. Using default scaling.")
                if max_error_value <= consistent_scale_baseline * 1.5: y_max = consistent_scale_baseline
                else: y_max = max_error_value * 1.2
            y_max = max(y_max, 0.1) # Ensure minimum height

            # --- Plotting loop for each person ---
            color_palette = plt.cm.tab10.colors
            for i, (person_id, data) in enumerate(analysis_data.items()):
                if 'times' in data and 'reconstruction_errors' in data and len(data['times']) > 0:
                    times = np.array(data['times']); errors = np.array(data['reconstruction_errors']); valid_mask = np.isfinite(errors) & (errors > 0)
                    if np.any(valid_mask):
                        valid_times = times[valid_mask]; valid_errors = errors[valid_mask]
                        if len(valid_times) > 1:
                            person_color = color_palette[i % len(color_palette)]
                            # Plot the continuous line ONLY (marker=None)
                            ax.plot(valid_times, valid_errors,
                                color=person_color,
                                linewidth=1.5,
                                label=f'Person {person_id}',
                                alpha=0.7,
                                marker=None) # <--- ENSURE THIS IS PRESENT

            # --- Plot thresholds, title, labels, grid, limits ---
            if self.low_threshold is not None and np.isfinite(self.low_threshold): ax.axhline(y=self.low_threshold, color='green', linestyle='--', alpha=0.7, label=f'Low Thresh ({self.low_threshold:.2f})')
            if self.high_threshold is not None and np.isfinite(self.high_threshold): ax.axhline(y=self.high_threshold, color='red', linestyle='--', alpha=0.7, label=f'High Thresh ({self.high_threshold:.2f})')
            if has_plot_data:
                summary_text = f"Analysis Summary ({video_path.stem}):\nJob: {job_type if job_type else 'N/A'}, Persons: {len(analysis_data)}\nData points: {len(all_errors)}, Max Error: {max_error_value:.3f}"
                ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9, va='top', ha='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            ax.set_title(f'Reconstruction Error Analysis - {video_path.stem}', fontsize=14, fontweight='bold'); ax.set_xlabel('Time (seconds)', fontsize=12); ax.set_ylabel('Reconstruction Error', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--'); ax.set_ylim(bottom=0, top=y_max)
            if all_times: x_min = min(all_times); x_max = max(all_times); x_padding = (x_max - x_min) * 0.05; ax.set_xlim(left=max(0, x_min - x_padding), right=x_max + x_padding)

            # --- Output path determination ---
            job_suffix = f"_{job_type}" if job_type else ""
            if custom_output_dir is not None:
                output_dir = Path(custom_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"movement_sequence_analysis_{video_path.stem}{job_suffix}.png"
                logger.info(f"Saving reconstruction error plot to custom directory: {output_dir}")
            else:
                # Fall back to the default visualization directory (ensure self.visualization_dir is defined)
                if not hasattr(self, 'visualization_dir'):
                     self.visualization_dir = self.output_dir / 'visualization' / 'movement_analysis' # Define it if missing
                self.visualization_dir.mkdir(parents=True, exist_ok=True)
                output_path = self.visualization_dir / f"movement_sequence_analysis_{video_path.stem}{job_suffix}.png"


            # --- Legend and saving ---
            handles, labels = ax.get_legend_handles_labels(); threshold_indices = [i for i, label in enumerate(labels) if 'Thresh' in label]; person_indices = [i for i, label in enumerate(labels) if 'Person' in label][:3]
            selected_indices = threshold_indices + person_indices; ax.legend([handles[i] for i in selected_indices], [labels[i] for i in selected_indices], loc='upper right', fontsize=10)
            plt.tight_layout(); plt.savefig(output_path, dpi=120); plt.close(fig); logger.info(f"Sequence analysis plot saved to: {output_path}"); return str(output_path)
        except Exception as e: logger.error(f"Error creating sequence plot: {e}", exc_info=True); plt.close('all'); return None
        
    def _get_frame_interval(self, fps: float, skip_frames: int) -> int:
        """Calculates the frame interval based on skip_frames or sample_rate."""
        if skip_frames > 0:
            return skip_frames + 1
        else:
            # Calculate interval based on sample rate, ensuring it's at least 1
            interval = max(1, int(round(fps / self.config.sample_rate))) if self.config.sample_rate > 0 else 1
            return interval

    def _detect_poses(self, frame: np.ndarray, frame_count: int, original_frame_size=None) -> List[Tuple[int, Dict[str, np.ndarray]]]:
        """
        Detects poses in the frame using the YOLO pose model.
        """
        try:
            # Use tracking if enabled in config
            if self.config.use_tracking:
                tracker_args = {
                    'conf': self.config.yolo_conf_threshold,
                    'iou': self.config.yolo_iou_threshold,
                    'persist': True, # Keep tracks between frames
                    'tracker': self.config.tracker_config, # e.g., 'bytetrack.yaml'
                    'verbose': False
                }
                results = self.pose_model.track(frame, **tracker_args)
            else:
                # Use standard detection if tracking is disabled
                results = self.pose_model(frame, conf=self.config.yolo_conf_threshold,
                                        iou=self.config.yolo_iou_threshold, verbose=False)

            # Handle potential different return types from YOLO versions/modes
            if isinstance(results, list) and results:
                yolo_results = results[0] # Typically detection returns a list
            elif hasattr(results, 'boxes'): # Check if it's a Results object (tracking often returns this)
                yolo_results = results
            else:
                logger.debug(f"Frame {frame_count}: Unexpected YOLO result type: {type(results)}")
                yolo_results = None

            # Extract upper body poses using the utility function
            poses_raw = detect_upper_body_pose(
                frame,
                yolo_results,
                self.pose_memory,
                self.config,
                original_frame_size=original_frame_size # Pass original size for scaling
            )
            logger.debug(f"Frame {frame_count}: Detected {len(poses_raw)} poses.")
            return poses_raw
        except Exception as pose_err:
            logger.warning(f"Error during pose detection/tracking at frame {frame_count}: {pose_err}", exc_info=True)
            return [] # Return empty list on error

    def _analyze_movement_in_frame(self, poses_raw: List[Tuple[int, Dict[str, np.ndarray]]],
                                 person_movement_data: Dict[int, Dict[str, List[Any]]],
                                 last_poses_normalized: Dict[int, Dict[str, np.ndarray]],
                                 current_time: float, job_type: Optional[str], frame_count: int):
        """
        Analyzes movement (displacement) for each detected person in the frame
        and updates the person_movement_data dictionary with raw displacements
        and placeholders for scores/errors.
        """
        current_person_ids = set()
        for person_id, current_pose_raw in poses_raw:
            current_person_ids.add(person_id)
            # Normalize the raw pose relative to a reference point (e.g., shoulder center)
            normalized_pose = normalize_joints(current_pose_raw)
            if normalized_pose is None:
                logger.debug(f"Frame {frame_count}, Person {person_id}: Skipping due to failed normalization.")
                continue

            # Initialize data structure if this is the first time seeing this person
            if person_id not in person_movement_data:
                 person_movement_data[person_id] = {'times': [], 'reconstruction_errors': [], 'anomaly_levels': [], 'poses': [], 'raw_displacements': [], 'direct_scores': []}

            # Calculate displacement if we have a previous pose for this person
            last_normalized = last_poses_normalized.get(person_id)
            if last_normalized:
                 # Calculate displacement vector between current and last normalized pose
                 raw_displacement_vector = calculate_displacement(normalized_pose, last_normalized)
                 if raw_displacement_vector is not None:
                     # Ensure displacement is valid before storing
                     disp_array = np.array(raw_displacement_vector, dtype=np.float32)
                     if np.all(np.isfinite(disp_array)) and len(disp_array) == self.input_size:
                         person_movement_data[person_id]['raw_displacements'].append(disp_array)
                     else:
                          logger.debug(f"Frame {frame_count}, Person {person_id}: Skipping invalid displacement vector (Shape: {disp_array.shape}, Finite: {np.all(np.isfinite(disp_array))})")
                 # else: logger.debug(f"Frame {frame_count}, Person {person_id}: Displacement calculation returned None.")

            # Append data points for this frame
            person_movement_data[person_id]['times'].append(current_time)
            person_movement_data[person_id]['poses'].append(normalized_pose) # Store normalized pose
            # Append placeholders - reconstruction error, anomaly level, and direct score will be calculated later
            person_movement_data[person_id]['reconstruction_errors'].append(0.0) # Placeholder
            person_movement_data[person_id]['anomaly_levels'].append("UNKNOWN") # Placeholder
            person_movement_data[person_id]['direct_scores'].append(0.0) # Placeholder

            # Update the last known normalized pose for this person
            last_poses_normalized[person_id] = normalized_pose

        # Clean up data for persons who disappeared from the frame
        disappeared_ids = set(last_poses_normalized.keys()) - current_person_ids
        for person_id in disappeared_ids:
            if person_id in last_poses_normalized:
                del last_poses_normalized[person_id]
                # logger.debug(f"Removed disappeared person {person_id} from last_poses_normalized.")

    def _create_output_video(self, cap, frame_width, frame_height, output_path,
                       analysis_data, frame_poses, job_type=None, fps_override=None):
        """
        Creates the output video by:
        1. Calculating reconstruction errors and direct scores for all sequences.
        2. Interpolating missing values.
        3. Iterating through frames, drawing poses, generating plots, and writing to the output file.
        
        Args:
            cap: OpenCV video capture object
            frame_width: Width of video frames
            frame_height: Height of video frames
            output_path: Path to save the output video
            analysis_data: Dictionary containing movement data per person
            frame_poses: Dictionary mapping frame numbers to detected poses
            job_type: Optional job type for context
            fps_override: Optional FPS override
            
        Returns:
            Path: Path to the generated output video, or None if failed
        """
        # --- Setup Output Path and Video Writer ---
        if output_path is None:
            video_file_name = getattr(cap, 'filename', None)
            video_stem = Path(video_file_name).stem if video_file_name else "output"
            output_path = self.output_dir / f"{video_stem}_analyzed.mp4"
        else: 
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output video will be saved to: {output_path}")

        out_fps = fps_override if fps_override else cap.get(cv2.CAP_PROP_FPS)
        if out_fps <= 0: 
            logger.warning(f"Invalid video FPS ({out_fps}). Using 30.")
            out_fps = 30.0

        plot_height = 400  # Height for EACH plot
        output_video_width = frame_width
        # Adjust height for THREE sections: frame + recon_plot + direct_score_plot
        output_video_height = frame_height + plot_height + plot_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(str(output_path), fourcc, out_fps, (output_video_width, output_video_height))
        if not out.isOpened(): 
            logger.error(f"Failed to create output video writer for {output_path}")
            return None

        try:
            # --- STEP 1: Calculate Reconstruction Errors ---
            logger.info("Calculating reconstruction errors for all sequences...")
            for person_id, data in tqdm(analysis_data.items(), desc="Analyzing AE Sequences", leave=False):
                # Skip if not enough displacements to form a sequence
                if len(data.get('raw_displacements',[])) < self.sequence_length: 
                    continue

                # Define stride for overlapping sequences
                stride = max(1, self.sequence_length // 4)
                num_displacements = len(data['raw_displacements'])
                sequences_to_predict = []
                original_indices = []  # To map predictions back to the correct time index

                # Create sequences using a sliding window
                for i in range(0, num_displacements - self.sequence_length + 1, stride):
                    current_sequence_np = np.array(data['raw_displacements'][i : i + self.sequence_length], dtype=np.float32)
                    # Validate sequence shape before adding
                    if current_sequence_np.shape == (self.sequence_length, self.input_size):
                        sequences_to_predict.append(current_sequence_np)
                        # Store the time index corresponding to the *end* of this sequence window
                        time_idx = i + self.sequence_length - 1
                        original_indices.append({'person_id': person_id, 'time_idx': time_idx})
                    else:
                        logger.debug(f"Skipping AE sequence for person {person_id} at index {i}: Incorrect shape {current_sequence_np.shape}")

                # Predict errors for the collected sequences in a batch
                if sequences_to_predict:
                    predicted_errors = self.predict_batch(sequences_to_predict, job_type)

                    # Assign calculated errors back to the correct time index in analysis_data
                    for k, error in enumerate(predicted_errors):
                        index_info = original_indices[k] if k < len(original_indices) else None
                        if index_info and np.isfinite(error):
                            pid = index_info['person_id']
                            t_idx = index_info['time_idx']
                            # Ensure indices are valid before assignment
                            if pid in analysis_data and t_idx < len(analysis_data[pid]['reconstruction_errors']):
                                analysis_data[pid]['reconstruction_errors'][t_idx] = error
                                # Classify Anomaly Level based on error and thresholds
                                anomaly_level = "MEDIUM"  # Default
                                if error < self.low_threshold: 
                                    anomaly_level = "LOW"
                                elif error > self.high_threshold: 
                                    anomaly_level = "HIGH"
                                # Assign the level
                                if t_idx < len(analysis_data[pid]['anomaly_levels']):
                                    analysis_data[pid]['anomaly_levels'][t_idx] = anomaly_level
                        elif not np.isfinite(error):
                            logger.warning(f"Skipping non-finite prediction error: {error}")

            # --- STEP 2: Calculate Direct Movement Scores ---
            logger.info("Calculating direct movement scores for all sequences...")
            # Use the same RMS logic as in data_processor/evaluate_methods
            direct_score_scale_factor = 5.0  # Match the scale factor used elsewhere
            for person_id, data in tqdm(analysis_data.items(), desc="Analyzing Direct Scores", leave=False):
                if len(data.get('raw_displacements',[])) < self.sequence_length: 
                    continue

                stride = max(1, self.sequence_length // 4)
                num_displacements = len(data['raw_displacements'])

                # Iterate through sequences again to calculate direct score
                for i in range(0, num_displacements - self.sequence_length + 1, stride):
                    current_sequence_np = np.array(data['raw_displacements'][i : i + self.sequence_length], dtype=np.float32)
                    if current_sequence_np.shape != (self.sequence_length, self.input_size): 
                        continue

                    # Calculate direct score for this sequence
                    direct_score = 0.0
                    if current_sequence_np.size > 0:
                        try:
                            # Calculate RMS per joint, then max RMS
                            joint_rms = np.sqrt(np.mean(np.square(current_sequence_np), axis=0))
                            max_rms = np.max(joint_rms) if joint_rms.size > 0 else 0.0
                            movement_score = max(0.0, max_rms * direct_score_scale_factor)
                            direct_score = min(100.0, movement_score)  # Cap score at 100
                        except Exception as score_err:
                            logger.warning(f"Could not calc direct score for person {person_id} seq {i}: {score_err}")
                            direct_score = 0.0  # Default to 0 on error

                    # Store the score at the time index corresponding to the end of the sequence window
                    time_idx = i + self.sequence_length - 1
                    if person_id in analysis_data and time_idx < len(analysis_data[person_id]['direct_scores']):
                        analysis_data[person_id]['direct_scores'][time_idx] = direct_score

            # --- STEP 3: Interpolate Missing Errors AND Direct Scores ---
            logger.info("Interpolating missing error and score values...")
            for person_id, data in analysis_data.items():
                times = np.array(data.get('times', []))
                if len(times) == 0: 
                    continue  # Skip if no time data for this person

                # Interpolate reconstruction errors
                errors = np.array(data.get('reconstruction_errors', []))
                # Find indices where error was calculated (not the initial 0.0 placeholder) and is finite
                valid_indices_err = np.where((errors != 0.0) & np.isfinite(errors))[0]
                if len(valid_indices_err) > 1:  # Need at least two points to interpolate
                    valid_times_err = times[valid_indices_err]
                    valid_errors = errors[valid_indices_err]
                    # Interpolate over all original time points
                    interp_errors = np.interp(times, valid_times_err, valid_errors, left=valid_errors[0], right=valid_errors[-1])
                    data['reconstruction_errors'] = interp_errors.tolist()
                    # Re-assign anomaly levels based on interpolated errors
                    new_levels = []
                    for err_val in interp_errors:
                        if err_val < self.low_threshold: 
                            new_levels.append("LOW")
                        elif err_val > self.high_threshold: 
                            new_levels.append("HIGH")
                        else: 
                            new_levels.append("MEDIUM")
                    data['anomaly_levels'] = new_levels
                elif len(valid_indices_err) == 1:  # If only one valid point, fill with that value
                    fill_value = errors[valid_indices_err[0]]
                    fill_level = "MEDIUM"  # Determine level based on the single valid error
                    if fill_value < self.low_threshold: 
                        fill_level = "LOW"
                    elif fill_value > self.high_threshold: 
                        fill_level = "HIGH"
                    data['reconstruction_errors'] = [fill_value] * len(times)
                    data['anomaly_levels'] = [fill_level] * len(times)
                # If no valid points, errors and levels remain placeholders (0.0 and UNKNOWN)

                # Interpolate direct scores
                direct_scores = np.array(data.get('direct_scores', []))
                # Find indices where score was calculated (not the initial 0.0 placeholder) and is finite
                valid_indices_score = np.where((direct_scores != 0.0) & np.isfinite(direct_scores))[0]
                if len(valid_indices_score) > 1:  # Need at least two points
                    valid_times_score = times[valid_indices_score]
                    valid_scores = direct_scores[valid_indices_score]
                    # Interpolate over all original time points
                    interp_scores = np.interp(times, valid_times_score, valid_scores, left=valid_scores[0], right=valid_scores[-1])
                    data['direct_scores'] = interp_scores.tolist()
                elif len(valid_indices_score) == 1:  # If only one valid point, fill
                    fill_score = direct_scores[valid_indices_score[0]]
                    data['direct_scores'] = [fill_score] * len(times)
                # If no valid points, scores remain placeholders (0.0)

            # --- Extra STEP: Calculate consistent direct score scaling ---
            direct_score_y_max = None
            try:
                # Calculate a consistent y_max for direct score plots
                all_scores = []
                for person_id, data in analysis_data.items():
                    scores = data.get('direct_scores', [])
                    valid_scores = [s for s in scores if np.isfinite(s) and s > 0]
                    all_scores.extend(valid_scores)
                
                if all_scores:
                    # Use percentile to avoid outliers skewing the scale
                    score_max = np.percentile(all_scores, 99.5) if len(all_scores) > 10 else max(all_scores)
                    
                    # Use training data stats if available for consistency
                    if self.direct_score_q3 is not None and np.isfinite(self.direct_score_q3):
                        direct_score_y_max = max(score_max * 1.3, self.direct_score_q3 * 2.0)
                    else:
                        direct_score_y_max = score_max * 1.5
                        
                    # Set minimum and maximum height
                    direct_score_y_max = max(direct_score_y_max, 5.0)
                    direct_score_y_max = min(direct_score_y_max, 200.0)
                    
                    # Override if data exceeds maximum
                    if max(all_scores) > 200.0:
                        direct_score_y_max = max(all_scores) * 1.2
                        
                    logger.info(f"Calculated consistent direct score y_max={direct_score_y_max:.2f}")
                else:
                    direct_score_y_max = 100.0
                    logger.info(f"No valid direct scores found, using default y_max={direct_score_y_max}")
            except Exception as scale_err:
                direct_score_y_max = 100.0
                logger.warning(f"Error calculating direct score scale: {scale_err}. Using default y_max={direct_score_y_max}")

            # --- STEP 4: Create Frames with Visualization ---
            logger.info("Generating output video frames...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video capture
            frame_count_out = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar_out = tqdm(total=total_frames, desc="Creating Output Video", unit="frame")

            # Pre-get original frame size for draw_poses consistency
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_frame_size = (orig_width, orig_height)

            while True:
                ret, frame = cap.read()
                if not ret: 
                    break  # End of video
                frame_count_out += 1
                pbar_out.update(1)

                # Get poses stored for this frame number
                poses_for_frame = frame_poses.get(frame_count_out, [])
                # Draw poses onto the current frame
                frame_with_poses = draw_poses(frame, poses_for_frame, self.pose_memory, original_frame_size=original_frame_size)

                # Get current time in video
                current_time = frame_count_out / out_fps

                # --- Generate Plot 1: Reconstruction Error ---
                try:
                    # Pass the analysis data containing reconstruction errors and levels
                    recon_plot_img = visualization.create_movement_plot_classification(
                        movement_data=analysis_data,  # Contains recon_errors and levels
                        current_time=current_time, width=frame_width, job_type=job_type,
                        classifier=self, config=self.config  # Pass self to access thresholds
                    )
                except Exception as plot_err:
                    logger.error(f"Error creating reconstruction plot for frame {frame_count_out}: {plot_err}")
                    # Create a fallback error image
                    recon_plot_img = np.ones((plot_height, frame_width, 3), dtype=np.uint8) * 220
                    cv2.putText(recon_plot_img, f"Recon Plot Error", (20, plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                # --- Generate Plot 2: Direct Movement Score ---
                try:
                    direct_score_plot_img = visualization.create_direct_score_plot_image(
                        direct_score_data=analysis_data,  # Contains direct_scores
                        current_time=current_time, width=frame_width,
                        train_mean=self.direct_score_mean,  # Pass saved stats
                        train_q1=self.direct_score_q1,
                        train_q3=self.direct_score_q3,
                        job_type=job_type,
                        y_max=direct_score_y_max  # Pass consistent y-axis limit
                    )
                except Exception as plot_err:
                    logger.error(f"Error creating direct score plot for frame {frame_count_out}: {plot_err}")
                    # Create a fallback error image
                    direct_score_plot_img = np.ones((plot_height, frame_width, 3), dtype=np.uint8) * 220
                    cv2.putText(direct_score_plot_img, f"Direct Score Plot Error", (20, plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                # --- Combine frame and plots ---
                # Ensure frame dimensions match expected output video frame height
                if frame_with_poses.shape[1] != frame_width or frame_with_poses.shape[0] != frame_height:
                    frame_display = cv2.resize(frame_with_poses, (frame_width, frame_height))
                else: 
                    frame_display = frame_with_poses

                # Ensure plot dimensions match target plot_height
                if recon_plot_img is None or recon_plot_img.shape[1] != frame_width or recon_plot_img.shape[0] != plot_height:
                    recon_plot_display = cv2.resize(recon_plot_img if recon_plot_img is not None else np.zeros((plot_height, frame_width, 3), dtype=np.uint8), (frame_width, plot_height))
                else: 
                    recon_plot_display = recon_plot_img

                # Resize direct score plot
                if direct_score_plot_img is None or direct_score_plot_img.shape[1] != frame_width or direct_score_plot_img.shape[0] != plot_height:
                    direct_score_plot_display = cv2.resize(direct_score_plot_img if direct_score_plot_img is not None else np.zeros((plot_height, frame_width, 3), dtype=np.uint8), (frame_width, plot_height))
                else: 
                    direct_score_plot_display = direct_score_plot_img

                # Stack all three parts vertically: Frame, Recon Plot, Direct Score Plot
                try:
                    combined_frame = np.vstack((frame_display, recon_plot_display, direct_score_plot_display))
                    # Write the combined frame to the output video
                    out.write(combined_frame)
                except Exception as combine_err:
                    logger.error(f"Error combining frame and plots: {combine_err}")
                    logger.error(f"Shapes - Frame: {frame_display.shape}, ReconPlot: {recon_plot_display.shape}, DirectPlot: {direct_score_plot_display.shape}")
                    # Fallback: Write frame with blank plots below it
                    try:
                        blank_plots = np.zeros((plot_height * 2, frame_width, 3), dtype=np.uint8)  # Combined height for two plots
                        fallback_frame = np.vstack((frame_display, blank_plots))
                        out.write(fallback_frame)
                    except: 
                        logger.error("Failed to write even fallback frame")

                # Periodic memory cleanup during frame generation
                if frame_count_out % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available(): 
                        torch.cuda.empty_cache()

            pbar_out.close()
            out.release()  # Release the video writer
            logger.info(f"Analysis video created successfully: {output_path}")
            
            # Store movement data for later access by analyze.py
            self.last_movement_data = analysis_data
            
            return output_path  # Return the path to the generated video

        except Exception as e:
            logger.error(f"Error creating output video: {e}", exc_info=True)
            if out.isOpened(): 
                out.release()  # Ensure release on error
            return None  # Indicate failure
        finally:
            # Ensure release even if errors occurred before this point
            if out.isOpened(): 
                out.release()

    # --- Helper methods (likely unchanged and potentially unused now) ---
    def _combine_frames(self, frame: np.ndarray, plot_img: np.ndarray, output_width: Optional[int] = None, output_height: Optional[int] = None) -> np.ndarray:
        """Combines the original frame with the movement plot."""
        # This function might need updating if used elsewhere, as it only handles one plot.
        # Currently seems unused by the main flow which uses direct stacking.
        logger.warning("_combine_frames is likely deprecated. Stacking is done directly in _create_output_video.")
        if output_width is None: output_width = max(600, frame.shape[1])
        if output_height is None: output_height = frame.shape[0] + 400 # Assumes one plot
        plot_target_height = 400
        frame_target_height = output_height - plot_target_height

        if frame.shape[1] != output_width or frame.shape[0] != frame_target_height:
            frame_display = cv2.resize(frame, (output_width, frame_target_height))
        else: frame_display = frame.copy()

        if plot_img.shape[1] != output_width or plot_img.shape[0] != plot_target_height:
             plot_display = cv2.resize(plot_img, (output_width, plot_target_height))
        else: plot_display = plot_img

        return np.vstack((frame_display, plot_display))

    def _prepare_sequence_data(self, sequence: List[np.ndarray]) -> np.ndarray:
        """Validates and prepares the input sequence data."""
        # This function seems unused directly by the main flow now,
        # as standardization happens within _prepare_sequence.
        logger.warning("_prepare_sequence_data is likely unused.")
        if not isinstance(sequence, (list, np.ndarray)): raise TypeError(f"Unsupported sequence type: {type(sequence)}")
        cleaned_vectors = []
        if isinstance(sequence, list):
            for vec in sequence:
                if isinstance(vec, (list, tuple, np.ndarray)):
                    vec_np = np.array(vec, dtype=np.float32); vec_np = np.nan_to_num(vec_np, nan=0.0, posinf=0.0, neginf=0.0)
                    if vec_np.ndim == 1 and len(vec_np) == self.input_size: cleaned_vectors.append(vec_np)
                    else: logger.debug(f"Skipping invalid vector: shape {vec_np.shape}")
                else: logger.debug(f"Skipping invalid type: {type(vec)}")
            if not cleaned_vectors: raise ValueError("Sequence contains no valid vectors.")
            sequence_np = np.array(cleaned_vectors, dtype=np.float32)
        elif isinstance(sequence, np.ndarray): sequence_np = np.nan_to_num(sequence.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        else: raise TypeError(f"Unsupported sequence type: {type(sequence)}")
        if sequence_np.ndim != 2 or sequence_np.shape[1] != self.input_size: raise ValueError(f"Invalid sequence shape: {sequence_np.shape}. Expected (SeqLen, {self.input_size})")
        current_length = sequence_np.shape[0]; standardized_sequence = np.zeros((self.sequence_length, self.input_size), dtype=np.float32)
        if current_length == self.sequence_length: standardized_sequence = sequence_np
        elif current_length < self.sequence_length: standardized_sequence[:current_length, :] = sequence_np
        else: standardized_sequence = sequence_np[-self.sequence_length:, :]
        return standardized_sequence

    def _generate_direct_score_plot(self, video_path, analysis_data, job_type=None, plot_width=1200, output_dir=None, y_max=None):
        """
        Generates and saves the standalone direct movement score plot
        to the specified output directory.

        Args:
            video_path: Path to the video file (for naming)
            analysis_data: Dictionary containing movement data per person
            job_type: Optional job type for context
            plot_width: Width of the plot in pixels
            output_dir: Optional directory to save the plot (if None, uses default location)
            y_max: Optional explicit y-axis maximum for consistent scaling

        Returns:
            str: Path to the saved plot, or None if failed
        """
        try:
            plt.close('all')
            fig_width_inches = max(10, plot_width / 100)
            fig, ax = plt.subplots(figsize=(fig_width_inches, 6))
            has_plot_data = False
            max_score_value = 0
            min_score_value = float('inf')
            all_times = []
            all_scores = []

            for person_id, data in analysis_data.items():
                if 'times' in data and 'direct_scores' in data and len(data['times']) > 0:
                    times = np.array(data['times'])
                    scores = np.array(data['direct_scores'])
                    valid_mask = np.isfinite(scores) & (scores > 0)
                    if np.any(valid_mask):
                        valid_times = times[valid_mask]
                        valid_scores = scores[valid_mask]
                        all_times.extend(valid_times)
                        all_scores.extend(valid_scores)
                        if len(valid_scores) > 0:
                            max_score_value = max(max_score_value, np.max(valid_scores))
                            min_score_value = min(min_score_value, np.min(valid_scores))

            if not all_scores:
                logger.warning(f"No valid direct score data for plot {video_path.stem}.")
                plt.close(fig)
                return None

            has_plot_data = True

            # Determine y-axis max dynamically or use provided value
            if y_max is not None and np.isfinite(y_max) and y_max > 0:
                logger.debug(f"Using provided y_max={y_max:.2f} for direct score plot consistency")
            else:
                if all_scores:
                    score_max = np.percentile(all_scores, 99.5) if len(all_scores) > 10 else max_score_value
                    if self.direct_score_q3 is not None and np.isfinite(self.direct_score_q3):
                        y_max = max(score_max * 1.3, self.direct_score_q3 * 2.0)
                    else:
                        y_max = score_max * 1.5
                    y_max = max(y_max, 5.0); y_max = min(y_max, 200.0)
                    if max_score_value > 200.0: y_max = max_score_value * 1.2
                else:
                    y_max = 100.0
                logger.debug(f"Calculated direct score y_max={y_max:.2f}")

            color_palette = plt.cm.tab10.colors
            for i, (person_id, data) in enumerate(analysis_data.items()):
                if 'times' in data and 'direct_scores' in data and len(data['times']) > 0:
                    times = np.array(data['times'])
                    scores = np.array(data['direct_scores'])
                    valid_mask = np.isfinite(scores) & (scores > 0)
                    if np.any(valid_mask):
                        valid_times = times[valid_mask]
                        valid_scores = scores[valid_mask]
                        if len(valid_times) > 1:
                            person_color = color_palette[i % len(color_palette)]
                            # Plot continuous line without markers
                            ax.plot(valid_times, valid_scores, 
                                color=person_color,
                                linewidth=1.5, 
                                label=f'Person {person_id}', 
                                alpha=0.7,
                                marker=None)  # Explicitly set no markers

                            # REMOVED: Marker highlighting logic that added dots to the line

            # Add statistics from training data if available
            if self.direct_score_mean is not None and np.isfinite(self.direct_score_mean):
                ax.axhline(y=self.direct_score_mean, color='blue', linestyle='--',
                        alpha=0.7, label=f'Mean ({self.direct_score_mean:.2f})')
            if self.direct_score_q1 is not None and np.isfinite(self.direct_score_q1):
                ax.axhline(y=self.direct_score_q1, color='green', linestyle='--',
                        alpha=0.7, label=f'Q1 ({self.direct_score_q1:.2f})')
            if self.direct_score_q3 is not None and np.isfinite(self.direct_score_q3):
                ax.axhline(y=self.direct_score_q3, color='red', linestyle='--',
                        alpha=0.7, label=f'Q3 ({self.direct_score_q3:.2f})')

            if has_plot_data:
                summary_text = f"Direct Score Summary ({video_path.stem}):\nJob: {job_type if job_type else 'N/A'}, Persons: {len(analysis_data)}\nData points: {len(all_scores)}, Max Score: {max_score_value:.3f}"
                ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
                    va='top', ha='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

            ax.set_title(f'Direct Movement Score Analysis - {video_path.stem}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (seconds)', fontsize=12); ax.set_ylabel('Direct Movement Score', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--'); ax.set_ylim(bottom=0, top=y_max)

            if all_times:
                x_min = min(all_times); x_max = max(all_times)
                x_padding = (x_max - x_min) * 0.05
                ax.set_xlim(left=max(0, x_min - x_padding), right=x_max + x_padding)

            video_path = Path(video_path) # Ensure Path object

            # IMPORTANT FIX: Use the provided output directory if specified
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving direct score plot to specified directory: {output_dir}")
            else:
                # Fallback to default location (use video-specific directory)
                video_output_dir = self.output_dir / video_path.stem
                output_dir = video_output_dir / 'plots'
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving direct score plot to default directory: {output_dir}")

            # Use consistent naming with job type suffix
            job_suffix = f"_{job_type}" if job_type else ""
            output_path = output_dir / f"{video_path.stem}_direct_score{job_suffix}.png"

            # Handle legend
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                stat_indices = [i for i, label in enumerate(labels) if any(x in label for x in ['Mean', 'Q1', 'Q3'])]
                person_indices = [i for i, label in enumerate(labels) if 'Person' in label][:3]
                selected_indices = stat_indices + person_indices
                if selected_indices:
                    ax.legend([handles[i] for i in selected_indices],
                            [labels[i] for i in selected_indices],
                            loc='upper right', fontsize=10)

            plt.tight_layout()
            plt.savefig(output_path, dpi=120)
            plt.close(fig)
            logger.info(f"Direct score analysis plot saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error creating direct score plot: {e}", exc_info=True)
            plt.close('all')
            return None