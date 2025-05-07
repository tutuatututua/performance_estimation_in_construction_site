from dataclasses import dataclass, field
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import os
import psutil
import logging
import sys
import yaml

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for movement analysis system (now anomaly detection)."""

    # System Configuration
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    num_workers: int = field(
        default_factory=lambda: min(os.cpu_count() or 1, 4)
    )

    # Processing Optimization
    enable_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    max_cache_size_gb: float = 50.0
    frame_resize: tuple = (640, 480)
    batch_size_frames: int = 32
    max_memory_frames: int = 100  # Frames to process before potential memory clear during data prep

    # Data Loading Optimization
    dataloader_workers: int = 3
    prefetch_factor: int = 2
    pin_memory: bool = field(default_factory=lambda: torch.cuda.is_available())
    persistent_workers: bool = True
    # Removed displacement_scale_factor and iqr_multiplier

    # --- Model Architecture (Autoencoder) ---
    hidden_size: int = 128  # LSTM hidden size
    num_layers: int = 1     # Number of LSTM layers
    num_joints: int = 6     # Number of joints/features (Used to set AE input_size)
    embedding_dim: int = 64  # Size of the compressed representation (AE embedding)
    dropout: float = 0.2
    bidirectional_encoder: bool = False  # Option for encoder
    conditional_autoencoder: bool = False # Flag for Conditional AE
    num_job_categories: int = 3 # Only if conditional_autoencoder is True

    # Training Parameters (Autoencoder)
    batch_size: int = 32
    learning_rate: float = 0.001 # Adjust as needed for AE
    num_epochs: int = 100 # Increase epochs for AE training?
    early_stopping_patience: int = 20 # Adjust patience for AE
    l2_regularization: float = 1e-4

    # Data Processing
    video_dir: Path = field(default_factory=lambda: Path("data/videos"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    sample_rate: int = 8  # Target analysis Hz
    min_sequence_length: int = 80  # Standard sequence length for LSTM
    standardized_sequence_length: Optional[int] = None # Set dynamically during processing

    # Model Paths
    model_save_path: Path = field(default_factory=lambda: Path("models/saved/lstm_autoencoder.pth"))
    yolo_model_path: Path = field(default_factory=lambda: Path("models/yolo11m-pose.pt"))
    model_name: str = "yolo11m-pose.pt"

    # Tracking Configuration
    use_tracking: bool = True
    tracker_config: str = "bytetrack.yaml"

    # YOLO Detection
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    yolo_min_keypoint_conf: float = 0.25

    # Job Categories
    job_categories: List[str] = field(default_factory=lambda: ["masonry", "painting", "plastering"])

    # Internal flag
    single_video_mode: bool = False
    # Input path argument (used in analyze.py)
    input_path_arg: Optional[Path] = None

    # --- Methods ---
    def __post_init__(self):
        """Initialize and validate configuration."""
        self.video_dir = Path(self.video_dir)
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        self.model_save_path = Path(self.model_save_path)
        self.yolo_model_path = Path(self.yolo_model_path)
        self._validate_parameters()
        self._setup_directories()
        self._check_system_resources()

    def _validate_parameters(self):
        """Validate configuration parameters."""
        if not 0 <= self.yolo_conf_threshold <= 1: raise ValueError("YOLO confidence threshold must be between 0 and 1")
        if not 0 <= self.yolo_iou_threshold <= 1: raise ValueError("YOLO IOU threshold must be between 0 and 1")
        if not 0 <= self.yolo_min_keypoint_conf <= 1: raise ValueError("YOLO keypoint confidence must be between 0 and 1")
        if self.batch_size < 1: raise ValueError("Batch size must be positive")
        if self.batch_size_frames < 1: raise ValueError("Frame batch size must be positive")
        if self.learning_rate <= 0: raise ValueError("Learning rate must be positive")
        if len(self.frame_resize) != 2 or self.frame_resize[0] <= 0 or self.frame_resize[1] <= 0: raise ValueError("Frame resize must be a tuple of two positive integers (width, height)")
        if self.use_tracking and not Path(self.tracker_config).exists(): self._create_default_tracker_config()
        if not self.job_categories: raise ValueError("job_categories cannot be empty.")
        if self.min_sequence_length <= 0: raise ValueError("min_sequence_length must be positive.")

    def _create_default_tracker_config(self):
        """Creates a default tracker configuration file if it doesn't exist."""
        tracker_config_path = Path(self.tracker_config)
        if not tracker_config_path.exists():
            logger.info(f"Tracker config '{tracker_config_path}' not found. Creating default bytetrack.yaml")
            bytetrack_config = { 'tracker_type': 'bytetrack', 'track_high_thresh': 0.6, 'track_low_thresh': 0.1, 'new_track_thresh': 0.7, 'track_buffer': 30, 'match_thresh': 0.8, 'motion_scale': 0.1, 'appearance_scale': 0.9, 'fuse_score': True }
            try:
                tracker_config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tracker_config_path, 'w') as f: yaml.dump(bytetrack_config, f, default_flow_style=False)
                logger.info(f"Created default ByteTrack config at {tracker_config_path}")
            except Exception as e: logger.error(f"Failed to create default tracker config at {tracker_config_path}: {e}")

    def _setup_directories(self):
        """Create necessary directories."""
        if not hasattr(self, '_dirs_setup'):
            logger.info("Setting up output directories...")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
            if self.enable_cache: self.cache_dir.mkdir(parents=True, exist_ok=True); self._cleanup_cache()
            self._dirs_setup = True

    def _cleanup_cache(self):
        """Clean up cache if it exceeds size limit."""
        # (Implementation unchanged)
        if not self.enable_cache or not self.cache_dir.exists(): return
        try:
            cache_files = list(self.cache_dir.glob('*.npz'))
            cache_size = sum(f.stat().st_size for f in cache_files if f.is_file())
            cache_size_gb = cache_size / (1024**3)
            if cache_size_gb > self.max_cache_size_gb:
                logger.warning(f"Cache size ({cache_size_gb:.2f}GB) exceeds limit ({self.max_cache_size_gb}GB). Cleaning oldest files...")
                cache_files.sort(key=lambda f: f.stat().st_mtime)
                space_to_free = cache_size - (self.max_cache_size_gb * (1024**3)); freed_space = 0; removed_count = 0
                for f in cache_files:
                    if freed_space >= space_to_free: break
                    try: size = f.stat().st_size; f.unlink(); freed_space += size; removed_count += 1
                    except OSError as e: logger.error(f"Error removing cache file {f}: {e}")
                logger.info(f"Cache cleanup removed {removed_count} oldest files, freeing {(freed_space / (1024**3)):.2f}GB.")
        except Exception as e: logger.error(f"Error during cache cleanup: {e}")


    def _check_system_resources(self):
        """Check system resources and potentially adjust batch size."""
        # (Implementation unchanged)
        if hasattr(self, '_resources_checked'): return
        logger.info("System Resources Check:")
        try:
            cpu_count = os.cpu_count(); logger.info(f"  CPU Count: {cpu_count}")
            mem = psutil.virtual_memory(); total_mem_gb = mem.total / (1024**3); available_mem_gb = mem.available / (1024**3)
            logger.info(f"  Total RAM: {total_mem_gb:.2f} GB"); logger.info(f"  Available RAM: {available_mem_gb:.2f} GB")
            if self.device.type == 'cuda':
                props = torch.cuda.get_device_properties(self.device); gpu_mem_gb = props.total_memory / (1024**3)
                logger.info(f"  GPU: {props.name} ({gpu_mem_gb:.2f} GB)")
                if gpu_mem_gb < 6 and self.batch_size > 16: logger.warning(f"  Low GPU memory detected ({gpu_mem_gb:.2f}GB). Consider reducing --batch_size (current: {self.batch_size}) if OOM errors occur.")
            max_workers = cpu_count or 1
            if self.dataloader_workers > max_workers: logger.warning(f"  Requested dataloader workers ({self.dataloader_workers}) exceeds CPU count ({max_workers}). Setting to {max_workers}."); self.dataloader_workers = max_workers
            if self.persistent_workers and self.dataloader_workers == 0: logger.info("  Setting persistent_workers=False because dataloader_workers=0."); self.persistent_workers = False
            elif not self.persistent_workers and self.dataloader_workers > 0 and self.num_epochs > 1: logger.info("  Consider setting persistent_workers=True for potential speedup with multiple epochs.")
        except Exception as e: logger.warning(f"  Could not perform full system resource check: {e}")
        self._resources_checked = True

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration (for autoencoder)."""
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "input_size": self.num_joints, # Correctly maps num_joints config to AE's input_size
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "conditional": self.conditional_autoencoder,
            "num_job_categories": self.num_job_categories,
            "sequence_length": self.min_sequence_length,
            "bidirectional_encoder": self.bidirectional_encoder,
            # REMOVED the redundant "num_joints" key here
        }

    def set_standardized_length(self, length: int):
        """Stores the sequence length used after standardization."""
        logger.info(f"Setting standardized sequence length used for model: {length}")
        self.standardized_sequence_length = length