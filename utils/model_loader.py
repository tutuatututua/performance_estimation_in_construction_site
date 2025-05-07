from ultralytics import YOLO
import torch
import json
from pathlib import Path
import logging
import yaml
import traceback
# Ensure necessary imports are present
from models.lstm_movement import LSTMAutoencoder # Make sure this import works

# Configure logging
# logging.basicConfig(level=logging.INFO) # Avoid reconfiguring if already done elsewhere
logger = logging.getLogger(__name__)

# --- load_pose_model function (keep as is) ---
def load_pose_model(model_path, config=None):
    """Load YOLO pose detection model with tracking."""
    try:
        model_path = Path(model_path)
        logger.info(f"Loading YOLO model with tracking: {model_path}")

        # Initialize model variable to None to ensure it's defined
        model = None

        # First check if the model exists
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}, downloading from Ultralytics...")
            try:
                # Try to download the model using a more reliable method
                model_name = getattr(config, 'model_name', 'yolov8m-pose.pt') # Using medium as default now
                model = YOLO(model_name)

                # Ensure the directory exists
                model_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the downloaded model
                model.save(str(model_path))
                logger.info(f"Downloaded model and saved to {model_path}")
            except Exception as download_error:
                logger.error(f"Error downloading model: {str(download_error)}")
                logger.info("Falling back to default pose model...")
                model = YOLO('yolov8n-pose.pt')  # Use smaller model as fallback
        else:
            # Try to load existing model
            try:
                model = YOLO(str(model_path))
            except Exception as load_error:
                logger.error(f"Error loading from {model_path}: {str(load_error)}")
                logger.info("Falling back to default pose model...")
                model = YOLO('yolov8n-pose.pt')

        # Verify model was loaded successfully
        if model is None:
            logger.error("Failed to load model, using fallback")
            model = YOLO('yolov8n-pose.pt')

        # --- MODIFIED: Check if tracker config from Config exists before creating default ---
        tracker_config_path = Path(getattr(config, 'tracker_config', 'bytetrack.yaml'))
        if not tracker_config_path.exists():
            logger.info(f"Tracker config '{tracker_config_path}' not found. Creating default bytetrack.yaml")
            # Define default config content
            bytetrack_config = {
                'tracker_type': 'bytetrack',
                'track_high_thresh': 0.6,
                'track_low_thresh': 0.1,
                'new_track_thresh': 0.7,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'motion_scale': 0.1,      # Added default based on common examples
                'appearance_scale': 0.9,  # Added default based on common examples
                'fuse_score': True        # Added default based on common examples
            }
            try:
                # Create the tracker config file (ensure parent dir exists)
                tracker_config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tracker_config_path, 'w') as f:
                    yaml.dump(bytetrack_config, f, default_flow_style=False)
                logger.info(f"Created default ByteTrack config at {tracker_config_path}")
            except Exception as e:
                logger.error(f"Failed to create default tracker config at {tracker_config_path}: {e}")
        # --- END MODIFICATION ---

        # Tracking configuration
        if config: # Ensure config is available
            tracking_args = {
                'conf': getattr(config, 'yolo_conf_threshold', 0.25),
                'iou': getattr(config, 'yolo_iou_threshold', 0.45),
                'imgsz': 640, # Keep default or get from config if available
                'verbose': False
            }

            # Update model configuration - only if model is valid and overrides exist
            if hasattr(model, 'overrides'):
                model.overrides.update(tracking_args)
            else:
                logger.warning("Model object does not have 'overrides'. Tracking args not set.")
        else:
            logger.warning("Config object not provided to load_pose_model. Using default tracking args.")


        # Verify model is loaded correctly
        if not hasattr(model, 'names') or not model.names:
            raise ValueError("Invalid YOLO model loaded")

        logger.info(f"YOLO model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}")
        # Create a last-resort fallback
        try:
            logger.info("Attempting to load basic YOLO model as fallback")
            model = YOLO('yolov8n-pose.pt')
            return model
        except:
            logger.critical("Critical error: Could not load any YOLO model")
            raise

# --- FIXED load_movement_classifier function ---
def load_movement_classifier(model_path, config):
    """Load trained LSTM Autoencoder model and its associated info."""
    model = None
    model_info = None
    model_cfg = None # Initialize model_cfg

    try:
        device = config.device
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")

        logger.info(f"Loading model checkpoint from {model_path}")
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

        # --- Determine Model Config ---
        # Priority 1: From checkpoint
        if 'model_config' in checkpoint:
            model_cfg = checkpoint['model_config']
            logger.info("Using model_config from checkpoint.")
        # Priority 2: From model_info.json (will be loaded next)
        # Priority 3: From Config object (used if JSON fails or lacks config)

        # --- Load model_info.json (ALWAYS attempt this to get thresholds) ---
        info_path = model_path.parent / 'model_info.json'
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                logger.info(f"Successfully loaded model info from: {info_path}")
                # If model_cfg wasn't in checkpoint, try getting it from JSON
                if model_cfg is None and 'model_config' in model_info:
                    model_cfg = model_info['model_config']
                    logger.info("Using model_config from model_info.json.")
                # Check if thresholds are present
                if 'anomaly_thresholds' not in model_info:
                    logger.warning(f"'anomaly_thresholds' key NOT FOUND in loaded JSON: {info_path}")
            except Exception as e:
                logger.warning(f"Error loading model info from {info_path}: {e}. Proceeding without it.")
                model_info = None # Set to None if loading fails
        else:
             logger.warning(f"model_info.json NOT found at {info_path}")
             model_info = None

        # --- Final Fallback for model_cfg ---
        if model_cfg is None:
             logger.warning("Model config not found in checkpoint or model_info.json. Using config object's get_model_config.")
             model_cfg = config.get_model_config()

        # --- Validate the determined model_cfg ---
        if not model_cfg or 'input_size' not in model_cfg or 'sequence_length' not in model_cfg:
             logger.error("Critical: Valid 'model_config' could not be determined. Cannot initialize model.")
             raise ValueError("Valid 'model_config' is missing.")

        # --- Initialize Model using the determined model_config ---
        try:
            model = LSTMAutoencoder(**model_cfg).to(device)
            logger.info(f"LSTM Autoencoder model initialized with config: {model_cfg}")
        except Exception as init_err:
             logger.error(f"Failed to initialize LSTMAutoencoder with determined config {model_cfg}: {init_err}")
             raise

        # --- Load Weights into the initialized model ---
        try:
            if 'model_state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['model_state_dict'])
                 logger.info("Model weights loaded successfully from checkpoint")
            else:
                 logger.error("Checkpoint doesn't contain 'model_state_dict' key. Cannot load weights.")
                 raise ValueError("Invalid checkpoint format: missing model_state_dict")
        except Exception as state_dict_err:
            logger.error(f"Error loading model state dict: {state_dict_err}")
            logger.debug(f"Model keys: {list(model.state_dict().keys())}")
            logger.debug(f"Checkpoint keys: {list(checkpoint['model_state_dict'].keys())}")
            raise # Re-raise the exception

        model.eval()

        # --- Create or finalize model_info for return ---
        if model_info is None: # If JSON loading failed or file didn't exist
             logger.info("Creating basic model_info as JSON was not loaded.")
             model_info = {
                 'model_config': model_cfg,
                 'job_categories': config.job_categories,
                 'standardized_sequence_length': model_cfg.get('sequence_length')
             }
             # Indicate missing thresholds explicitly if JSON wasn't loaded
             model_info['anomaly_thresholds'] = {'source': 'missing (JSON load failed or file absent)'}
             logger.warning("Anomaly thresholds could not be loaded from model_info.json.")
        elif 'anomaly_thresholds' not in model_info:
             # If JSON loaded but lacked the key
             model_info['anomaly_thresholds'] = {'source': 'missing (key not in JSON)'}
             logger.warning("Anomaly thresholds key missing from loaded model_info.json.")
        else:
             logger.info("Anomaly thresholds successfully loaded from model_info.json.")

        # Ensure model_config in the final model_info matches the one used
        model_info['model_config'] = model_cfg

        return model, model_info # Return the initialized and weighted model and the info

    except FileNotFoundError as fnf_error:
         logger.error(f"Model file not found: {fnf_error}")
         return None, None # Indicate failure
    except ValueError as val_error:
         logger.error(f"Model loading failed due to value error: {val_error}")
         return None, None # Indicate failure
    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None # Indicate failure

