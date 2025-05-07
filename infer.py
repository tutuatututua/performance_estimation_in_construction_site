import torch
import numpy as np
from utils.model_loader import load_movement_classifier  
from utils.pose_extraction import normalize_joints, calculate_displacement
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

class MovementClassifier:
    """Movement analysis class (anomaly detection via reconstruction error)."""

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = None
        self.model_info = None

        # --- Load Model and Model Info ---
        try:
            # This loader should return the AE model and its info
            self.model, self.model_info = load_movement_classifier(config.model_save_path, config)
            if self.model is None or self.model_info is None:
                raise RuntimeError("Failed to load movement model or model info.")
            logger.info(f"Movement analysis model (AE) loaded from {config.model_save_path}")
            self.model.eval() # Ensure model is in eval mode
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load movement analysis model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load movement analysis model: {e}") from e

        # --- Load model parameters ---
        try:
            model_cfg = self.model_info.get('model_config', {})
            self.input_size = model_cfg.get('input_size')
            # Prefer standardized length from info if available
            self.sequence_length = self.model_info.get('standardized_sequence_length', model_cfg.get('sequence_length'))
            self.embedding_dim = model_cfg.get('embedding_dim')
            self.conditional = model_cfg.get('conditional')
            self.num_job_categories = model_cfg.get('num_job_categories')

            if any(param is None for param in [self.input_size, self.sequence_length, self.embedding_dim]):
                # Log details if parameters are missing
                missing = [k for k in ['input_size', 'sequence_length', 'embedding_dim'] if model_cfg.get(k) is None]
                logger.error(f"Incomplete model configuration for Autoencoder. Missing: {missing}")
                raise ValueError("Incomplete model configuration for Autoencoder.")

            logger.info(f"Model parameters loaded: input_size={self.input_size}, sequence_length={self.sequence_length}, embedding_dim={self.embedding_dim}")

        except (TypeError, ValueError) as e:
            logger.error(f"Error processing model parameters: {e}", exc_info=True)
            raise ValueError("Invalid model parameters.") from e

    def predict_sequence(self, sequence, job_type=None):
        """
        Predict (reconstruct) a sequence and return the reconstruction error.
        For backward compatibility, also returns a class prediction and probabilities.
        Returns (reconstruction_error, class_prediction, probabilities)
        """
        if self.model is None:
            logger.error("Model not loaded, cannot predict.")
            return float('inf'), "UNKNOWN", {"LOW": 0.0, "NORMAL": 0.0, "HIGH": 0.0}
        self.model.eval()
        try:
            with torch.no_grad():
                # --- Input Validation and Preparation ---
                if not isinstance(sequence, np.ndarray):
                    try:
                        sequence = np.array(sequence, dtype=np.float32)
                    except Exception as conv_err:
                        raise ValueError(f"Input sequence must be convertible to a NumPy array. Error: {conv_err}")

                # Handle potential empty sequence after conversion
                if sequence.ndim == 0 or sequence.size == 0:
                    logger.warning("predict_sequence received an empty sequence.")
                    return float('inf'), "UNKNOWN", {"LOW": 0.0, "NORMAL": 0.0, "HIGH": 0.0}

                if sequence.ndim != 2:
                    # Attempt reshape if 1D (SeqLen*Features,)
                    if sequence.ndim == 1 and self.input_size is not None and sequence.size % self.input_size == 0:
                        seq_len_inferred = sequence.size // self.input_size
                        logger.warning(f"Input sequence is 1D, attempting reshape to ({seq_len_inferred}, {self.input_size})")
                        sequence = sequence.reshape(seq_len_inferred, self.input_size)
                    else:
                        err_msg = f"predict_sequence expects a 2D numpy array (SeqLen, Features={self.input_size}), got ndim={sequence.ndim}, shape={sequence.shape}"
                        logger.error(err_msg)
                        return float('inf'), "UNKNOWN", {"LOW": 0.0, "NORMAL": 0.0, "HIGH": 0.0}

                # Check sequence feature count against expected input size
                if sequence.shape[1] != self.input_size:
                    logger.error(f"Sequence has {sequence.shape[1]} features, expected {self.input_size}")
                    return float('inf'), "UNKNOWN", {"LOW": 0.0, "NORMAL": 0.0, "HIGH": 0.0}

                # Pad or truncate the sequence to the standardized length
                current_length = sequence.shape[0]
                standardized_sequence = np.zeros((self.sequence_length, self.input_size), dtype=np.float32)
                if current_length == self.sequence_length:
                    standardized_sequence = sequence
                elif current_length < self.sequence_length:
                    # Pad at the end instead of beginning to be consistent with training
                    standardized_sequence = np.zeros((self.sequence_length, self.input_size), dtype=np.float32)
                    standardized_sequence[:current_length, :] = sequence  # Pad end
                else: # current_length > self.sequence_length
                    standardized_sequence = sequence[-self.sequence_length:, :]

                # Ensure standardized_sequence is finite
                standardized_sequence = np.nan_to_num(standardized_sequence, nan=0.0, posinf=0.0, neginf=0.0)

                x = torch.from_numpy(standardized_sequence.astype(np.float32)).unsqueeze(0).to(self.device)

                # --- Conditional Input (if applicable) ---
                job_ids = None
                if self.conditional and job_type is not None and hasattr(self.config, 'job_categories'):
                    if job_type in self.config.job_categories:
                        try:
                            job_idx = self.config.job_categories.index(job_type)
                            job_ids = torch.tensor([job_idx], device=self.device, dtype=torch.long)
                        except (ValueError, IndexError):
                            logger.warning(f"Job type '{job_type}' index error.") # Should not happen if check passes
                    else:
                        logger.warning(f"Job type '{job_type}' not found in config.job_categories. Proceeding without condition.")

                # --- Model Prediction (Reconstruction) ---
                reconstructed_sequence = self.model(x, job_ids) # Pass condition if available

                # --- Calculate Reconstruction Error ---
                reconstruction_error = torch.mean((reconstructed_sequence - x) ** 2).item()

                # Ensure error is a finite float
                if not np.isfinite(reconstruction_error):
                    logger.warning(f"Non-finite reconstruction error calculated: {reconstruction_error}. Returning inf.")
                    return float('inf'), "UNKNOWN", {"LOW": 0.0, "NORMAL": 0.0, "HIGH": 0.0}

                # --- For backward compatibility: Determine class and probabilities based on error ---
                # These thresholds should be set in the model_info or config
                low_threshold = getattr(self, 'low_threshold', 30.0)  # Default value
                high_threshold = getattr(self, 'high_threshold', 70.0)  # Default value
                
                # Get thresholds from model_info if available
                if hasattr(self, 'model_info') and 'anomaly_thresholds' in self.model_info:
                    thresholds = self.model_info['anomaly_thresholds']
                    low_threshold = thresholds.get('low_threshold', low_threshold)
                    high_threshold = thresholds.get('high_threshold', high_threshold)
                
                # Determine class based on thresholds
                if reconstruction_error < low_threshold:
                    class_prediction = "LOW"
                    probs = {"LOW": 0.8, "NORMAL": 0.2, "HIGH": 0.0}
                elif reconstruction_error > high_threshold:
                    class_prediction = "HIGH"
                    probs = {"LOW": 0.0, "NORMAL": 0.2, "HIGH": 0.8}
                else:
                    class_prediction = "NORMAL"
                    probs = {"LOW": 0.1, "NORMAL": 0.8, "HIGH": 0.1}
                
                # Return consistent with both old and new interfaces
                return float(reconstruction_error), class_prediction, probs

        except Exception as e:
            logger.error(f"Error predicting sequence: {str(e)}", exc_info=True)
            return float('inf'), "UNKNOWN", {"LOW": 0.0, "NORMAL": 0.0, "HIGH": 0.0} 
    
    def analyze_movement(self, current_pose, previous_pose, job_type=None):
        """
        Analyzes movement between two consecutive poses using direct calculation.
        This function is kept for potential use in the video analysis, but its
        relevance is reduced in the autoencoder context.
        """
        normalized_current = normalize_joints(current_pose)
        normalized_previous = normalize_joints(previous_pose)
        if normalized_current is None or normalized_previous is None:
            return self._create_default_result()
        displacement = calculate_displacement(normalized_current, normalized_previous)
        if displacement is None:
            return self._create_default_result()

        movement_score = 0.0
        try:
            disp_array = np.array(displacement, dtype=np.float32)
            disp_array = np.nan_to_num(disp_array)
            joint_abs_disp = np.abs(disp_array)
            max_abs_disp = np.max(joint_abs_disp) if len(joint_abs_disp) > 0 else 0.0
            movement_score = max(0.0, max_abs_disp * 5.0)  # Example scaling factor
            movement_score = min(100.0, movement_score)
            movement_score = float(movement_score)
        except Exception as e:
            logger.error(f"Error calculating movement score from displacement: {e}", exc_info=True)
            return self._create_default_result()

        return {
            'movement_score': movement_score,
            'raw_displacement': displacement
        }

    def _create_default_result(self):
        """Return a default result when movement analysis fails."""
        return {
            'movement_score': 0.0,
            'raw_displacement': None
        }

    def analyze_sequence(self, sequence, job_type=None, actual_movements=None, video_name=None):
        """
        Analyze a sequence of displacement vectors using the LSTM Autoencoder.
        Calculates and returns the reconstruction error for the sequence.
        """
        # This method now primarily calls predict_sequence
        if self.model is None:
            return {'error': 'Movement analysis model not loaded.'}

        try:
            # predict_sequence handles validation, standardization, and prediction
            reconstruction_error = self.predict_sequence(sequence, job_type)

            if not np.isfinite(reconstruction_error):
                 return {'error': f'Prediction resulted in non-finite error: {reconstruction_error}'}

            # --- Prepare Result ---
            result = {
                'reconstruction_error': reconstruction_error,
                # Visualization generation might be removed or adapted separately
                'visualization_path': None
            }

            return result

        except ValueError as ve: # Catch specific validation errors from predict_sequence
             logger.error(f"Data error analyzing sequence for video '{video_name}': {ve}")
             return {'error': str(ve)}
        except Exception as e:
            logger.error(f"Unexpected error in analyze_sequence for video '{video_name}': {e}", exc_info=True)
            return {'error': f'Unexpected error: {e}'}

    def _create_comparison_visualization(self, predictions, actual_movements, job_type=None, video_name=None):
        """
        Create visualization comparing reconstruction error with (optionally)
        actual movement measures.  Adapt this as needed for your analysis.
        """
        # (Adapt this function to visualize reconstruction error)
        # Placeholder - needs implementation
        logger.warning("_create_comparison_visualization is a placeholder and not implemented for autoencoders.")
        return None