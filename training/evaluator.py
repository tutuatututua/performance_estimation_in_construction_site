# In training/evaluator.py

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score # Keep if used elsewhere
import matplotlib
matplotlib.use('Agg') # Use Agg backend before importing pyplot
import matplotlib.pyplot as plt
import logging
import pandas as pd
import seaborn as sns
from pathlib import Path
import traceback
from tqdm import tqdm
from collections import defaultdict
import json # Added for saving thresholds
from utils.visualization import visualize_original_vs_reconstructed, create_reconstruction_error_visualization # Ensure imports

logger = logging.getLogger(__name__)

# Define JOINT_NAMES if needed for other functions in this file
JOINT_NAMES = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']

class AnomalyEvaluator:
    """Evaluator for movement anomaly detection using LSTM Autoencoder."""

    def __init__(self, model, config, model_info):
        """
        Initialize the evaluator.

        Args:
            model: The trained autoencoder model instance
            config: The configuration object
            model_info: Information loaded alongside the model
        """
        self.model = model
        self.config = config
        self.device = config.device
        # Store a copy of model_info to update later if needed
        self.model_info = model_info.copy() if model_info else {}
        self.model.eval()  # Set model to evaluation mode

        # Define error percentiles for thresholds
        self.low_percentile = 15
        self.high_percentile = 95

        # <<< NEW: Define percentiles for direct movement score magnitude >>>
        self.direct_score_q1_percentile = 15
        self.direct_score_q3_percentile = 85 # Captures central 70%

        # Initialize thresholds - they will be calculated during evaluation
        self.low_threshold = None
        self.high_threshold = None
        # <<< NEW: Initialize direct score stats >>>
        self.direct_score_mean = None
        self.direct_score_q1 = None
        self.direct_score_q3 = None


        # Get model configuration details
        self.model_config = self.model_info.get('model_config', {})

        # Verify model configuration has necessary parameters
        required_params = ['input_size', 'sequence_length', 'embedding_dim']
        if any(self.model_config.get(param) is None for param in required_params):
            logger.error("Incomplete model configuration found in model_info.")

        logger.info(f"Anomaly Evaluator initialized.")

    def evaluate_model(self, val_loader):
        """
        Evaluate autoencoder model, calculate reconstruction error thresholds,
        calculate direct movement score statistics, and save them to model_info.json.

        Args:
            val_loader: DataLoader containing validation data

        Returns:
            dict: Evaluation results with error statistics and visualizations, or None if failed.
        """
        if val_loader is None or not hasattr(val_loader, 'dataset') or len(val_loader.dataset) == 0:
            logger.error("Validation loader is required and cannot be empty for evaluation.")
            return None

        logger.info(f"Evaluating autoencoder on validation data (Size: {len(val_loader.dataset)} samples)")
        self.model.eval()

        # Store reconstruction errors, sequences, job types, etc.
        all_errors = []
        all_job_ids = []
        all_sequences = []
        all_reconstructions = []
        # <<< NEW: Store direct scores (labels) >>>
        all_direct_scores = []

        # --- Collect Errors and Data ---
        try:
            with torch.no_grad():
                pbar_eval = tqdm(val_loader, desc="Evaluating", leave=False, total=len(val_loader))
                for batch_data in pbar_eval:
                    if len(batch_data) != 3:
                        logger.warning(f"Skipping batch: Expected 3 items, got {len(batch_data)}")
                        continue

                    seq, label, job_id = batch_data # Unpack data (sequence, label, job_id)
                    seq = seq.to(self.device)
                    job_id = job_id.to(self.device)
                    # <<< NEW: Move labels (direct scores) to CPU/numpy >>>
                    direct_scores_batch = label.cpu().numpy()

                    job_condition = job_id if self.model_config.get('conditional', False) else None
                    reconstructed_seq = self.model(seq, job_condition)
                    errors = torch.mean((reconstructed_seq - seq) ** 2, dim=(1, 2)).cpu().numpy()

                    # Store data for analysis
                    all_errors.extend(errors)
                    all_job_ids.extend(job_id.cpu().numpy())
                    all_sequences.extend(seq.cpu().numpy())
                    all_reconstructions.extend(reconstructed_seq.cpu().numpy())
                    # <<< NEW: Store direct scores >>>
                    all_direct_scores.extend(direct_scores_batch)

        except Exception as e:
            logger.error(f"Error during evaluation loop: {e}", exc_info=True)
            return None

        # --- Convert collected lists to numpy arrays ---
        all_errors = np.array(all_errors)
        all_direct_scores = np.array(all_direct_scores) # New

        if len(all_errors) == 0:
             logger.error("No errors collected during evaluation. Cannot proceed.")
             return None
        # <<< NEW: Check if direct scores were collected >>>
        if len(all_direct_scores) == 0 or len(all_direct_scores) != len(all_errors):
            logger.error("Failed to collect direct scores or length mismatch. Cannot calculate direct score stats.")
            # Decide how to proceed: either return None or continue without direct score stats
            # For now, let's warn and continue, setting stats to None
            can_calculate_direct_stats = False
        else:
            can_calculate_direct_stats = True


        # --- Calculate and Save Reconstruction Error Thresholds ---
        try:
            self.low_threshold = float(np.percentile(all_errors, self.low_percentile))
            self.high_threshold = float(np.percentile(all_errors, self.high_percentile))
            logger.info(f"Calculated RECONSTRUCTION ERROR thresholds: "
                        f"Low ({self.low_percentile}%)={self.low_threshold:.4f}, "
                        f"High ({self.high_percentile}%)={self.high_threshold:.4f}")
        except Exception as calc_err:
             logger.error(f"Error calculating reconstruction error thresholds: {calc_err}", exc_info=True)
             self.low_threshold = None
             self.high_threshold = None
             logger.warning("Reconstruction error thresholds calculation failed.")


        # <<< NEW: Calculate Direct Movement Score Statistics >>>
        if can_calculate_direct_stats:
            try:
                # Filter out non-finite values before calculating stats
                finite_direct_scores = all_direct_scores[np.isfinite(all_direct_scores)]
                if len(finite_direct_scores) > 0:
                    self.direct_score_mean = float(np.mean(finite_direct_scores))
                    self.direct_score_q1 = float(np.percentile(finite_direct_scores, self.direct_score_q1_percentile))
                    self.direct_score_q3 = float(np.percentile(finite_direct_scores, self.direct_score_q3_percentile))
                    logger.info(f"Calculated DIRECT SCORE stats: "
                                f"Mean={self.direct_score_mean:.2f}, "
                                f"Q1 ({self.direct_score_q1_percentile}%)={self.direct_score_q1:.2f}, "
                                f"Q3 ({self.direct_score_q3_percentile}%)={self.direct_score_q3:.2f}")
                else:
                    logger.warning("No finite direct scores found in validation data. Cannot calculate direct score stats.")
                    can_calculate_direct_stats = False # Mark as failed
                    self.direct_score_mean, self.direct_score_q1, self.direct_score_q3 = None, None, None
            except Exception as calc_direct_err:
                logger.error(f"Error calculating direct score statistics: {calc_direct_err}", exc_info=True)
                can_calculate_direct_stats = False # Mark as failed
                self.direct_score_mean, self.direct_score_q1, self.direct_score_q3 = None, None, None
        else:
            logger.warning("Skipping direct score statistics calculation due to earlier data collection issues.")
            self.direct_score_mean, self.direct_score_q1, self.direct_score_q3 = None, None, None


        # --- Save Thresholds and Stats to model_info.json ---
        info_path = self.config.model_save_path.parent / 'model_info.json'
        try:
            if info_path.exists():
                with open(info_path, 'r') as f: existing_info = json.load(f)
            else:
                logger.warning(f"model_info.json not found at {info_path}. Creating a new one.")
                existing_info = {
                    'model_config': self.model_config,
                    'job_categories': self.config.job_categories,
                    'standardized_sequence_length': getattr(self.config, 'standardized_sequence_length', self.model_info.get('standardized_sequence_length', self.model_config.get('sequence_length')))
                }

            # Add/update reconstruction error threshold info
            if self.low_threshold is not None and self.high_threshold is not None:
                existing_info['anomaly_thresholds'] = {
                    'low_threshold': self.low_threshold,
                    'high_threshold': self.high_threshold,
                    'source': f'percentile_{self.low_percentile}_{self.high_percentile}_validation',
                    'validation_sample_count': len(all_errors)
                }
            else:
                 existing_info['anomaly_thresholds'] = {'source': 'calculation_failed'}

            # <<< NEW: Add/update direct score stats info >>>
            if can_calculate_direct_stats and self.direct_score_mean is not None:
                existing_info['direct_score_stats'] = {
                    'calculation_method': 'mean_rms',
                    'mean': self.direct_score_mean,
                    'q1': self.direct_score_q1,
                    'q3': self.direct_score_q3,
                    'q1_percentile': self.direct_score_q1_percentile,
                    'q3_percentile': self.direct_score_q3_percentile,
                    'source': 'validation_data_stats',
                    'validation_sample_count': len(finite_direct_scores) # Use count of finite scores
                }
            else:
                 existing_info['direct_score_stats'] = {'source': 'calculation_failed_or_no_data'}


            # Save updated info
            with open(info_path, 'w') as f: json.dump(existing_info, f, indent=4)
            logger.info(f"Updated model info with thresholds and direct score stats saved to: {info_path}")

            # Update the evaluator's own model_info attribute
            self.model_info['anomaly_thresholds'] = existing_info['anomaly_thresholds']
            # <<< NEW: Update evaluator's info >>>
            self.model_info['direct_score_stats'] = existing_info['direct_score_stats']

        except Exception as save_err:
            logger.error(f"Failed to save updated model_info.json: {save_err}", exc_info=True)


        # --- Calculate Overall Statistics (Reconstruction Error) ---
        stats = {
            'mean_error': float(np.mean(all_errors)),
            'median_error': float(np.median(all_errors)),
            'min_error': float(np.min(all_errors)),
            'max_error': float(np.max(all_errors)),
            'std_error': float(np.std(all_errors)),
            'low_threshold': self.low_threshold,
            'high_threshold': self.high_threshold,
            'anomaly_count': int(np.sum(all_errors > self.high_threshold)) if self.high_threshold is not None else np.nan,
            'anomaly_percentage': float(100 * np.sum(all_errors > self.high_threshold) / len(all_errors)) if self.high_threshold is not None else np.nan
        }
        # ...(rest of the statistics logging remains the same)...

        # --- Process Results by Job Type ---
        # ...(This part remains largely the same, focused on reconstruction error)...
        # ...(It could be enhanced later to include direct score stats per job)...
        job_results = {}
        all_job_ids = np.array(all_job_ids)
        job_categories = self.config.job_categories

        for job_idx, job_type in enumerate(job_categories):
            job_indices = np.where(all_job_ids == job_idx)[0]
            if len(job_indices) == 0:
                logger.info(f"  {job_type.capitalize()}: No validation samples found.")
                continue

            job_errors = all_errors[job_indices]
            job_seqs = [all_sequences[i] for i in job_indices]
            job_recons = [all_reconstructions[i] for i in job_indices]
            # <<< NEW: Get direct scores for the job >>>
            job_direct_scores = all_direct_scores[job_indices] if can_calculate_direct_stats else np.array([])


            # Calculate job-specific RECONSTRUCTION ERROR stats
            job_stats = {
                'count': len(job_errors),
                'mean_error': float(np.mean(job_errors)),
                'median_error': float(np.median(job_errors)),
                'std_error': float(np.std(job_errors)),
                'min_error': float(np.min(job_errors)),
                'max_error': float(np.max(job_errors)),
                'anomaly_count': int(np.sum(job_errors > self.high_threshold)) if self.high_threshold is not None else np.nan,
                'anomaly_percentage': float(100 * np.sum(job_errors > self.high_threshold) / len(job_errors)) if self.high_threshold is not None else np.nan
            }
            # <<< NEW: Calculate job-specific DIRECT SCORE stats >>>
            if job_direct_scores.size > 0:
                 finite_job_direct_scores = job_direct_scores[np.isfinite(job_direct_scores)]
                 if finite_job_direct_scores.size > 0:
                      job_stats['direct_score_mean'] = float(np.mean(finite_job_direct_scores))
                      job_stats['direct_score_median'] = float(np.median(finite_job_direct_scores))
                      job_stats['direct_score_std'] = float(np.std(finite_job_direct_scores))


            logger.info(f"  {job_type.capitalize()} (n={len(job_indices)}): Mean Error={job_stats['mean_error']:.4f}, "
                        f"Anomalies={job_stats['anomaly_percentage']:.2f}%")

            # --- Generate Job-Specific Visualizations ---
            job_stats['visualizations'] = {} # Initialize visualization paths
            try:
                viz_output_dir = self.config.output_dir / 'visualization' / 'evaluation'
                viz_output_dir.mkdir(parents=True, exist_ok=True)

                # Create error distribution plot
                error_viz_path = create_reconstruction_error_visualization(
                    job_type=job_type,
                    error_values=job_errors,
                    job_categories=job_categories, # Pass all categories for context if needed by viz
                    output_dir=viz_output_dir
                )
                if error_viz_path: job_stats['visualizations']['error_distribution'] = str(error_viz_path)

                # Create sequence comparison plot (e.g., for highest error examples)
                if len(job_errors) > 0:
                    top_error_indices = np.argsort(job_errors)[-5:] # Get indices of 5 highest errors
                    top_job_seqs = [job_seqs[i] for i in top_error_indices]
                    top_job_recons = [job_recons[i] for i in top_error_indices]
                    top_job_errors = [job_errors[i] for i in top_error_indices]

                    seq_viz_path = visualize_original_vs_reconstructed(
                        job_type=f"{job_type}_top_errors", # More specific name
                        original_sequences=top_job_seqs,
                        reconstructed_sequences=top_job_recons,
                        reconstruction_errors=top_job_errors,
                        sample_count=3, # Show top 3
                        output_dir=viz_output_dir
                    )
                    if seq_viz_path: job_stats['visualizations']['sequence_comparison'] = str(seq_viz_path)

            except Exception as viz_err:
                logger.error(f"Error generating visualizations for {job_type}: {viz_err}", exc_info=True)

            # Store job results
            job_results[job_type] = {
                'errors': job_errors.tolist(), # Store as list for JSON compatibility if needed
                'direct_scores': job_direct_scores.tolist(), # Also store direct scores
                'stats': job_stats
            }

        # --- Generate Overall Visualizations (for reconstruction error) ---
        stats['visualizations'] = {} # Initialize overall visualization paths
        try:
            overall_viz_output_dir = self.config.output_dir / 'visualization' / 'evaluation'
            overall_viz_output_dir.mkdir(parents=True, exist_ok=True)

            # Overall error distribution
            overall_error_viz_path = create_reconstruction_error_visualization(
                job_type="overall",
                error_values=all_errors,
                job_categories=job_categories,
                output_dir=overall_viz_output_dir
            )
            if overall_error_viz_path: stats['visualizations']['overall_distribution'] = str(overall_error_viz_path)

            # Overall sequence examples (low, medium, high error)
            if self.low_threshold is not None and self.high_threshold is not None:
                 error_levels = [
                     (all_errors < self.low_threshold, "low_error"),
                     ((all_errors >= self.low_threshold) & (all_errors < self.high_threshold), "normal_error"),
                     (all_errors >= self.high_threshold, "high_error")
                 ]
                 # ...(logic for plotting sequence examples remains the same)...
                 for mask, level_name in error_levels:
                      indices = np.where(mask)[0]
                      if len(indices) > 0:
                           sample_indices = np.random.choice(indices, size=min(2, len(indices)), replace=False)
                           sample_seqs = [all_sequences[i] for i in sample_indices]
                           sample_recons = [all_reconstructions[i] for i in sample_indices]
                           sample_errors = [all_errors[i] for i in sample_indices]
                           level_viz_path = visualize_original_vs_reconstructed(
                                job_type=f"overall_{level_name}",
                                original_sequences=sample_seqs,
                                reconstructed_sequences=sample_recons,
                                reconstruction_errors=sample_errors,
                                sample_count=len(sample_indices),
                                output_dir=overall_viz_output_dir
                           )
                           if level_viz_path: stats['visualizations'][f'sequence_examples_{level_name}'] = str(level_viz_path)
            else:
                 logger.warning("Cannot generate level-specific sequence examples: Thresholds missing.")

        except Exception as overall_viz_err:
            logger.error(f"Error creating overall visualizations: {overall_viz_err}", exc_info=True)


        # Add overall stats to results
        job_results['overall'] = {
            'errors': all_errors.tolist(), # Store as list
            'direct_scores': all_direct_scores.tolist(), # Store overall direct scores
            'stats': stats
        }

        # --- Final Save of evaluation_results ---
        # ...(Saving logic remains the same)...
        try:
            results_path = self.config.output_dir / 'evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(job_results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) # Added default handler for numpy arrays just in case
            logger.info(f"Full evaluation results saved to: {results_path}")
        except TypeError as json_err:
             logger.error(f"Could not serialize evaluation results to JSON: {json_err}. Check for NumPy arrays or other non-serializable types.")
        except Exception as save_res_err:
             logger.error(f"Error saving evaluation results JSON: {save_res_err}")


        return job_results

    # --- analyze_error_distribution, visualize_error_distribution, find_anomalies remain the same ---
    # They primarily operate on reconstruction error, which is unchanged.
    # visualize_error_distribution was already updated to correctly handle the 'errors' key.

    def analyze_error_distribution(self, val_loader, num_bins=20):
        """
        Analyze reconstruction error distribution for validation data.

        Args:
            val_loader: DataLoader containing validation data
            num_bins: Number of bins for histogram

        Returns:
            dict: Error distribution analysis
        """
        if val_loader is None or not hasattr(val_loader, 'dataset') or len(val_loader.dataset) == 0:
            logger.warning("No validation loader provided or loader is empty.")
            return None

        errors_by_job = defaultdict(list)
        with torch.no_grad():
            for seq, _, job_id in val_loader: # We don't need the label (direct score) here
                seq = seq.to(self.device)
                job_id = job_id.to(self.device)
                job_condition = job_id if self.model_config.get('conditional', False) else None
                reconstructed_seq = self.model(seq, job_condition)
                batch_errors = torch.mean((reconstructed_seq - seq) ** 2, dim=(1, 2)).cpu().numpy()
                for i, j_id in enumerate(job_id.cpu().numpy()):
                    job_type = self.config.job_categories[j_id]
                    errors_by_job[job_type].append(batch_errors[i])

        # Calculate statistics
        error_distribution = {}
        all_errors = []
        for job_type, errors in errors_by_job.items():
            errors_np = np.array(errors) # Convert to numpy array
            all_errors.extend(errors_np)
            finite_errors = errors_np[np.isfinite(errors_np)] # Use only finite errors for stats
            if len(finite_errors) == 0:
                 logger.warning(f"No finite errors found for job type {job_type} in analyze_error_distribution.")
                 hist, bin_edges = [], []
                 stats = {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'count': 0, 'errors': np.array([])}
            else:
                 hist, bin_edges = np.histogram(finite_errors, bins=num_bins)
                 stats = {
                    'mean': float(np.mean(finite_errors)),
                    'median': float(np.median(finite_errors)),
                    'std': float(np.std(finite_errors)),
                    'min': float(np.min(finite_errors)),
                    'max': float(np.max(finite_errors)),
                    'count': len(finite_errors),
                    'errors': finite_errors # Store finite errors for visualization
                 }

            error_distribution[job_type] = {
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                **stats # Unpack the stats dictionary
            }

        # Calculate overall statistics
        all_errors = np.array(all_errors)
        finite_all_errors = all_errors[np.isfinite(all_errors)]
        if len(finite_all_errors) == 0:
             logger.warning("No finite errors found overall in analyze_error_distribution.")
             hist_all, bin_edges_all = [], []
             overall_stats = {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'count': 0}
        else:
             hist_all, bin_edges_all = np.histogram(finite_all_errors, bins=num_bins)
             overall_stats = {
                 'mean': float(np.mean(finite_all_errors)),
                 'median': float(np.median(finite_all_errors)),
                 'std': float(np.std(finite_all_errors)),
                 'min': float(np.min(finite_all_errors)),
                 'max': float(np.max(finite_all_errors)),
                 'count': len(finite_all_errors)
             }

        error_distribution['overall'] = {
            'histogram': hist_all.tolist(),
            'bin_edges': bin_edges_all.tolist(),
            **overall_stats
        }

        # Visualize error distribution
        self.visualize_error_distribution(error_distribution)

        return error_distribution

    def visualize_error_distribution(self, error_distribution):
        """
        Create visualization of error distribution across job types.

        Args:
            error_distribution: Dict containing error statistics

        Returns:
            Path: Path to saved visualization
        """
        try:
            output_dir = self.config.output_dir / 'visualization' / 'evaluation'
            output_dir.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: Error distributions by job type
            ax1 = axes[0]
            job_types = [jt for jt in error_distribution.keys() if jt != 'overall']

            for job_type in job_types:
                stats = error_distribution[job_type]
                # Ensure histogram data is valid before plotting
                if stats.get('histogram') and stats.get('bin_edges') and len(stats['histogram']) == len(stats['bin_edges']) - 1:
                    bin_centers = (np.array(stats['bin_edges'][:-1]) + np.array(stats['bin_edges'][1:])) / 2
                    ax1.plot(bin_centers, stats['histogram'], label=f"{job_type.capitalize()} (n={stats['count']})")
                else:
                    logger.warning(f"Skipping histogram plot for {job_type} due to missing/invalid data.")


            ax1.set_title("Reconstruction Error Distribution by Job Type", fontsize=14)
            ax1.set_xlabel("Reconstruction Error", fontsize=12)
            ax1.set_ylabel("Count", fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Box plot comparison
            ax2 = axes[1]
            box_data = []
            box_labels = []
            for job_type in job_types:
                # Use the stored 'errors' array which should contain only finite values now
                job_errors = error_distribution[job_type].get('errors')
                if isinstance(job_errors, np.ndarray) and job_errors.size > 0:
                    box_data.append(job_errors)
                    box_labels.append(job_type.capitalize())

            if box_data:
                ax2.boxplot(box_data, labels=box_labels, showfliers=True)
                ax2.set_title("Error Distribution Comparison (Box Plot)", fontsize=14)
                ax2.set_ylabel("Reconstruction Error", fontsize=12)
                ax2.grid(True, alpha=0.3, axis='y')
            else:
                ax2.text(0.5, 0.5, "No finite error data available for box plot", ha='center', va='center')
                ax2.set_title("Error Distribution Comparison (Box Plot)", fontsize=14)

            plt.tight_layout()

            output_path = output_dir / "error_distribution_comparison.png"
            plt.savefig(output_path, dpi=150)
            plt.close(fig)

            logger.info(f"Error distribution visualization saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating error distribution visualization: {e}", exc_info=True)
            plt.close('all')
            return None

    def find_anomalies(self, val_loader, output_limit=10):
        """
        Find and visualize sequences with highest reconstruction errors.

        Args:
            val_loader: DataLoader containing validation data
            output_limit: Maximum number of anomalies to return

        Returns:
            list: Indices and errors of top anomalies
        """
        if self.high_threshold is None:
            logger.warning("High threshold for reconstruction error is not set. Cannot find anomalies.")
            # Optionally, run evaluate_model again, but this might be redundant if called before.
            # self.evaluate_model(val_loader)
            # if self.high_threshold is None: # Check again
            #     logger.error("Could not establish error thresholds even after re-evaluation.")
            return [] # Return empty list if threshold is missing


        errors = []
        indices = []
        sequences = []
        reconstructions = []
        job_ids = []

        with torch.no_grad():
            batch_idx = 0
            for seq, _, job_id in val_loader: # Don't need label here
                seq = seq.to(self.device)
                job_id = job_id.to(self.device)
                job_condition = job_id if self.model_config.get('conditional', False) else None
                reconstructed_seq = self.model(seq, job_condition)
                batch_errors = torch.mean((reconstructed_seq - seq) ** 2, dim=(1, 2)).cpu().numpy()

                for i in range(len(batch_errors)):
                    global_idx = batch_idx * val_loader.batch_size + i
                    errors.append(batch_errors[i])
                    indices.append(global_idx)
                    sequences.append(seq[i].cpu().numpy())
                    reconstructions.append(reconstructed_seq[i].cpu().numpy())
                    job_ids.append(job_id[i].item()) # Use .item() for single value tensor

                batch_idx += 1

        # Filter out non-finite errors before sorting
        finite_mask = np.isfinite(errors)
        errors = np.array(errors)[finite_mask]
        indices = np.array(indices)[finite_mask]
        sequences = [sequences[i] for i, mask_val in enumerate(finite_mask) if mask_val]
        reconstructions = [reconstructions[i] for i, mask_val in enumerate(finite_mask) if mask_val]
        job_ids = [job_ids[i] for i, mask_val in enumerate(finite_mask) if mask_val]


        # Sort by error (highest first)
        sorted_indices_local = np.argsort(errors)[::-1] # Sort based on filtered errors

        # Get top anomalies
        top_anomalies = []
        for i in range(min(output_limit, len(sorted_indices_local))):
            local_idx = sorted_indices_local[i]
            error = errors[local_idx] # Get error from filtered array

            if error > self.high_threshold:  # Only include actual anomalies
                original_index = indices[local_idx] # Get original index
                job_type = self.config.job_categories[job_ids[local_idx]] # Get job type using job_id
                top_anomalies.append({
                    'index': int(original_index), # Ensure it's standard int
                    'error': float(error),
                    'job_type': job_type,
                    'is_anomaly': True,
                    'sequence': sequences[local_idx], # Store as numpy array or convert to list later
                    'reconstruction': reconstructions[local_idx] # Store as numpy array or convert to list later
                })

        # Visualize top anomalies
        try:
            if top_anomalies:
                output_dir = self.config.output_dir / 'visualization' / 'anomalies'
                output_dir.mkdir(parents=True, exist_ok=True)

                # Group by job type
                anomalies_by_job = defaultdict(list)
                for anomaly in top_anomalies:
                    anomalies_by_job[anomaly['job_type']].append(anomaly)

                # Create visualizations for each job type
                for job_type, job_anomalies in anomalies_by_job.items():
                    job_sequences = [a['sequence'] for a in job_anomalies]
                    job_reconstructions = [a['reconstruction'] for a in job_anomalies]
                    job_errors = [a['error'] for a in job_anomalies]

                    viz_path = visualize_original_vs_reconstructed(
                        job_type=f"{job_type}_anomalies",
                        original_sequences=job_sequences,
                        reconstructed_sequences=job_reconstructions,
                        reconstruction_errors=job_errors,
                        sample_count=min(5, len(job_anomalies)),
                        output_dir=output_dir
                    )

                    if viz_path:
                        logger.info(f"Top anomalies visualization for {job_type} saved to: {viz_path}")

        except Exception as viz_err:
            logger.error(f"Error visualizing anomalies: {viz_err}", exc_info=True)

        return top_anomalies