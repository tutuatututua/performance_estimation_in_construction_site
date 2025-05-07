import matplotlib
matplotlib.use('Agg') # Force matplotlib to use Agg backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score # Likely not needed
from utils.pose_extraction import normalize_joints, calculate_displacement # Might be needed
import json
import seaborn as sns
from infer import MovementClassifier # Import from infer.py # Likely not needed
import logging
import torch
import pandas as pd

from matplotlib.patches import Patch
import io 
import cv2 
from collections import defaultdict
import traceback
# Configure logging
logger = logging.getLogger(__name__)


def visualize_class_distribution(job_type, predicted_classes, true_classes, predicted_counts,
                                 true_counts, config):
    """
    Create visualization of predicted vs true class distributions. Handles potential zero divisions.
    (This function is likely not needed for autoencoders, but keeping it for now.
     It might be adapted later for analyzing error distributions per class.)
    """
    # ... (implementation unchanged - but likely to be removed or heavily modified later) ...
    try:
        output_dir = config.output_dir / 'visualization'
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(16, 12))
        class_names = ["LOW", "NORMAL", "HIGH"]

        # --- 1. Confusion Matrix ---
        ax1 = plt.subplot(2, 2, 1)
        cm = confusion_matrix(true_classes, predicted_classes, labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'Confusion Matrix - {job_type.capitalize()}', fontsize=14)
        ax1.set_ylabel('True Class', fontsize=12)
        ax1.set_xlabel('Predicted Class', fontsize=12)

        # --- 2. Normalized Confusion Matrix (Handled Division by Zero) ---
        ax_norm = plt.subplot(2, 2, 2) # Added separate subplot for normalized
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_pct = cm.astype('float') / row_sums * 100
            cm_pct = np.nan_to_num(cm_pct)
        cm_pct = np.round(cm_pct, 1) # Round percentages
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax_norm, vmin=0, vmax=100)
        ax_norm.set_title(f'Normalized CM (%) - {job_type.capitalize()}', fontsize=14)
        ax_norm.set_ylabel('True Class', fontsize=12)
        ax_norm.set_xlabel('Predicted Class', fontsize=12)

        # --- 3. Prediction Distribution ---
        ax_pred = plt.subplot(2, 2, 3) # Changed subplot index
        low_count_pred = predicted_counts.get('LOW', 0)
        normal_count_pred = predicted_counts.get('NORMAL', 0)
        high_count_pred = predicted_counts.get('HIGH', 0)
        bars_pred = ax_pred.bar(class_names, [low_count_pred, normal_count_pred, high_count_pred],
                                color=['blue', 'green', 'red'])
        ax_pred.bar_label(bars_pred, padding=3) # Add counts above bars
        ax_pred.set_title(f'Predicted Class Distribution - {job_type.capitalize()}', fontsize=14)
        ax_pred.set_ylabel('Count', fontsize=12)
        ax_pred.grid(axis='y', alpha=0.3)

        # --- 4. True Distribution ---
        ax_true = plt.subplot(2, 2, 4) # Changed subplot index
        low_count_true = true_counts.get('LOW', 0)
        normal_count_true = true_counts.get('NORMAL', 0)
        high_count_true = true_counts.get('HIGH', 0)
        bars_true = ax_true.bar(class_names, [low_count_true, normal_count_true, high_count_true],
                                color=['blue', 'green', 'red'])
        ax_true.bar_label(bars_true, padding=3) # Add counts above bars
        ax_true.set_title(f'True Class Distribution - {job_type.capitalize()}', fontsize=14)
        ax_true.set_ylabel('Count', fontsize=12)
        ax_true.grid(axis='y', alpha=0.3)

        if len(predicted_classes) > 0 and len(true_classes) > 0:
            overall_accuracy = np.sum(np.diag(cm)) / max(1, np.sum(cm)) * 100
            stats_text = (
                 f"Overall Stats ({job_type.capitalize()}):\n"
                 f"• Samples: {len(predicted_classes)}\n"
                 f"• Accuracy: {overall_accuracy:.2f}%"
            )
            plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=11,
                       bbox={"facecolor":"white", "alpha":0.8, "pad":5})

        plt.suptitle(f'Validation Classification Analysis - {job_type.capitalize()}', fontsize=16, fontweight='bold')
        output_path = output_dir / f"{job_type}_classification_analysis.png"
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    except Exception as e:
        logger.error(f"Error creating class distribution visualization for {job_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return None


def create_movement_plot_classification(movement_data, current_time, width, job_type=None,
                                       classifier=None, config=None):
    """
    Create enhanced movement visualization plot WITHOUT highlighted actual data points.
    """
    try:
        # Get thresholds from classifier with better fallbacks
        if classifier is None or config is None:
            logger.warning("Classifier or config not provided. Using default thresholds.")
            low_threshold = 0.2
            high_threshold = 0.6
        else:
            # Use proper thresholds with validation
            low_threshold = getattr(classifier, 'low_threshold', None)
            high_threshold = getattr(classifier, 'high_threshold', None)

            # Handle missing or invalid thresholds
            if low_threshold is None or not np.isfinite(low_threshold) or low_threshold < 0:
                logger.warning(f"Invalid low threshold: {low_threshold}, using default 0.2")
                low_threshold = 0.2

            if high_threshold is None or not np.isfinite(high_threshold) or high_threshold <= low_threshold:
                logger.warning(f"Invalid high threshold: {high_threshold}, using default {max(0.6, low_threshold * 2)}")
                high_threshold = max(0.6, low_threshold * 2)

        # Use a consistent color scheme
        anomaly_colors = {
            "LOW": '#4CAF50',    # Green
            "MEDIUM": '#FF9800',  # Orange
            "HIGH": '#F44336',     # Red
            "UNKNOWN": '#B0BEC5' # Gray
        }

        # Create figure with appropriate dimensions
        plt.figure(figsize=(width / 100, 4), dpi=110)
        plt.clf()  # Clear any existing figure
        ax = plt.gca()

        # Set time window for better visualization
        window_seconds = 10

        # Process data first to determine appropriate scaling
        has_data = False
        max_overall_error = 0
        current_anomaly_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0}
        person_error_count = 0

        # First pass - collect data ranges and validate
        all_times = []
        all_errors = []

        for person_id, data in movement_data.items():
            if 'times' not in data or 'reconstruction_errors' not in data:
                continue

            times = np.array(data['times'])
            errors = np.array(data['reconstruction_errors'])

            mask = np.isfinite(times) & np.isfinite(errors)
            if np.any(mask):
                all_times.extend(times[mask])
                all_errors.extend(errors[mask])

        # Determine max error value for consistent y-axis scaling
        y_max = high_threshold * 1.5 # Default y_max based on high threshold
        if all_errors:
            max_error_value = max(all_errors) if all_errors else 0
            # Consistent y-axis scaling logic
            consistent_scale_baseline = 0.6
            if max_error_value <= consistent_scale_baseline * 1.5:
                y_max = max(high_threshold * 1.5, consistent_scale_baseline)
            else:
                y_max = max(max_error_value * 1.2, high_threshold * 1.5)
        else:
             y_max = max(high_threshold * 1.5, 0.6)

        # Ensure y_max is at least slightly above high_threshold
        y_max = max(y_max, high_threshold + 0.05)

        # Draw background regions
        ax.axhspan(0, low_threshold, alpha=0.15, color=anomaly_colors["LOW"])
        ax.axhspan(low_threshold, high_threshold, alpha=0.15, color=anomaly_colors["MEDIUM"])
        ax.axhspan(high_threshold, y_max, alpha=0.15, color=anomaly_colors["HIGH"])

        # Add grid first (behind the data)
        ax.grid(True, alpha=0.3, linestyle=':')

        # Add threshold lines
        ax.axhline(y=low_threshold, color='darkgreen', linestyle='--', alpha=0.6, linewidth=1, label=f'Low Thresh ({low_threshold:.2f})')
        ax.axhline(y=high_threshold, color='darkred', linestyle='--', alpha=0.6, linewidth=1, label=f'High Thresh ({high_threshold:.2f})')

        # Color palette for multiple persons
        color_palette = plt.cm.tab10.colors
        current_magnitude_levels = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0}
        points_in_window = 0

        # Second pass - plot data
        for person_idx, (person_id, data) in enumerate(movement_data.items()):
            if 'times' not in data or 'reconstruction_errors' not in data:
                continue

            times = np.array(data['times'])
            errors = np.array(data['reconstruction_errors'])

            # Process anomaly levels
            anomaly_levels = []
            if 'anomaly_levels' in data and len(data['anomaly_levels']) == len(errors):
                 anomaly_levels = data['anomaly_levels']
                 valid_keys = set(anomaly_colors.keys())
                 anomaly_levels = [lvl if lvl in valid_keys else "UNKNOWN" for lvl in anomaly_levels]
            else:
                # Generate levels based on errors and thresholds if not provided
                for err in errors:
                    if not np.isfinite(err):
                        anomaly_levels.append("UNKNOWN")
                    elif err < low_threshold:
                        anomaly_levels.append("LOW")
                    elif err > high_threshold:
                        anomaly_levels.append("HIGH")
                    else:
                        anomaly_levels.append("MEDIUM")

            # Filter out invalid values and limit to time window
            window_mask = (times >= current_time - window_seconds) & (times <= current_time + 1)
            valid_mask = np.isfinite(times) & np.isfinite(errors) & window_mask

            if not np.any(valid_mask):
                continue

            times_valid = times[valid_mask]
            errors_valid = errors[valid_mask]

            if len(times_valid) < 2:
                continue

            has_data = True
            person_error_count += 1
            max_overall_error = max(max_overall_error, np.nanmax(errors_valid))

            # Get anomaly levels within window
            anomaly_levels_valid_indices = np.where(valid_mask)[0]
            anomaly_levels_valid = [anomaly_levels[i] for i in anomaly_levels_valid_indices if i < len(anomaly_levels)]

            # Use consistent color for each person
            person_color = color_palette[person_idx % len(color_palette)]

            # Plot this person's error line
            ax.plot(times_valid, errors_valid,
                   color=person_color,
                   linewidth=1.5, # Keep line visible
                   alpha=0.7,
                   marker=None,
                   label=f"Person {person_id}")

            # --- REMOVED BOLD MARKER LOGIC ---

            # Count current anomaly levels (for last 5 seconds)
            current_window_mask_5s = (times_valid >= current_time - 5) & (times_valid <= current_time)
            indices_in_5s_window = np.where(current_window_mask_5s)[0]

            # Get levels corresponding to the indices within the 5s window
            anomaly_levels_in_5s_window = [anomaly_levels_valid[i] for i in indices_in_5s_window if i < len(anomaly_levels_valid)]

            # Increment counts correctly
            for level in anomaly_levels_in_5s_window:
                 level_key = level if level in current_anomaly_counts else "UNKNOWN"
                 current_anomaly_counts[level_key] += 1
                 points_in_window += 1

        # Add current time indicator with animation-like effect
        if has_data:
            ax.axvline(x=current_time, color='black', linewidth=1.5, alpha=0.8, linestyle='-')

        # Add summary metrics to the plot
        if has_data:
            # Determine dominant anomaly level in the current window
            valid_counts = {k: v for k, v in current_anomaly_counts.items() if k != "UNKNOWN" and v > 0}
            total_valid_detections = sum(valid_counts.values())

            if total_valid_detections > 0:
                dominant_anomaly_level = max(valid_counts, key=valid_counts.get)
                dominant_color = anomaly_colors.get(dominant_anomaly_level, anomaly_colors["UNKNOWN"])
                status_text = f"STATUS: {dominant_anomaly_level}"
            elif current_anomaly_counts["UNKNOWN"] > 0:
                 dominant_anomaly_level = "UNKNOWN"
                 dominant_color = anomaly_colors["UNKNOWN"]
                 status_text = "STATUS: UNKNOWN"
            else:
                 dominant_anomaly_level = "NONE"
                 dominant_color = '#E0E0E0'
                 status_text = "STATUS: NO DATA"

            # Calculate percentages based on ALL detections in the window
            total_detections_all = sum(current_anomaly_counts.values())

            metrics_text = f"Analysis (last 5s):\n"
            if total_detections_all > 0:
                 metrics_lines = []
                 for level in ["LOW", "MEDIUM", "HIGH", "UNKNOWN"]:
                     count = current_anomaly_counts[level]
                     if count > 0:
                         pct = count / total_detections_all * 100
                         metrics_lines.append(f"  {level}: {pct:.0f}% ({count})")
                 metrics_text += "\n".join(metrics_lines)
                 metrics_text += f"\nTotal points: {total_detections_all}"
            else:
                 metrics_text += "  No points in window."

            # Improved box styling for metrics
            props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85)
            ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='bottom', bbox=props)

            # Improved status box
            ax.text(0.98, 0.98, status_text,
                   transform=ax.transAxes, fontsize=10,
                   horizontalalignment='right', verticalalignment='top',
                   bbox=dict(facecolor=dominant_color, alpha=0.7, boxstyle='round,pad=0.4'),
                   color='white' if dominant_anomaly_level not in ["NONE", "UNKNOWN"] else 'black',
                   fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No movement data detected in window", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='grey')

        # Set proper axis limits
        x_max = current_time + 1
        x_min = max(0, current_time - window_seconds)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)

        # Add labels and title
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Reconstruction Error', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)

        # Improved title
        title = 'Movement Anomaly Analysis'
        if job_type:
            title += f' - {job_type.capitalize()}'
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Enhanced legend - Placed outside
        handles, labels = ax.get_legend_handles_labels()

        # Only add person labels if there are persons plotted
        person_handles = [h for h, l in zip(handles, labels) if l.startswith("Person")]
        person_labels = [l for l in labels if l.startswith("Person")]

        # Add threshold labels from handles
        threshold_handles = [h for h, l in zip(handles, labels) if l.endswith("Thresh")]
        threshold_labels = [l for l in labels if l.endswith("Thresh")]

        # Combine person and threshold legend items
        combined_handles = person_handles + threshold_handles
        combined_labels = person_labels + threshold_labels

        if combined_handles:
            # Limit persons shown in legend if there are many
            max_persons_in_legend = 3
            if len(person_handles) > max_persons_in_legend:
                combined_handles = person_handles[:max_persons_in_legend] + threshold_handles
                combined_labels = person_labels[:max_persons_in_legend] + [f'... ({len(person_handles)-max_persons_in_legend} more)'] + threshold_labels

            legend = ax.legend(combined_handles, combined_labels,
                                loc='upper center', fontsize=8,
                                bbox_to_anchor=(0.5, -0.18),
                                ncol=min(4, len(combined_handles)),
                                framealpha=0.8)

        # Tight layout to ensure everything fits
        plt.tight_layout(rect=[0, 0.1, 1, 1])

        # Create the plot image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plot_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if plot_img is None:
             logger.error("cv2.imdecode failed to decode plot image.")
             error_img = np.ones((400, width, 3), dtype=np.uint8) * 200
             cv2.putText(error_img, "Plot Decode Error", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             return error_img

        # Resize to target dimensions if necessary
        target_height = 400
        target_width = width
        if plot_img.shape[0] != target_height or plot_img.shape[1] != target_width:
            plot_img = cv2.resize(plot_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

        return plot_img

    except Exception as e:
        logger.error(f"Error creating movement anomaly plot: {e}", exc_info=True)
        plt.close('all')

        # Create an error message image
        error_img = np.ones((400, width, 3), dtype=np.uint8) * 220
        font_scale = 0.6
        thickness = 1
        color = (0, 0, 255)
        y_pos = 50
        error_lines = traceback.format_exc().splitlines()[-3:]
        lines_to_draw = [f"Plot Error: {str(e)[:70]}"] + error_lines

        for line in lines_to_draw:
             cv2.putText(error_img, line[:80],
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
             y_pos += int(font_scale * 30)

        return error_img
        
def create_comprehensive_metrics_visualization(model, config, training_data, val_loaders, model_info):
    """
    Create a comprehensive visualization summarizing model performance,
    adapted for autoencoder reconstruction error.

    Args:
        model: The trained PyTorch model (LSTM Autoencoder).
        config: The configuration object.
        training_data (dict): Dictionary containing training history
                              (e.g., 'epochs', 'train_losses', 'val_losses', 'learning_rates').
        val_loaders (dict):  (UNUSED for now) Dictionary mapping job_type to its validation DataLoader.
        model_info (dict):   (UNUSED for now) Dictionary loaded alongside the model (may contain metadata).

    Returns:
        Path: Path to the saved visualization file, or None if failed.
    """
    try:
        output_dir = Path(config.output_dir) / 'visualization'
        output_dir.mkdir(parents=True, exist_ok=True)
        device = config.device
        # class_names = ["LOW", "NORMAL", "HIGH"] # Not relevant for AE

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # Adjust layout
        fig.suptitle('Autoencoder Performance Metrics', fontsize=16, fontweight='bold')

        # --- Plot 1: Reconstruction Loss ---
        ax_loss = axes[0]
        if training_data and 'epochs' in training_data and 'train_losses' in training_data and 'val_losses' in training_data and training_data['epochs']:
            epochs = training_data['epochs']
            train_loss = training_data['train_losses']
            val_loss = training_data['val_losses']
            ax_loss.plot(epochs, train_loss, 'b-', label='Training Loss', alpha=0.8)
            ax_loss.plot(epochs, val_loss, 'r-', label='Validation Loss', alpha=0.8)
            if val_loss:
                best_epoch_idx = np.nanargmin(val_loss)
                best_epoch = epochs[best_epoch_idx]
                best_loss = val_loss[best_epoch_idx]
                if np.isfinite(best_loss):
                    ax_loss.plot(best_epoch, best_loss, 'r*', markersize=12, label=f'Best Val Loss: {best_loss:.4f}')
            ax_loss.set_title('Reconstruction Loss', fontsize=14)
            ax_loss.set_xlabel('Epoch', fontsize=12)
            ax_loss.set_ylabel('MSE Loss', fontsize=12) # Changed label
            ax_loss.grid(True, alpha=0.3)
            ax_loss.legend(fontsize=10)
        else:
            ax_loss.text(0.5, 0.5, 'Loss data unavailable', ha='center', va='center')
            ax_loss.set_title('Reconstruction Loss', fontsize=14)

        # --- Plot 2: Learning Rate ---
        ax_lr = axes[1]
        if training_data and 'epochs' in training_data and 'learning_rates' in training_data and training_data['epochs']:
            epochs = training_data['epochs']
            learning_rates = training_data['learning_rates']
            ax_lr.plot(epochs, learning_rates, 'c.-', label='Learning Rate')
            ax_lr.set_title('Learning Rate Schedule', fontsize=14)
            ax_lr.set_xlabel('Epoch', fontsize=12)
            ax_lr.set_ylabel('Learning Rate', fontsize=12)
            ax_lr.grid(True, alpha=0.3)
            ax_lr.set_yscale('log')
            ax_lr.legend(fontsize=10)
        else:
            ax_lr.text(0.5, 0.5, 'Learning rate data unavailable', ha='center', va='center')
            ax_lr.set_title('Learning Rate Schedule', fontsize=14)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = output_dir / 'autoencoder_metrics.png' # Changed filename
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Comprehensive autoencoder metrics visualization saved to: {output_path}")
        return output_path

    except ImportError as e:
        logger.error(f"Import error during visualization: {e}. Install matplotlib, seaborn, scipy.")
        plt.close('all')
        return None
    except Exception as e:
        logger.error(f"Error creating comprehensive autoencoder metrics visualization: {str(e)}", exc_info=True)
        plt.close('all')
        return None


def create_activity_distribution_plot(movement_data, job_type=None, output_dir=None):
    """Create a time-series plot showing activity level distribution over time.
       This function needs to be adapted to use reconstruction error instead of activity scores.
    """
    if not movement_data:
        return None

    try:
        # Collect all time points and corresponding reconstruction errors
        all_times = []
        all_anomaly_levels = []  # Renamed from all_classes

        for person_id, data in movement_data.items():
            if 'times' in data and 'reconstruction_errors' in data: # Changed from 'scores'
                times = np.array(data['times'])
                errors = np.array(data['reconstruction_errors']) # Changed from scores

                for i, time in enumerate(times):
                    error = errors[i]
                    if error < 30.0:  # Example anomaly thresholds
                        anomaly_level = "LOW"
                    elif error > 70.0:
                        anomaly_level = "HIGH"
                    else:
                        anomaly_level = "MEDIUM" # Changed from NORMAL

                    all_times.append(time)
                    all_anomaly_levels.append(anomaly_level) # Renamed

        # Create time bins
        if all_times:
            time_bins = np.arange(0, max(all_times) + 10, 10)  # 10-second bins

            fig, ax = plt.subplots(figsize=(12, 6))

            # Count anomaly levels in each bin
            bin_data = {"LOW": [], "MEDIUM": [], "HIGH": []} # Renamed
            for i in range(len(time_bins) - 1):
                bin_start = time_bins[i]
                bin_end = time_bins[i + 1]

                bin_indices = [idx for idx, t in enumerate(all_times)
                             if bin_start <= t < bin_end]

                bin_anomaly_levels = [all_anomaly_levels[idx] for idx in bin_indices] # Renamed

                low_count = bin_anomaly_levels.count("LOW")
                medium_count = bin_anomaly_levels.count("MEDIUM") # Renamed
                high_count = bin_anomaly_levels.count("HIGH")
                total = low_count + medium_count + high_count

                if total > 0:
                    bin_data["LOW"].append(low_count / total * 100)
                    bin_data["MEDIUM"].append(medium_count / total * 100) # Renamed
                    bin_data["HIGH"].append(high_count / total * 100)
                else:
                    bin_data["LOW"].append(0)
                    bin_data["MEDIUM"].append(0) # Renamed
                    bin_data["HIGH"].append(0)

            # Create stacked area plot
            bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

            ax.stackplot(bin_centers,
                        bin_data["LOW"], bin_data["MEDIUM"], bin_data["HIGH"], # Renamed
                        labels=['LOW', 'MEDIUM', 'HIGH'],
                        colors=['#3498db', '#f39c12', '#e74c3c'],  # Example color scheme
                        alpha=0.8)

            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Anomaly Level Distribution (%)', fontsize=12) # Changed label
            ax.set_title(f'Anomaly Level Distribution Over Time{" - " + job_type.capitalize() if job_type else ""}', # Changed title
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)

            plt.tight_layout()

            if output_dir:
                output_path = Path(output_dir) / f"anomaly_distribution_{job_type if job_type else 'all'}.png" # Changed filename
                plt.savefig(output_path, dpi=150)
                plt.close()
                return output_path
            else:
                plt.close()
                return None

    except Exception as e:
        logger.error(f"Error creating anomaly distribution plot: {e}") # Changed error message
        plt.close('all')
        return None


def create_person_movement_analysis(movement_data, job_type=None, output_dir=None):
    """Create a plot comparing movement patterns across different people.
       This function needs to be adapted to use reconstruction error.
    """
    if not movement_data:
        return None

    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Individual reconstruction error traces
        ax1 = axes[0]
        person_colors = plt.cm.tab10(np.linspace(0, 1, len(movement_data)))

        for idx, (person_id, data) in enumerate(movement_data.items()):
            if 'times' in data and 'reconstruction_errors' in data: # Changed from 'scores'
                times = np.array(data['times'])
                errors = np.array(data['reconstruction_errors']) # Changed from scores

                valid_mask = np.isfinite(times) & np.isfinite(errors)
                times = times[valid_mask]
                errors = errors[valid_mask]

                if len(times) > 0:
                    ax1.plot(times, errors, label=f'Person {person_id}',
                            color=person_colors[idx], alpha=0.7, linewidth=2)

        ax1.axhline(y=30.0, color='grey', linestyle='--', alpha=0.5, label='Low Anomaly Threshold') # Changed label
        ax1.axhline(y=70.0, color='grey', linestyle='--', alpha=0.5, label='High Anomaly Threshold') # Changed label
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Reconstruction Error', fontsize=12) # Changed label
        ax1.set_title(f'Individual Anomaly Patterns{" - " + job_type.capitalize() if job_type else ""}', # Changed title
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Person anomaly summary
        ax2 = axes[1]
        person_ids = []
        avg_errors = [] # Changed from avg_scores
        anomaly_distributions = {"LOW": [], "MEDIUM": [], "HIGH": []} # Renamed

        for person_id, data in movement_data.items():
            if 'reconstruction_errors' in data and len(data['reconstruction_errors']) > 0: # Changed from scores
                errors = np.array(data['reconstruction_errors']) # Changed from scores
                valid_errors = errors[np.isfinite(errors)]

                if len(valid_errors) > 0:
                    person_ids.append(f'Person {person_id}')
                    avg_errors.append(np.mean(valid_errors)) # Changed to errors

                    low_count = np.sum(valid_errors < 30.0)
                    high_count = np.sum(valid_errors > 70.0)
                    medium_count = len(valid_errors) - low_count - high_count # Renamed
                    total = len(valid_errors)

                    anomaly_distributions["LOW"].append(low_count / total * 100)
                    anomaly_distributions["MEDIUM"].append(medium_count / total * 100) # Renamed
                    anomaly_distributions["HIGH"].append(high_count / total * 100)

        # Create grouped bar chart
        x = np.arange(len(person_ids))
        width = 0.25

        ax2.bar(x - width, anomaly_distributions["LOW"], width, label='LOW', color='#3498db')
        ax2.bar(x, anomaly_distributions["MEDIUM"], width, label='MEDIUM', color='#f39c12') # Renamed
        ax2.bar(x + width, anomaly_distributions["HIGH"], width, label='HIGH', color='#e74c3c')

        ax2.set_xlabel('Person', fontsize=12)
        ax2.set_ylabel('Anomaly Distribution (%)', fontsize=12) # Changed label
        ax2.set_title('Anomaly Distribution by Person', fontsize=14, fontweight='bold') # Changed title
        ax2.set_xticks(x)
        ax2.set_xticklabels(person_ids, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir) / f"person_anomaly_analysis_{job_type if job_type else 'all'}.png" # Changed filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.close()
            return None

    except Exception as e:
        logger.error(f"Error creating person anomaly analysis: {e}") # Changed error message
        plt.close('all')
        return None


def create_movement_heatmap(movement_data, job_type=None, output_dir=None):
    """Create a heatmap showing movement intensity over time.
       This function needs to be adapted to show reconstruction error instead of movement score.
    """
    if not movement_data:
        return None

    try:
        # Create time bins and person grid
        all_times = []
        for data in movement_data.values():
            if 'times' in data:
                all_times.extend(data['times'])

        if not all_times:
            return None

        max_time = max(all_times)
        time_bins = np.arange(0, max_time + 5, 5)  # 5-second bins
        num_bins = len(time_bins) - 1
        num_people = len(movement_data)

        # Create heatmap data (using reconstruction errors)
        heatmap_data = np.zeros((num_people, num_bins))

        for person_idx, (person_id, data) in enumerate(movement_data.items()):
            if 'times' in data and 'reconstruction_errors' in data: # Changed from 'scores'
                times = np.array(data['times'])
                errors = np.array(data['reconstruction_errors']) # Changed from scores

                for bin_idx in range(num_bins):
                    bin_start = time_bins[bin_idx]
                    bin_end = time_bins[bin_idx + 1]

                    bin_mask = (times >= bin_start) & (times < bin_end)
                    bin_errors = errors[bin_mask] # Changed to errors

                    if len(bin_errors) > 0:
                        heatmap_data[person_idx, bin_idx] = np.mean(bin_errors) # Changed to errors

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))

        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd',
                      interpolation='nearest', vmin=0, vmax=100) # Assuming error range is 0-100

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Reconstruction Error', fontsize=12) # Changed label

        # Set labels
        ax.set_xlabel('Time Bins (5s intervals)', fontsize=12)
        ax.set_ylabel('Person ID', fontsize=12)
        ax.set_title(f'Reconstruction Error Heatmap{" - " + job_type.capitalize() if job_type else ""}', # Changed title
                    fontsize=14, fontweight='bold')

        # Set ticks
        ax.set_yticks(range(num_people))
        ax.set_yticklabels([f'Person {pid}' for pid in movement_data.keys()])

        # Set x-ticks for every 30 seconds
        x_tick_positions = range(0, num_bins, 6)  # Every 30 seconds (6 * 5s)
        x_tick_labels = [f'{i*30}s' for i in range(len(x_tick_positions))]
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels)

        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir) / f"reconstruction_error_heatmap_{job_type if job_type else 'all'}.png" # Changed filename
            plt.savefig(output_path, dpi=150)
            plt.close()
            return output_path
        else:
            plt.close()
            return None

    except Exception as e:
        logger.error(f"Error creating reconstruction error heatmap: {e}") # Changed error message
        plt.close('all')
        return None
    
def create_reconstruction_error_visualization(job_type, error_values, job_categories=None, output_dir=None):
    """
    Create visualization showing the distribution of reconstruction errors.
    
    Args:
        job_type (str): Job type for this analysis
        error_values (np.ndarray): Array of reconstruction error values
        job_categories (list, optional): List of all job categories
        output_dir (Path, optional): Directory to save visualization
        
    Returns:
        Path: Path to the saved visualization, or None if failed
    """
    if not isinstance(error_values, np.ndarray) or error_values.size == 0:
        logger.error(f"Invalid error values for visualization: {job_type}")
        return None
        
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Reconstruction Error Analysis: {job_type.capitalize()}", fontsize=16, fontweight='bold')
        axes = axes.flatten()
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # --- Plot 1: Error Distribution ---
        ax = axes[0]
        sns.histplot(error_values, kde=True, stat="density", alpha=0.6,
                    label=f"Reconstruction Errors (n={len(error_values)})", color="#1f77b4", ax=ax)
        
        mean_error = np.mean(error_values)
        median_error = np.median(error_values)
        ax.axvline(mean_error, color="#1f77b4", linestyle='--', label=f"Mean: {mean_error:.2f}")
        ax.axvline(median_error, color="#ff7f0e", linestyle='--', label=f"Median: {median_error:.2f}")
        
        # Determine appropriate thresholds for anomaly detection
        # Using percentiles for a data-driven approach
        low_thresh = np.percentile(error_values, 15)
        high_thresh = np.percentile(error_values, 95)  # Higher percentile for anomalies
        
        ax.axvline(low_thresh, color="green", linestyle='-', alpha=0.5, 
                  label=f"Low Threshold (15%): {low_thresh:.2f}")
        ax.axvline(high_thresh, color="red", linestyle='-', alpha=0.5, 
                   label=f"High Threshold (95%): {high_thresh:.2f}")
        
        ax.set_title("Reconstruction Error Distribution", fontsize=14)
        ax.set_xlabel("Error Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_xlim(left=0)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # --- Plot 2: Anomaly Detection ---
        ax = axes[1]
        # Identify anomalies using thresholds
        normal = error_values[(error_values >= low_thresh) & (error_values < high_thresh)]
        anomalies = error_values[error_values >= high_thresh]
        low_errors = error_values[error_values < low_thresh]
        
        # Plot categories as stacked bars
        categories = ['Low Error', 'Normal', 'Anomaly']
        counts = [len(low_errors), len(normal), len(anomalies)]
        percentages = [100 * c / len(error_values) for c in counts]
        
        bars = ax.bar(categories, counts, color=['green', 'blue', 'red'], alpha=0.7)
        
        # Add count and percentage labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f"{counts[i]}\n({percentages[i]:.1f}%)",
                   ha='center', va='bottom', fontsize=10)
                   
        ax.set_title("Error Categories Breakdown", fontsize=14)
        ax.set_ylabel("Count", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        
        # --- Plot 3: Error Over Sequence Index ---
        ax = axes[2]
        if len(error_values) > 1:
            # Plot error as sequence
            x = np.arange(len(error_values))
            sorted_errors = np.sort(error_values)
            ax.plot(x, sorted_errors, marker='o', markersize=3, alpha=0.5, linewidth=1)
            
            # Highlight thresholds
            ax.axhline(y=low_thresh, color='green', linestyle='--', alpha=0.7)
            ax.axhline(y=high_thresh, color='red', linestyle='--', alpha=0.7)
            
            # Color regions
            ax.fill_between(x, 0, low_thresh, alpha=0.1, color='green')
            ax.fill_between(x, low_thresh, high_thresh, alpha=0.1, color='blue')
            ax.fill_between(x, high_thresh, max(sorted_errors)*1.1, alpha=0.1, color='red')
            
            ax.set_title("Sorted Reconstruction Errors", fontsize=14)
            ax.set_xlabel("Sequence Index (sorted)", fontsize=12)
            ax.set_ylabel("Reconstruction Error", fontsize=12)
        else:
            ax.text(0.5, 0.5, "Insufficient data for sequence plot", ha='center', va='center')
            
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # --- Plot 4: Statistics Table ---
        ax = axes[3]
        # Calculate statistics
        stats = {
            'Mean': f"{mean_error:.4f}",
            'Median': f"{median_error:.4f}",
            'Std Dev': f"{np.std(error_values):.4f}",
            'Min': f"{np.min(error_values):.4f}",
            'Max': f"{np.max(error_values):.4f}",
            '15th %': f"{np.percentile(error_values, 15):.4f}",
            '75th %': f"{np.percentile(error_values, 75):.4f}",
            '95th %': f"{np.percentile(error_values, 95):.4f}",
            'Total Samples': f"{len(error_values)}",
            'Anomalies (>95%)': f"{len(anomalies)} ({100*len(anomalies)/len(error_values):.1f}%)"
        }
        
        # Create a table of statistics
        table_data = [[k, v] for k, v in stats.items()]
        table = ax.table(cellText=table_data, 
                       colLabels=['Statistic', 'Value'],
                       loc='center', cellLoc='center', 
                       colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.axis('off')
        ax.set_title("Error Statistics", fontsize=14, pad=20)
        
        # Save figure
        if output_dir is None:
            output_dir = Path('output/visualization')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{job_type}_reconstruction_error_analysis.png"
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating reconstruction error visualization: {e}")
        logger.error(traceback.format_exc())
        plt.close('all')
        return None


def visualize_original_vs_reconstructed(job_type, original_sequences, reconstructed_sequences, 
                                      reconstruction_errors=None, sample_count=3, output_dir=None):
    """
    Create visualization comparing original sequences to their reconstructions.
    
    Args:
        job_type (str): Job type for this analysis
        original_sequences (list): List of original input sequences
        reconstructed_sequences (list): List of reconstructed sequences from the autoencoder
        reconstruction_errors (list, optional): Corresponding reconstruction errors
        sample_count (int): Number of examples to show (default: 3)
        output_dir (Path, optional): Directory to save visualization
        
    Returns:
        Path: Path to the saved visualization, or None if failed
    """
    if not original_sequences or not reconstructed_sequences:
        logger.error("Missing sequence data for visualization.")
        return None
    
    JOINT_NAMES = ['Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist']

    # Make sure sequences are numpy arrays
    orig_seqs = [np.array(seq) for seq in original_sequences if isinstance(seq, (list, np.ndarray))]
    recon_seqs = [np.array(seq) for seq in reconstructed_sequences if isinstance(seq, (list, np.ndarray))]
    
    # Match lengths
    min_len = min(len(orig_seqs), len(recon_seqs))
    if min_len == 0:
        logger.error("No valid sequences for comparison.")
        return None
        
    orig_seqs = orig_seqs[:min_len]
    recon_seqs = recon_seqs[:min_len]
    
    # Get errors if available, otherwise calculate
    if reconstruction_errors and len(reconstruction_errors) >= min_len:
        errors = reconstruction_errors[:min_len]
    else:
        errors = [np.mean((orig_seqs[i] - recon_seqs[i])**2) for i in range(min_len)]
    
    # Sort by error (highest first) if we have errors
    if errors:
        sorted_indices = np.argsort(errors)[::-1]  # Highest errors first
        orig_seqs = [orig_seqs[i] for i in sorted_indices]
        recon_seqs = [recon_seqs[i] for i in sorted_indices]
        errors = [errors[i] for i in sorted_indices]
    
    # Take top examples
    sample_count = min(sample_count, min_len)
    orig_samples = orig_seqs[:sample_count]
    recon_samples = recon_seqs[:sample_count]
    error_samples = errors[:sample_count] if errors else None
    
    try:
        # Create figure - one row per example, with original and reconstructed side by side
        fig, axes = plt.subplots(sample_count, 2, figsize=(14, 4 * sample_count))
        
        # Handle the case where sample_count=1 (convert to 2D array for consistent indexing)
        if sample_count == 1:
            axes = np.array([axes])
            
        plt.suptitle(f"Original vs. Reconstructed Sequences: {job_type.capitalize()}", fontsize=16)
        
        # Plot each example
        for i in range(sample_count):
            orig = orig_samples[i]
            recon = recon_samples[i]
            error = error_samples[i] if error_samples else None
            
            # Check dimensions
            if orig.ndim != 2 or recon.ndim != 2:
                logger.warning(f"Skipping example {i}: Invalid dimensions. Original: {orig.shape}, Reconstructed: {recon.shape}")
                continue
                
            # Plot original
            ax1 = axes[i, 0]
            seq_len, num_features = orig.shape
            for j in range(num_features):
                joint_name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f'Joint {j+1}'
                ax1.plot(range(seq_len), orig[:, j], label=joint_name)
            
            ax1.set_title(f"Original Sequence {i+1}" + (f" (Error: {error:.4f})" if error else ""))
            ax1.set_xlabel("Sequence Step")
            ax1.set_ylabel("Value")
            ax1.grid(True, alpha=0.3)
            
            # Only add legend to the first row to save space
            if i == 0:
                ax1.legend(loc='upper right', fontsize=9)
                
            # Plot reconstructed
            ax2 = axes[i, 1]
            for j in range(num_features):
                joint_name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f'Joint {j+1}'
                ax2.plot(range(seq_len), recon[:, j], label=joint_name)
                
            ax2.set_title(f"Reconstructed Sequence {i+1}")
            ax2.set_xlabel("Sequence Step")
            ax2.grid(True, alpha=0.3)
            # Legend for reconstructed only in first row
            if i == 0:
                ax2.legend(loc='upper right', fontsize=9)
                
            # Try to match y-axis limits for easy comparison
            y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
            y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
        
        # Save figure
        if output_dir is None:
            output_dir = Path('output/visualization')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{job_type}_sequence_reconstruction_examples.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating sequence comparison visualization: {e}")
        logger.error(traceback.format_exc())
        plt.close('all')
        return None
    

# In utils/visualization.py

def create_direct_score_plot_image(direct_score_data, current_time, width,
                                 train_mean=None, train_q1=None, train_q3=None,
                                 job_type=None, y_max=None):
    """
    Create visualization plot for direct movement scores WITHOUT highlighted actual data points.
    Compatible with both array and dictionary displacement formats.
    """
    plot_height = 400
    try:
        plt.figure(figsize=(width / 100, plot_height / 100), dpi=110)
        plt.clf()
        ax = plt.gca()

        # Define time window and setup
        window_seconds = 10
        has_data = False
        all_scores = []

        # Stats available check
        stats_available = all(v is not None and np.isfinite(v) for v in [train_mean, train_q1, train_q3])

        # First pass to collect data for y-axis scaling
        for person_id, data in direct_score_data.items():
            if 'times' not in data or 'direct_scores' not in data:
                continue
            times = np.array(data['times'])
            scores = np.array(data['direct_scores'])
            window_mask = (times >= current_time - window_seconds) & (times <= current_time + 1)
            valid_mask = np.isfinite(times) & np.isfinite(scores) & window_mask
            if np.any(valid_mask):
                all_scores.extend(scores[valid_mask])

        # Determine y-axis scaling
        if y_max is not None and np.isfinite(y_max) and y_max > 0:
            logger.debug(f"Using provided y_max={y_max:.2f} for direct score plot consistency")
        else:
            y_max = 50.0
            if all_scores:
                score_max = np.percentile(all_scores, 99.5) if len(all_scores) > 10 else np.max(all_scores)
                y_max = max(y_max, score_max * 1.3)
            if stats_available:
                y_max = max(y_max, train_q3 * 2.0)
            y_max = max(y_max, 5.0)
            y_max = min(y_max, 200.0)
            if all_scores and np.max(all_scores) > 200.0:
                y_max = np.max(all_scores) * 1.2

        # Draw background regions for stats if available
        if stats_available:
            ax.axhspan(train_q1, train_q3, alpha=0.15, color='grey')
            ax.axhline(y=train_mean, color='blue', linestyle='--', alpha=0.6, linewidth=1, label=f'Train Mean ({train_mean:.1f})')
            ax.axhline(y=train_q1, color='green', linestyle=':', alpha=0.6, linewidth=1, label=f'Train Q1 ({train_q1:.1f})')
            ax.axhline(y=train_q3, color='orange', linestyle=':', alpha=0.6, linewidth=1, label=f'Train Q3 ({train_q3:.1f})')
        else:
            ax.text(0.5, 0.6, "Training score stats unavailable", ha='center', va='center',
                  transform=ax.transAxes, fontsize=10, color='red')

        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':')

        # Plot data
        color_palette = plt.cm.tab10.colors
        points_in_window = 0

        for person_idx, (person_id, data) in enumerate(direct_score_data.items()):
            if 'times' not in data or 'direct_scores' not in data:
                continue
            times = np.array(data['times'])
            scores = np.array(data['direct_scores'])

            # Filter to window
            window_mask = (times >= current_time - window_seconds) & (times <= current_time + 1)
            valid_mask = np.isfinite(times) & np.isfinite(scores) & window_mask
            if not np.any(valid_mask): continue
            times_valid = times[valid_mask]
            scores_valid = scores[valid_mask]
            if len(times_valid) < 2: continue

            has_data = True
            person_color = color_palette[person_idx % len(color_palette)]

            # Plot the line (MODIFIED: added marker=None)
            ax.plot(times_valid, scores_valid, color=person_color,
                    linewidth=1.5, # Keep line visible
                    alpha=0.7,
                    marker=None, # Explicitly disable markers on the line
                    label=f"Person {person_id}")

            # --- REMOVED BOLD MARKER LOGIC ---
            # (The block calculating actual_indices and plotting 'o' markers was removed previously)
            # --- END REMOVED BOLD MARKER LOGIC ---


        # Add current time indicator
        if has_data:
            ax.axvline(x=current_time, color='black', linewidth=1.5, alpha=0.8, linestyle='-')

        # Set axis limits
        x_max = current_time + 1
        x_min = max(0, current_time - window_seconds)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)

        # Add labels and title
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Direct Movement Score', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        title = 'Direct Movement Score Analysis'
        if job_type: title += f' - {job_type.capitalize()}'
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Add legend and layout
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels, loc='upper center', fontsize=8,
                          bbox_to_anchor=(0.5, -0.18), framealpha=0.8)

        plt.tight_layout(rect=[0, 0.1, 1, 1])

        # Create and return image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plot_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        # Resize to target dimensions
        target_width = width
        target_height = plot_height
        if plot_img.shape[0] != target_height or plot_img.shape[1] != target_width:
            plot_img = cv2.resize(plot_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

        return plot_img

    except Exception as e:
        logger.error(f"Error creating direct score plot image: {e}", exc_info=True)
        plt.close('all')
        error_img = np.ones((plot_height, width, 3), dtype=np.uint8) * 220
        cv2.putText(error_img, f"Plot Error: {str(e)[:50]}", (20, plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return error_img