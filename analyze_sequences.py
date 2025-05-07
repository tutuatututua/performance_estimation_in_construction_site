import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import traceback
import torch
from scipy import stats

logger = logging.getLogger(__name__)

# --- Main Orchestration Function ---
def compare_movement_predictions(video_data, config, job_type=None, output_dir=None):
    """
    Create comparison visualizations of actual movement vs. AE reconstruction error
    for relevant job types found in the video_data.
    """
    if not video_data: logger.error("No video data provided for movement comparison."); return []
    if output_dir is None: output_dir = Path(config.output_dir) / 'sequence_comparison_analysis'
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving sequence comparison visualizations to: {output_dir.resolve()}")

    try:
        from infer import MovementClassifier
        movement_classifier = MovementClassifier(config)
        logger.info("MovementClassifier initialized for sequence comparison.")
    except Exception as e: logger.error(f"Error initializing MovementClassifier: {e}", exc_info=True); return []

    generated_paths = []
    job_types_in_data = set(data.get('job_type') for data in video_data.values() if data.get('job_type'))
    if not job_types_in_data: logger.warning("No job types found in video data."); return []

    job_types_to_process = sorted(list(job_types_in_data)) if job_type is None or job_type == 'all' else ([job_type] if job_type in job_types_in_data else [])
    if not job_types_to_process: logger.warning(f"Job type '{job_type}' not found or no types available."); return []

    logger.info(f"Processing sequence comparison for job types: {job_types_to_process}")
    overall_results = defaultdict(lambda: {'actual_score': [], 'model_error': []}) # Store overall results

    for current_job_type in job_types_to_process:
        logger.info(f"--- Comparing sequences for job type: {current_job_type} ---")
        all_sequences_job, all_actual_scores_job, all_model_errors_job = [], [], []

        for video_name, data in video_data.items():
            if data.get('job_type') != current_job_type: continue
            sequences = data.get('sequences', [])
            actual_scores = [ stat.get('scaled_movement', stat.get('movement_score'))
                              for stat in data.get('stats', [])
                              if stat.get('scaled_movement') is not None or stat.get('movement_score') is not None ]
            min_len = min(len(sequences), len(actual_scores))
            if min_len == 0: logger.debug(f"Skipping {video_name}: No sequences or actual scores."); continue
            sequences, actual_scores = sequences[:min_len], actual_scores[:min_len]

            # Predict reconstruction error for each sequence
            for i, seq in enumerate(sequences):
                try:
                    if not isinstance(seq, np.ndarray): seq = np.array(seq, dtype=np.float32)
                    if seq.ndim == 0 or seq.size == 0: model_error = np.nan; continue

                    # Get reconstruction error (first element returned by predict_sequence)
                    recon_error, _, _ = movement_classifier.predict_sequence(seq, current_job_type)

                    if np.isfinite(recon_error) and np.isfinite(actual_scores[i]):
                        all_sequences_job.append(seq) # Keep sequence if needed later
                        all_actual_scores_job.append(actual_scores[i])
                        all_model_errors_job.append(recon_error)
                        # Add to overall results
                        overall_results[current_job_type]['actual_score'].append(actual_scores[i])
                        overall_results[current_job_type]['model_error'].append(recon_error)
                    else:
                        # Optionally log if prediction or actual score was invalid
                         logger.debug(f"Skipping seq {i} for {video_name} due to invalid error ({recon_error}) or score ({actual_scores[i]})")

                except Exception as e: logger.error(f"Error predicting seq {i} for {video_name}: {e}", exc_info=False)

        if not all_actual_scores_job or not all_model_errors_job:
            logger.warning(f"No valid comparison data for {current_job_type}. Skipping plots."); continue

        logger.info(f"Collected {len(all_actual_scores_job)} valid comparison points for {current_job_type}.")

        # Create visualization for this job type
        try:
            job_output_path = create_job_comparison_visualization_ae( # Use AE-specific function
                current_job_type,
                np.array(all_actual_scores_job),
                np.array(all_model_errors_job), # Pass errors
                output_dir
            )
            if job_output_path: generated_paths.append(str(job_output_path))
        except Exception as e: logger.error(f"Error creating viz for {current_job_type}: {e}", exc_info=True)

    # Create overall visualization
    try:
        overall_output_path = create_overall_comparison_visualization_ae( # Use AE-specific function
            overall_results, config, output_dir
        )
        if overall_output_path: generated_paths.append(str(overall_output_path))
    except Exception as e: logger.error(f"Error creating overall viz: {e}", exc_info=True)

    return generated_paths


# --- Visualization Function for Single Job Type (AE version) ---
def create_job_comparison_visualization_ae(job_type, actual_scores, model_errors, output_dir):
    """
    Create comparison plots: Actual Score (disp.) vs Model Reconstruction Error.
    """
    if len(actual_scores) == 0 or len(model_errors) == 0 or len(actual_scores) != len(model_errors):
        logger.warning(f"Cannot create AE comparison plot for {job_type}: Invalid or mismatched data.")
        return None

    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Actual Score vs. AE Error: {job_type.capitalize()}", fontsize=16, fontweight='bold')
        axes = axes.flatten()
        plt.style.use('seaborn-v0_8-whitegrid')

        # --- Plot 1: Distributions (Score vs Error) ---
        ax = axes[0]
        sns.histplot(actual_scores, kde=True, stat="density", alpha=0.6, label=f"Actual Score (Disp.) (n={len(actual_scores)})", color="#1f77b4", ax=ax)
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Density (Actual Score)", color="#1f77b4", fontsize=12)
        ax.tick_params(axis='y', labelcolor="#1f77b4")
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

        ax_err = ax.twinx() # Share x-axis
        sns.histplot(model_errors, kde=True, stat="density", alpha=0.6, label=f"Model Error (AE) (n={len(model_errors)})", color="#ff7f0e", ax=ax_err)
        ax_err.set_ylabel("Density (Model Error)", color="#ff7f0e", fontsize=12)
        ax_err.tick_params(axis='y', labelcolor="#ff7f0e")
        ax_err.legend(loc='upper right', fontsize=10)
        ax_err.grid(False) # Turn off grid for secondary axis

        ax.set_title("Distribution Comparison", fontsize=14)
        ax.set_xlim(left=0) # Scores and Errors should be >= 0

        # --- Plot 2: Scatter Plot (Actual Score vs. Model Error) ---
        ax = axes[1]
        correlation = 'N/A'; p_value_str = 'N/A'
        # Filter non-finite values before correlation calculation
        finite_mask = np.isfinite(actual_scores) & np.isfinite(model_errors)
        actual_plot = actual_scores[finite_mask]
        error_plot = model_errors[finite_mask]

        if len(actual_plot) > 5 and np.std(actual_plot) > 1e-6 and np.std(error_plot) > 1e-6:
             ax.scatter(actual_plot, error_plot, alpha=0.4, s=15, color="#2ca02c", edgecolors='w', linewidths=0.5)
             try:
                 corr_val, p_val = stats.pearsonr(actual_plot, error_plot)
                 if np.isfinite(corr_val): correlation = f"{corr_val:.3f}"
                 if np.isfinite(p_val): p_value_str = f"{p_val:.2e}"
             except ValueError: pass # Handle constant arrays
             ax.text(0.05, 0.95, f"Correlation: {correlation}\n(p-value: {p_value_str})", transform=ax.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
             ax.set_title("Actual Score vs Model Error", fontsize=14); ax.set_xlabel("Actual Score (Displacement)", fontsize=12); ax.set_ylabel("Model Reconstruction Error", fontsize=12)
             ax.grid(True, linestyle=':', alpha=0.6); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "Insufficient data or variance for scatter/correlation", ha='center', va='center'); ax.set_title("Actual Score vs Model Error", fontsize=14)


        # --- Plot 3: Error Residuals (if needed - could be difference from a baseline error?) ---
        # Or: Plot Error vs. Time if time data is passed
        ax = axes[2]
        ax.text(0.5, 0.5, "Plot 3 Placeholder", ha='center', va='center'); ax.set_title("Placeholder", fontsize=14)


        # --- Plot 4: Statistics Table ---
        ax = axes[3]
        def get_stats(data):
            if len(data) == 0: return {k: 'N/A' for k in ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %', '75th %']}
            finite_data = data[np.isfinite(data)] # Use only finite data for stats
            if len(finite_data) == 0: return {k: 'N/A' for k in ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %', '75th %']}
            return { 'Mean': f"{np.mean(finite_data):.3f}", 'Median': f"{np.median(finite_data):.3f}", 'Std Dev': f"{np.std(finite_data):.3f}",
                     'Min': f"{np.min(finite_data):.3f}", 'Max': f"{np.max(finite_data):.3f}", '25th %': f"{np.percentile(finite_data, 25):.3f}", '75th %': f"{np.percentile(finite_data, 75):.3f}" }
        actual_stats = get_stats(actual_scores); error_stats = get_stats(model_errors)
        cell_text = [[key, actual_stats.get(key, 'N/A'), error_stats.get(key, 'N/A')]
                     for key in ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %', '75th %']]
        ax.axis('off'); table = ax.table(cellText=cell_text, colLabels=['Statistic', 'Actual Score', 'Model Error'], loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.3])
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1, 1.3)
        ax.set_title("Statistical Comparison", fontsize=14, pad=20)

        # --- Save Figure ---
        output_path = output_dir / f"{job_type}_score_vs_error_comparison.png"
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved AE comparison plot for {job_type} to {output_path}")
        return output_path

    except ImportError as e: logger.error(f"Import error for {job_type} viz: {e}. Install packages."); plt.close('all'); return None
    except Exception as e: logger.error(f"Error creating AE viz for {job_type}: {e}", exc_info=True); plt.close('all'); return None


# --- Visualization Function for Overall Comparison (AE version) ---
def create_overall_comparison_visualization_ae(overall_results, config, output_dir):
    """
    Create overall comparison plots: Actual Score (disp.) vs Model Reconstruction Error.
    """
    if not overall_results: logger.error("No overall results for AE comparison viz"); return None
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig_overall, axes_overall = plt.subplots(2, 2, figsize=(16, 12))
        fig_overall.suptitle("Overall Comparison: Actual Score vs. AE Error", fontsize=16, fontweight='bold')
        axes_overall = axes_overall.flatten()
        plt.style.use('seaborn-v0_8-whitegrid')

        all_actual_combined = []; all_errors_combined = []
        job_types = sorted(overall_results.keys())
        job_type_indices = {jt: i for i, jt in enumerate(job_types)}
        colors = plt.cm.get_cmap('tab10', len(job_types))

        # --- Plot 1: Mean Error vs. Mean Score per Job ---
        ax = axes_overall[0]
        job_means = []
        for job_type in job_types:
            actual = np.array(overall_results[job_type]['actual_score'])
            errors = np.array(overall_results[job_type]['model_error'])
            finite_mask = np.isfinite(actual) & np.isfinite(errors)
            actual, errors = actual[finite_mask], errors[finite_mask]
            if len(actual) > 0:
                 job_means.append({'job': job_type, 'mean_score': np.mean(actual), 'mean_error': np.mean(errors), 'count': len(actual)})
                 all_actual_combined.extend(actual)
                 all_errors_combined.extend(errors) # Collect combined finite data

        if job_means:
             df_means = pd.DataFrame(job_means)
             # Plot mean score on left y-axis
             bars1 = ax.bar(df_means['job'], df_means['mean_score'], color='#1f77b4', alpha=0.7, label='Mean Actual Score')
             ax.set_xlabel("Job Type", fontsize=12); ax.set_ylabel("Mean Actual Score", color='#1f77b4', fontsize=12)
             ax.tick_params(axis='y', labelcolor='#1f77b4'); ax.tick_params(axis='x', rotation=15)
             ax.grid(axis='y', linestyle='--', alpha=0.5)
             # Plot mean error on right y-axis
             ax_err = ax.twinx()
             bars2 = ax_err.bar(df_means['job'], df_means['mean_error'], color='#ff7f0e', alpha=0.7, width=0.5, label='Mean Model Error') # Slightly narrower bar
             ax_err.set_ylabel("Mean Model Error", color='#ff7f0e', fontsize=12)
             ax_err.tick_params(axis='y', labelcolor='#ff7f0e')
             ax_err.grid(False)
             ax.set_title("Mean Actual Score vs. Mean Model Error by Job", fontsize=14)
             # Combine legends
             lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax_err.get_legend_handles_labels()
             ax.legend(lines + lines2, labels + labels2, loc='upper center', fontsize=10, bbox_to_anchor=(0.5, -0.15), ncol=2) # Legend below plot
        else:
             ax.text(0.5, 0.5, "No data", ha='center', va='center')


        # --- Plot 2: Correlation by Job Type ---
        ax = axes_overall[1]; correlations = []
        for job_type in job_types:
            actual = np.array(overall_results[job_type]['actual_score'])
            errors = np.array(overall_results[job_type]['model_error'])
            finite_mask = np.isfinite(actual) & np.isfinite(errors)
            actual, errors = actual[finite_mask], errors[finite_mask]
            corr = 0.0
            if len(actual) > 5 and np.std(actual) > 1e-6 and np.std(errors) > 1e-6:
                try: corr_val, _ = stats.pearsonr(actual, errors); corr = corr_val if np.isfinite(corr_val) else 0.0
                except ValueError: pass
            correlations.append(corr)
        x = np.arange(len(job_types))
        bars_corr = ax.bar(x, correlations, color='#2ca02c')
        ax.set_title("Correlation (Actual Score vs. Model Error) by Job", fontsize=14)
        ax.set_xlabel("Job Type", fontsize=12); ax.set_ylabel("Correlation Coefficient", fontsize=12)
        ax.set_xticks(x); ax.set_xticklabels([j.capitalize() for j in job_types], rotation=15)
        ax.grid(axis='y', alpha=0.5, linestyle='--'); ax.axhline(0, color='grey', linewidth=0.8); ax.set_ylim(-1.1, 1.1) # Range -1 to 1
        for i, bar in enumerate(bars_corr): ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (0.02 if bar.get_height() >= 0 else -0.05), f"{bar.get_height():.2f}", ha='center', va=('bottom' if bar.get_height() >= 0 else 'top'), fontsize=10)


        # --- Plot 3: Overall Distributions (Score vs Error) ---
        ax = axes_overall[2]
        all_actual_combined = np.array(all_actual_combined)
        all_errors_combined = np.array(all_errors_combined)
        if len(all_actual_combined) > 0: sns.kdeplot(all_actual_combined, label=f"Overall Actual Score (n={len(all_actual_combined)})", color='#1f77b4', fill=True, alpha=0.3, ax=ax, cut=0)
        ax.set_xlabel("Value", fontsize=12); ax.set_ylabel("Density (Actual Score)", color='#1f77b4', fontsize=12)
        ax.tick_params(axis='y', labelcolor='#1f77b4'); ax.legend(loc='upper left', fontsize=10); ax.grid(True, linestyle=':', alpha=0.6)
        ax_err_dist = ax.twinx()
        if len(all_errors_combined) > 0: sns.kdeplot(all_errors_combined, label=f"Overall Model Error (n={len(all_errors_combined)})", color='#ff7f0e', fill=True, alpha=0.3, ax=ax_err_dist, cut=0)
        ax_err_dist.set_ylabel("Density (Model Error)", color='#ff7f0e', fontsize=12)
        ax_err_dist.tick_params(axis='y', labelcolor='#ff7f0e'); ax_err_dist.legend(loc='upper right', fontsize=10); ax_err_dist.grid(False)
        ax.set_title("Overall Distributions (Score vs. Error)", fontsize=14); ax.set_xlim(left=0)

        # --- Plot 4: Overall Scatter Plot (Score vs Error) ---
        ax = axes_overall[3]
        if len(all_actual_combined) > 0 and len(all_errors_combined) > 0:
             ax.scatter(all_actual_combined, all_errors_combined, alpha=0.2, s=10, color="#9467bd")
             corr_all = 'N/A'
             if np.std(all_actual_combined) > 1e-6 and np.std(all_errors_combined) > 1e-6:
                  try: corr_val_all, p_val_all = stats.pearsonr(all_actual_combined, all_errors_combined); corr_all = f"{corr_val_all:.4f}" if np.isfinite(corr_val_all) else 'N/A'
                  except ValueError: pass
             ax.text(0.05, 0.95, f"Overall Corr: {corr_all}", transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
             ax.set_title("Overall Correlation (Score vs. Error)", fontsize=14); ax.set_xlabel("Actual Score (Displacement)", fontsize=12); ax.set_ylabel("Model Reconstruction Error", fontsize=12)
             ax.grid(alpha=0.5, linestyle=':'); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center'); ax.set_title("Overall Correlation (Score vs. Error)", fontsize=14)

        # Save
        output_path = output_dir / "overall_score_vs_error_comparison.png"
        fig_overall.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_overall.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close(fig_overall)
        logger.info(f"Overall AE comparison visualization saved to: {output_path}")
        return output_path

    except ImportError as e: logger.error(f"Import error for overall AE viz: {e}."); plt.close('all'); return None
    except Exception as e: logger.error(f"Error creating overall AE viz: {e}", exc_info=True); plt.close('all'); return None