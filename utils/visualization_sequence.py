import numpy as np
import matplotlib
# Ensure Agg backend is used BEFORE importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import logging
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict
import traceback # Added for better error logging
from tqdm import tqdm
# Configure logging
# Use getLogger to avoid duplicate handlers if basicConfig was called elsewhere
logger = logging.getLogger(__name__)

def visualize_video_sequence_distribution(config, job_type_filter=None, cache_dir=None):
    """
    Create visualizations showing how sequences (reconstruction errors) are distributed
    within and across training videos. Reads job_type directly from cache files.

    Args:
        config: Configuration object (needs output_dir, cache_dir, job_categories).
        job_type_filter (str, optional): Specific job category to filter by (if None, show all).
                                         Defaults to None.
        cache_dir (str or Path, optional): Directory containing cached sequence data.
                                           Defaults to config.cache_dir.

    Returns:
        list: List of paths to the generated visualization files, or empty list if failed.
    """
    # --- Setup: Cache directory, output directory ---
    cache_dir_path = Path(cache_dir) if cache_dir else Path(config.cache_dir)
    if not cache_dir_path.exists() or not cache_dir_path.is_dir():
        logger.error(f"Cache directory not found or not a directory: {cache_dir_path}")
        return []

    output_dir = Path(config.output_dir) / 'visualization_training_data' # Specific subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving sequence distribution visualizations to: {output_dir.resolve()}")

    cache_files = list(cache_dir_path.glob("*_cache.npz"))
    if not cache_files:
        logger.error(f"No cache files (*_cache.npz) found in {cache_dir_path}")
        return []

    # --- Data Aggregation from Cache ---
    video_data = {}
    logger.info(f"Analyzing {len(cache_files)} cache files for sequence visualization...")
    for cache_file in tqdm(cache_files, desc="Analyzing Cache Files"):
        video_name = cache_file.stem.replace("_cache", "")
        try:
            cache_data = np.load(str(cache_file), allow_pickle=True)

            # --- Read job_type from cache ---
            if 'job_type' not in cache_data:
                logger.warning(f"Skipping cache {cache_file.name}: 'job_type' key missing.")
                continue
            video_job_type = str(cache_data['job_type']) # Convert to string just in case

            # --- Validate sequences and labels ---
            if 'sequences' not in cache_data or 'labels' not in cache_data:
                logger.debug(f"Skipping cache {cache_file.name}: Missing 'sequences' or 'labels'.")
                continue
            sequences = cache_data['sequences']
            numeric_labels = cache_data['labels']

            # Basic structural validation
            if not isinstance(sequences, np.ndarray) or not isinstance(numeric_labels, np.ndarray):
                 logger.debug(f"Skipping cache {cache_file.name}: Invalid data types for sequences/labels.")
                 continue
            if sequences.size == 0 or numeric_labels.size == 0: # Check if arrays are empty
                 logger.debug(f"Skipping cache {cache_file.name}: Empty sequences or labels array.")
                 continue
            # Check if sequences is an array of arrays (object dtype) or a single 3D array
            is_object_array = sequences.dtype == object
            if is_object_array and sequences.shape[0] != numeric_labels.shape[0]:
                 logger.debug(f"Skipping cache {cache_file.name}: Mismatch shapes (object array) {sequences.shape[0]} vs {numeric_labels.shape[0]}.")
                 continue
            elif not is_object_array and sequences.shape[0] != numeric_labels.shape[0]: # Assuming 3D array N x SeqLen x Features
                 logger.debug(f"Skipping cache {cache_file.name}: Mismatch shapes (numeric array) {sequences.shape} vs {numeric_labels.shape}.")
                 continue

            # Apply job type filter if specified
            if job_type_filter is not None and video_job_type != job_type_filter:
                continue

            # --- Calculate Stats per Sequence (REPLACED with Reconstruction Error) ---
            sequence_stats = []
            for i, seq in enumerate(sequences):
                 # --- START VALIDATION ---
                 # Check if seq is a valid NumPy array with numeric data
                 if not isinstance(seq, np.ndarray) or not np.issubdtype(seq.dtype, np.number):
                     logger.warning(f"Skipping sequence {i} in {video_name}: Expected numeric NumPy array, got {type(seq)} with dtype {getattr(seq, 'dtype', 'N/A')}")
                     continue # Skip to the next sequence
                 # Ensure it's 2D and not empty (SeqLen x Features)
                 if seq.ndim != 2 or seq.shape[0] == 0 or seq.shape[1] == 0:
                     logger.warning(f"Skipping sequence {i} in {video_name}: Invalid shape {seq.shape}")
                     continue # Skip to the next sequence
                 # --- END VALIDATION ---

                 try:
                    # Validate label (reconstruction error) for this sequence
                    if i >= len(numeric_labels) or not np.isfinite(numeric_labels[i]):
                         logger.debug(f"Skipping sequence {i} due to invalid reconstruction error in {video_name}")
                         continue

                    # Clean sequence data
                    seq = np.where(np.isinf(seq), np.nan, seq) # Replace inf
                    seq_clean = np.nan_to_num(seq, nan=0.0) # Replace NaN with 0

                    # Store the reconstruction error directly (it's already calculated)
                    reconstruction_error = float(numeric_labels[i]) # Renamed variable
                    reconstruction_error = min(100.0, max(0.0, reconstruction_error)) # Clamp if needed

                    # Calculate statistics relevant to reconstruction error
                    seq_mean_abs = np.mean(np.abs(seq_clean))
                    seq_max_abs = np.max(np.abs(seq_clean)) if seq_clean.size > 0 else 0
                    seq_variance = np.var(seq_clean) if seq_clean.size > 0 else 0
                    seq_nonzero_ratio = np.count_nonzero(seq_clean) / seq_clean.size if seq_clean.size > 0 else 0

                    # Store if stats are valid
                    if np.isfinite([seq_mean_abs, seq_max_abs, seq_variance, seq_nonzero_ratio]).all():
                        sequence_stats.append({
                            'mean_abs_disp': float(seq_mean_abs),
                            'max_abs_disp': float(seq_max_abs),
                            'variance': float(seq_variance),
                            'nonzero_ratio': float(seq_nonzero_ratio),
                            'reconstruction_error': reconstruction_error # Renamed key
                        })
                    else:
                         logger.debug(f"Skipping sequence {i} due to non-finite stats in {video_name}")

                 except Exception as stat_calc_err:
                    # Log error for specific sequence but continue processing others
                    logger.warning(f"Error calculating stats for a sequence in {video_name} (Index: {i}): {stat_calc_err}")

            # --- Store aggregated data for the video ---
            if sequence_stats:
                video_data[video_name] = {
                    'stats': sequence_stats,
                    'job_type': video_job_type,
                    'count': len(sequence_stats)
                }
                logger.debug(f"Successfully processed {len(sequence_stats)} sequences from {cache_file.name}")
            else:
                # Log if no valid sequences were extracted from this file
                logger.debug(f"No valid sequences/stats extracted from {cache_file.name}")

        except Exception as file_proc_err:
             # Log error for specific file but continue processing others
             logger.warning(f"Error processing cache file {cache_file.name}: {file_proc_err}")
             logger.debug(traceback.format_exc()) # Add traceback for debugging file errors


    # --- Check if ANY data was aggregated ---
    if not video_data:
        logger.error("No valid video data aggregated from cache for visualization.")
        return [] # Return empty list
    logger.info(f"Aggregated valid data from {len(video_data)} videos for visualization.")


    # --- Prepare DataFrame for Plotting ---
    generated_paths = []
    viz_suffix = f"_{job_type_filter}" if job_type_filter else "_all"

    all_sequences_stats = []
    for video_name, data in video_data.items():
        for stat in data['stats']:
            # Ensure essential keys exist and error is finite before adding
            if 'reconstruction_error' in stat and np.isfinite(stat['reconstruction_error']): # Renamed key
                all_sequences_stats.append({
                    'video': video_name, 'job_type': data['job_type'],
                    'reconstruction_error': stat['reconstruction_error'], # Renamed key
                    'variance': stat.get('variance', np.nan), # Handle missing keys gracefully
                    'max_abs_disp': stat.get('max_abs_disp', np.nan)
                })

    if not all_sequences_stats:
        logger.error("No valid sequence statistics collected for DataFrame creation.")
        return []

    df = pd.DataFrame(all_sequences_stats)
    # Convert error to numeric, coercing errors (like non-numeric strings) to NaN
    df['reconstruction_error'] = pd.to_numeric(df['reconstruction_error'], errors='coerce') # Renamed column
    df = df.dropna(subset=['reconstruction_error']) # Drop rows where error became NaN

    if df.empty:
         logger.error("DataFrame is empty after cleaning. Cannot generate plots.")
         return []

    # Define colors for job types
    job_colors = {job: f'C{i}' for i, job in enumerate(config.job_categories)}
    job_colors['unknown'] = 'grey' # Color for any unknown job types found

    # --- Plot 1: Per-Video Distributions (REPLACED with Reconstruction Error) ---
    try:
        videos_to_plot = sorted(df['video'].unique()) # Sort video names
        n_videos = len(videos_to_plot)
        ncols = min(4, n_videos)
        nrows = max(1, (n_videos + ncols - 1) // ncols)
        fig_height = max(6, nrows * 3); fig_width = ncols * 4

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()

        for i, video_name in enumerate(videos_to_plot):
            if i >= len(axes): break
            ax = axes[i]
            video_df = df[df['video'] == video_name]
            if not video_df.empty:
                 job_type = video_df['job_type'].iloc[0]
                 color = job_colors.get(job_type, 'grey')
                 sns.histplot(video_df, x='reconstruction_error', color=color, kde=True, ax=ax, bins=20, stat="count") # Renamed column
                 mean_error = video_df['reconstruction_error'].mean() # Renamed variable
                 ax.axvline(mean_error, color='red', linestyle='--', linewidth=1)
                 ax.set_title(f"{video_name}\n(Job: {job_type}, Mean Error: {mean_error:.1f})", fontsize=9) # Changed title
                 ax.set_xlabel('')
                 ax.set_ylabel('')
                 ax.grid(True, linestyle=':', alpha=0.4) # Add grid
            else:
                 ax.text(0.5, 0.5, "No data", ha='center', va='center'); ax.set_title(f"{video_name}\n(No Data)", fontsize=9)
                 ax.set_xlabel(''); ax.set_ylabel('')

        for j in range(i + 1, len(axes)): axes[j].axis('off') # Hide unused

        fig.suptitle(f'Reconstruction Error Distribution per Video ({n_videos} Videos)', fontsize=14) # Changed title
        fig.supxlabel('Reconstruction Error', fontsize=11); fig.supylabel('Sequence Count', fontsize=11) # Changed labels
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        video_plot_path = output_dir / f"video_reconstruction_error_distributions{viz_suffix}.png" # Changed filename
        plt.savefig(video_plot_path, dpi=120); plt.close(fig)
        generated_paths.append(str(video_plot_path))
        logger.info(f"Per-video reconstruction error distribution plot generated: {video_plot_path}") # Changed log message

    except Exception as e:
        logger.error(f"Error creating per-video reconstruction error distribution plot: {e}", exc_info=True) # Changed error message
        plt.close('all')

    # --- Plot 2: Comparison Across Videos/Jobs (REPLACED with Reconstruction Error) ---
    try:
        if df.empty:
             logger.warning("Skipping summary comparison plot: DataFrame is empty.")
        else:
            plt.figure(figsize=(16, 6))
            plt.subplot(1, 2, 1)
            sns.boxplot(data=df, x='job_type', y='reconstruction_error', order=sorted(df['job_type'].unique()), palette=job_colors, showfliers=False) # Renamed column
            plt.title('Reconstruction Error Distribution by Job Type', fontsize=12); plt.xlabel('Job Type', fontsize=10); plt.ylabel('Reconstruction Error', fontsize=10) # Changed title and labels
            plt.xticks(rotation=15, ha='right')

            plt.subplot(1, 2, 2)
            avg_errors = df.groupby('job_type')['reconstruction_error'].mean().reset_index() # Renamed variable
            sns.barplot(data=avg_errors, x='job_type', y='reconstruction_error', order=sorted(df['job_type'].unique()), palette=job_colors) # Renamed column
            plt.title('Average Reconstruction Error by Job Type', fontsize=12); plt.xlabel('Job Type', fontsize=10); plt.ylabel('Average Reconstruction Error', fontsize=10) # Changed title and labels
            plt.xticks(rotation=15, ha='right'); plt.ylim(bottom=0)

            plt.suptitle(f'Sequence Summary Comparison ({len(df)} sequences)', fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            comparison_path = output_dir / f"sequence_reconstruction_error_comparison{viz_suffix}.png" # Changed filename
            plt.savefig(comparison_path, dpi=120); plt.close()
            generated_paths.append(str(comparison_path))
            logger.info(f"Summary comparison plot generated: {comparison_path}")

    except Exception as e:
        logger.error(f"Error creating summary comparison plot: {e}", exc_info=True)
        plt.close('all')

    # --- Plot 3: Overlaid Distribution HISTOGRAMS by Job Type (Dynamic Limit) (REPLACED) ---
    try:
        if df.empty:
             logger.warning("Skipping distribution histogram generation: DataFrame is empty.")
        else:
            plt.figure(figsize=(14, 8))

            # --- Dynamic x-axis limit calculation ---
            limit_percentile = 95
            finite_errors = df.loc[np.isfinite(df['reconstruction_error']), 'reconstruction_error'] # Renamed column
            x_limit = np.percentile(finite_errors, limit_percentile) if not finite_errors.empty else 100 # Renamed variable
            x_limit = max(x_limit, 10) # Ensure minimum width
            # --- End dynamic limit calculation ---

            # Create the histogram using seaborn's histplot with 'hue'
            sns.histplot(data=df, x='reconstruction_error', hue='job_type', # Renamed column
                         palette=job_colors,
                         alpha=0.6,
                         kde=False, # Keep as histogram bars
                         stat="count",
                         bins=50,
                         element="step", # Use steps for better overlap visibility
                         common_norm=False
                        )

            # Apply the dynamically calculated x-axis limit
            plt.xlim(0, x_limit)

            plt.title(f'Reconstruction Error Distribution by Job Type (Counts, up to {x_limit:.0f})', fontsize=14) # Changed title
            plt.xlabel('Reconstruction Error', fontsize=12) # Changed label
            plt.ylabel('Sequence Count', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)

            # Legend
            handles, labels = plt.gca().get_legend_handles_labels()
            unique_labels_map = {label: handle for handle, label in zip(handles, labels)}
            plt.legend(handles=unique_labels_map.values(),
                       labels=[l.capitalize() for l in unique_labels_map.keys()],
                       title='Job Type')

            plt.tight_layout()
            dist_hist_path = output_dir / f"sequence_reconstruction_error_distribution_overlay{viz_suffix}.png" # Changed filename
            plt.savefig(dist_hist_path, dpi=120)
            plt.close() # Close the figure
            generated_paths.append(str(dist_hist_path))
            logger.info(f"Overlaid reconstruction error histogram visualization generated: {dist_hist_path}") # Changed log message

    except Exception as e:
        logger.error(f"Error creating overlaid reconstruction error distribution histograms: {e}") # Changed error message
        logger.debug(traceback.format_exc())
        plt.close('all')

    # --- Return paths to generated plots ---
    return generated_paths