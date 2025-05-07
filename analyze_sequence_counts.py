
from typing import Optional, List, Dict, Tuple, Any, Union
# Filename: analyze_sequence_counts.py

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import traceback
import warnings
from collections import defaultdict

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Define Job Colors Consistently ---
JOB_COLOR_MAP = {
    'masonry': 'steelblue',
    'painting': 'mediumseagreen',
    'plastering': 'indianred',
    'Unknown': 'grey',            # Fallback color for unknown
    'Error': 'black'
}
KNOWN_JOB_TYPES = sorted([k for k in JOB_COLOR_MAP if k not in ['Unknown', 'Error']])

# --- Argument Parsing ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Analyze sequence counts and movement scores per video from cache files.')
    parser.add_argument('--cache_dir', type=str, default='cache',
                        help='Directory containing the cache files (*_cache.npz).')
    parser.add_argument('--output_dir', type=str, default='output/analysis_reports',
                        help='Directory to save the reports and visualizations.')
    return parser.parse_args()

# --- Core Logic ---
def analyze_cache_files(cache_dir: Path) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """
    Analyzes cache files to count sequences, calculate average movement score per video,
    and collect all individual sequence scores per job type.
    """
    cache_dir = Path(cache_dir)
    empty_df = pd.DataFrame(columns=['video_name', 'job_type', 'sequence_count', 'average_movement'])
    empty_scores = defaultdict(list)

    if not cache_dir.is_dir():
        logger.error(f"Cache directory not found: {cache_dir}")
        return empty_df, empty_scores

    cache_files = sorted(list(cache_dir.glob("*_cache.npz")))
    if not cache_files:
        logger.warning(f"No cache files found in {cache_dir}")
        return empty_df, empty_scores

    logger.info(f"Found {len(cache_files)} cache files to analyze.")

    video_results = []
    all_job_scores = defaultdict(list)

    for cache_file in cache_files:
        video_name = cache_file.stem.replace("_cache", "")
        sequence_count = 0
        avg_movement = np.nan
        job_type = 'Unknown'
        finite_labels_video = []

        try:
            data = np.load(cache_file, allow_pickle=True)
            job_type = str(data['job_type']) if 'job_type' in data else 'Unknown'
            if job_type not in JOB_COLOR_MAP:
                logger.warning(f"Unknown job type '{job_type}' found in {cache_file.name}. Treating as 'Unknown'.")
                job_type = 'Unknown'

            if 'sequences' in data:
                sequences = data['sequences']
                if isinstance(sequences, np.ndarray):
                    if sequences.dtype == object: sequence_count = len(sequences)
                    elif sequences.ndim == 3: sequence_count = sequences.shape[0]
                    else: logger.warning(f"Unexpected sequence array structure in {cache_file.name}. Count unavailable.")
                else:
                    try: sequence_count = len(sequences)
                    except TypeError: pass
            else:
                logger.warning(f"Skipping sequence count for {cache_file.name}: 'sequences' key missing.")

            if 'labels' in data:
                labels = data['labels']
                if isinstance(labels, np.ndarray) and labels.size > 0:
                    finite_labels_video = labels[np.isfinite(labels)].tolist()
                    if finite_labels_video:
                        avg_movement = np.mean(finite_labels_video)
                        if job_type != 'Unknown' and job_type != 'Error':
                             all_job_scores[job_type].extend(finite_labels_video)
                    else:
                        logger.debug(f"No finite movement scores ('labels') found in {cache_file.name}.")
                else:
                    logger.debug(f"Movement scores ('labels') are empty or not a numpy array in {cache_file.name}.")
            else:
                 logger.debug(f"Skipping movement scores for {cache_file.name}: 'labels' key missing.")

            video_results.append({
                'video_name': video_name,
                'job_type': job_type,
                'sequence_count': sequence_count,
                'average_movement': avg_movement
            })
            logger.debug(f"Processed {cache_file.name}: {sequence_count} sequences, Avg Move: {avg_movement:.2f}, Added {len(finite_labels_video)} scores to job '{job_type}'")

        except Exception as e:
            logger.error(f"Failed to process cache file {cache_file.name}: {e}")
            logger.debug(traceback.format_exc())
            video_results.append({ 'video_name': video_name, 'job_type': 'Error', 'sequence_count': 0, 'average_movement': np.nan })

    return pd.DataFrame(video_results), dict(all_job_scores)

# --- Visualization ---
def create_sequence_count_plot(df: pd.DataFrame, output_path: Path):
    """ Creates a bar chart visualizing sequence counts per video, colored by job type. """
    if df.empty or 'sequence_count' not in df.columns: logger.warning("DataFrame is empty or missing 'sequence_count', cannot create count plot."); return
    df_plot = df.dropna(subset=['sequence_count']);
    if df_plot.empty: logger.warning("No valid sequence counts to plot after dropping NaNs."); return

    num_videos = len(df_plot['video_name'].unique()); fig_width = max(8, num_videos * 0.5); fig_height = 6
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(x='video_name', y='sequence_count', hue='job_type', data=df_plot, palette=JOB_COLOR_MAP, hue_order=KNOWN_JOB_TYPES + ['Unknown', 'Error'], dodge=False)
    for patch in ax.patches:
        try:
            height = patch.get_height()
            if np.isfinite(height) and height > 0: ax.text(patch.get_x() + patch.get_width() / 2., height, f'{height:.0f}', fontsize=8, ha='center', va='bottom', color='black', rotation=0)
        except Exception as label_err: logger.warning(f"Could not add a bar label: {label_err}")
    plt.xlabel("Video Name", fontsize=12); plt.ylabel("Number of Sequences Extracted", fontsize=12); plt.title("Number of Sequences Extracted per Video (Colored by Job Type)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(fontsize=10); plt.grid(axis='y', linestyle='--', alpha=0.7)
    handles, labels = ax.get_legend_handles_labels(); unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels: unique_labels[label] = handle
    if unique_labels: ax.legend(unique_labels.values(), unique_labels.keys(), title="Job Type", loc='upper right', fontsize=9)
    elif ax.get_legend() is not None: ax.get_legend().remove()
    plt.tight_layout()
    try: plt.savefig(output_path, dpi=150, bbox_inches='tight'); logger.info(f"Sequence count plot saved to: {output_path}")
    except Exception as e: logger.error(f"Failed to save plot: {e}")
    finally: plt.close()

def create_movement_distribution_plots(
    df_video_avg: pd.DataFrame,
    all_job_scores: Dict[str, list],
    output_dir: Path
    ):
    """ Creates plots for average movement score distributions. """
    generated_paths = []

    # --- Plot 1: Average Movement Score per Video (Bar Plot) ---
    logger.info("Generating plot: Average Movement Score per Video (Bar Plot)")
    df_plot_avg = df_video_avg.dropna(subset=['average_movement'])
    if not df_plot_avg.empty:
        try:
            num_videos = len(df_plot_avg['video_name'].unique())
            fig_width = max(8, num_videos * 0.4); fig_height = 7
            plt.figure(figsize=(fig_width, fig_height))
            ax = sns.barplot(x='video_name', y='average_movement', hue='job_type', data=df_plot_avg, palette=JOB_COLOR_MAP, hue_order=KNOWN_JOB_TYPES + ['Unknown', 'Error'], dodge=False)
            for patch in ax.patches:
                try:
                    height = patch.get_height()
                    if np.isfinite(height): ax.text(patch.get_x() + patch.get_width() / 2., height, f'{height:.1f}', fontsize=8, ha='center', va='bottom', color='black', rotation=0)
                except Exception as label_err: logger.warning(f"Could not add bar label to avg movement plot: {label_err}")
            plt.xlabel("Video Name", fontsize=12); plt.ylabel("Average Movement Score", fontsize=12); plt.title("Average Movement Score per Video (Colored by Job Type)", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(fontsize=10); plt.grid(axis='y', linestyle='--', alpha=0.7);
            handles, labels = ax.get_legend_handles_labels(); unique_labels = {}
            for handle, label in zip(handles, labels):
                if label not in unique_labels: unique_labels[label] = handle
            if unique_labels: ax.legend(unique_labels.values(), unique_labels.keys(), title="Job Type", loc='upper right', fontsize=9)
            elif ax.get_legend() is not None: ax.get_legend().remove()
            plt.tight_layout()
            plot_path = output_dir / "average_movement_per_video.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight'); generated_paths.append(str(plot_path)); logger.info(f"Average movement per video plot saved to: {plot_path}")
        except Exception as e: logger.error(f"Failed to create average movement per video plot: {e}"); logger.debug(traceback.format_exc())
        finally: plt.close()
    else: logger.warning("No valid data for average movement per video plot.")


    # --- Plot 2: Distribution of ALL Individual Sequence Scores per Job Type (Subplots) ---
    logger.info("Generating plot: Distribution of Individual Sequence Scores (Subplots per Job Type)")
    if not all_job_scores: logger.warning("No individual scores collected, cannot create per-job distribution subplots."); return generated_paths
    job_types_with_scores = sorted([jt for jt, scores in all_job_scores.items() if scores and jt in KNOWN_JOB_TYPES])
    if not job_types_with_scores: logger.warning("No known job types have scores for distribution plots."); return generated_paths
    num_jobs = len(job_types_with_scores)
    fig_height_per_plot = 4
    fig, axes = plt.subplots(num_jobs, 1, figsize=(8, num_jobs * fig_height_per_plot), sharex=True)
    if num_jobs == 1: axes = [axes]

    all_scores_combined = [score for jt in job_types_with_scores for score in all_job_scores[jt]]
    if not all_scores_combined: logger.warning("No scores found across relevant job types for distribution plots."); return generated_paths
    min_score = np.percentile(all_scores_combined, 1); max_score = np.percentile(all_scores_combined, 99)
    x_limit_padding = (max_score - min_score) * 0.05; x_lim = (max(0, min_score - x_limit_padding), max_score + x_limit_padding)

    for i, job_type in enumerate(job_types_with_scores):
        scores = all_job_scores[job_type]
        if not isinstance(scores, (list, np.ndarray)) or not scores: continue
        try: scores_numeric = np.array(scores, dtype=float); scores_numeric = scores_numeric[np.isfinite(scores_numeric)]
        except Exception as e: logger.warning(f"Could not process scores for job type '{job_type}': {e}. Skipping plot."); continue
        if scores_numeric.size == 0: logger.warning(f"No finite scores for job type '{job_type}' after filtering, skipping plot."); continue

        ax = axes[i]
        logger.info(f"Plotting distribution for job: {job_type} (Subplot {i+1})")

        # --- CORRECTED COLOR FALLBACK ---
        # Use the value associated with 'Unknown' as the fallback color
        plot_color = JOB_COLOR_MAP.get(job_type, JOB_COLOR_MAP['Unknown'])
        # --- END CORRECTED COLOR FALLBACK ---
        sns.histplot(scores_numeric, kde=True, color=plot_color, element='step', bins=30, ax=ax)

        mean_score = np.mean(scores_numeric); median_score = np.median(scores_numeric); std_score = np.std(scores_numeric)
        stats_text = f"Mean: {mean_score:.2f}\nMedian: {median_score:.2f}\nStd Dev: {std_score:.2f}\nN: {len(scores_numeric)}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        ax.set_title(f"{job_type.capitalize()}", fontsize=13); ax.set_ylabel("Number of Sequences", fontsize=11); ax.grid(axis='y', linestyle='--', alpha=0.7); ax.set_xlim(x_lim)
        if i == num_jobs - 1: ax.set_xlabel("Individual Sequence Movement Score", fontsize=11)
        else: ax.set_xlabel("")

    fig.suptitle("Distribution of Individual Sequence Scores per Job Type", fontsize=15, fontweight='bold')
    try: fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    except ValueError as layout_err: logger.warning(f"Could not apply tight_layout: {layout_err}. Plot might have overlapping elements.")

    try:
        plot_path = output_dir / "individual_score_per_job_subplots.png"
        plt.savefig(plot_path, dpi=150); generated_paths.append(str(plot_path)); logger.info(f"Individual score per job type subplots saved to: {plot_path}")
    except Exception as e: logger.error(f"Failed to save per-job subplots figure: {e}"); logger.debug(traceback.format_exc())
    finally: plt.close(fig)

    return generated_paths

# --- Main Execution ---
def main():
    """Main function to run the analysis."""
    args = parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting sequence analysis for cache directory: {args.cache_dir}")
    df_video_stats, all_job_scores = analyze_cache_files(Path(args.cache_dir))
    if not df_video_stats.empty:
        csv_path = output_dir / "video_analysis_report.csv"
        try: df_video_stats.to_csv(csv_path, index=False, float_format='%.4f'); logger.info(f"Per-video analysis report saved to: {csv_path}")
        except Exception as e: logger.error(f"Failed to save CSV report: {e}")
        plot_path_count = output_dir / "sequence_counts_visualization.png"
        create_sequence_count_plot(df_video_stats[['video_name', 'job_type', 'sequence_count']].copy(), plot_path_count) # Pass job_type
        create_movement_distribution_plots(df_video_avg=df_video_stats[['video_name', 'job_type', 'average_movement']].copy(), all_job_scores=all_job_scores, output_dir=output_dir)
    else: logger.info("No results generated.")
    logger.info("Analysis finished.")

if __name__ == "__main__":
    main()