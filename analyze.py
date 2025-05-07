#!/usr/bin/env python3
"""
Movement Analysis Tool - Improved Batch Processing Script

This script analyzes videos for movement patterns and anomalies using the LSTM
Autoencoder model. It can process single videos or entire directories of videos
with improved multi-threading and resource management.

Example usage:
    # Process a single video
    python analyze.py --input path/to/video.mp4 --job_type painting
    
    # Process a directory of videos with 4 worker threads
    python analyze.py --input path/to/videos/ --job_type masonry --workers 4
    
    # Process a directory with custom output and visualization options
    python analyze.py --input path/to/videos/ --output_dir results/ --fps 30 --show_results
"""

import argparse
import logging
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to use Agg backend
from pathlib import Path
import sys
import os
import subprocess
import platform
import traceback
import numpy as np
import torch
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global list to keep track of all generated files
generated_files = []

def parse_args():
    """Parses command-line arguments with improved options for batch processing."""
    parser = argparse.ArgumentParser(description='Analyze movement anomalies in videos')
    parser.add_argument('--input', type=str, default='data/test_videos/construction_site.mp4',
                      help='Input video file or directory of videos')
    parser.add_argument('--output_dir', type=str, default='output/analyzed',
                      help='Directory for analyzed videos and plots')
    parser.add_argument('--model_path', type=str, default='models/saved/lstm_autoencoder.pth',
                      help='Path to trained LSTM Autoencoder model')
    parser.add_argument('--job_type', type=str, choices=['masonry', 'painting', 'plastering'], default='painting',
                      help='Job type for analysis context')
    parser.add_argument('--sample_rate', type=float, default=8.0,
                      help='Analysis sample rate in Hz (default: 8.0)')
    parser.add_argument('--fps', type=float, default=None,
                      help='Override video FPS for analysis and output')
    parser.add_argument('--skip_frames', type=int, default=0,
                      help='Number of frames to skip between analysis (overrides sample_rate if > 0)')
    parser.add_argument('--show_results', action='store_true',
                      help='Open generated output video and visualizations after processing')
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker threads for batch processing (default: auto)')
    parser.add_argument('--consistent_y_scale', action='store_true', default=True,
                      help='Use consistent Y-axis scale for graphs in video and standalone plots')
    parser.add_argument('--report_format', choices=['txt', 'json', 'csv'], default='csv',
                      help='Format for analysis report files (default: csv)')
    parser.add_argument('--timeout', type=int, default=3600,
                      help='Maximum time in seconds to process a single video in batch mode (default: 3600)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging for detailed processing information')
    return parser.parse_args()

def open_file(file_path):
    """Opens a file using the default system application."""
    abs_file_path = Path(file_path).resolve()
    logger.info(f"Attempting to open: {abs_file_path}")
    try:
        if not abs_file_path.exists():
            logger.error(f"File not found, cannot open: {abs_file_path}")
            return False
        if platform.system() == 'Windows':
            os.startfile(str(abs_file_path))
        elif platform.system() == 'Darwin':
            subprocess.run(['open', str(abs_file_path)], check=True)
        else:
            subprocess.run(['xdg-open', str(abs_file_path)], check=True)
        logger.info(f"Successfully requested to open: {abs_file_path}")
        return True
    except Exception as e:
        logger.error(f"Could not open file {abs_file_path}: {e}")
        return False
    
def register_output_file(file_path, category=None):
    """Registers an output file path if it exists, with optional category."""
    global generated_files
    if file_path:
        path_obj = Path(file_path).resolve()
        if path_obj.exists() and path_obj.is_file():
            entry = {
                'path': str(path_obj),
                'category': category if category else 'other',
                'name': path_obj.name,
                'size': path_obj.stat().st_size
            }
            
            # Check if already registered
            if not any(item['path'] == entry['path'] for item in generated_files):
                generated_files.append(entry)
                logger.info(f"✓ Registered output file: {entry['path']} (Category: {entry['category']})")
        else:
            logger.warning(f"✗ Output file not found or not a file, cannot register: {file_path}")

def ensure_directory(dir_path):
    """Ensure directory exists, creating it if necessary."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def setup_analyzer(config, args):
    """Configure and set up the analyzer with consistent settings."""
    try:
        # Import modules here to ensure project structure is properly loaded
        from processors.video_analyzer import VideoAnalyzer
        
        # Apply consistent y-scale if requested
        if args.consistent_y_scale:
            logger.info("Using consistent Y-axis scale for all visualizations")
            
        # Initialize analyzer
        analyzer = VideoAnalyzer(config)
        return analyzer
        
    except Exception as e:
        logger.error(f"Error setting up analyzer: {e}")
        logger.error(traceback.format_exc())
        raise

def process_batch_with_analyzer(config, analyzer, args):
    """Process a batch of videos using the AnalysisBatchProcessor."""
    try:
        # Import the batch processor
        from processors.batch_processor import AnalysisBatchProcessor
        
        input_dir = Path(args.input).resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        logger.info(f"Starting batch processing of videos in: {input_dir}")
        
        # Create batch processor with the analyzer
        batch_processor = AnalysisBatchProcessor(config, analyzer)
        
        # Set number of workers for batch processing
        if args.workers is not None:
            logger.info(f"Using specified {args.workers} worker threads for batch processing")
            config.analysis_workers = args.workers
        
        # Process videos in batches
        batch_results = batch_processor.batch_analyze_videos(
            directory_path=input_dir,
            job_type=args.job_type,
            skip_frames=args.skip_frames,
            fps_override=args.fps,
            timeout=args.timeout
        )
        
        # Generate batch summary
        if batch_results:
            summary_path = batch_processor.create_batch_summary(batch_results, job_type=args.job_type)
            if summary_path:
                register_output_file(summary_path, category='batch_summary')
                
        # Generate consolidated report
        create_consolidated_report(batch_results, output_dir=config.output_dir, report_format=args.report_format)
        
        return batch_results
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        logger.error(traceback.format_exc())
        return []

def create_consolidated_report(batch_results, output_dir, report_format='csv'):
    """Create a consolidated report of all processed videos."""
    try:
        output_dir = ensure_directory(output_dir)
        report_path = output_dir / f"consolidated_report.{report_format}"
        
        if report_format == 'csv':
            import csv
            with open(report_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['video', 'status', 'processing_time', 'output_path', 'error'])
                # Write data
                for result in batch_results:
                    writer.writerow([
                        os.path.basename(result.get('input', '')),
                        result.get('status', 'unknown'),
                        f"{result.get('processing_time', 0):.2f}",
                        os.path.basename(result.get('output', 'N/A')),
                        result.get('error', 'N/A')
                    ])
        
        elif report_format == 'json':
            # Create simplified report structure
            report_data = []
            for result in batch_results:
                report_data.append({
                    'video': os.path.basename(result.get('input', '')), 
                    'status': result.get('status', 'unknown'),
                    'processing_time': round(result.get('processing_time', 0), 2),
                    'output_path': os.path.basename(result.get('output', 'N/A')),
                    'error': result.get('error', 'N/A')
                })
                
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=4)
        
        else:  # txt format
            with open(report_path, 'w') as f:
                f.write(f"Consolidated Analysis Report\n")
                f.write(f"===========================\n\n")
                f.write(f"Total videos processed: {len(batch_results)}\n")
                success_count = sum(1 for r in batch_results if r.get('status') == 'success')
                f.write(f"Successful: {success_count}, Failed: {len(batch_results) - success_count}\n\n")
                
                for i, result in enumerate(batch_results, 1):
                    f.write(f"Video {i}: {os.path.basename(result.get('input', ''))}\n")
                    f.write(f"  Status: {result.get('status', 'unknown')}\n")
                    f.write(f"  Processing time: {result.get('processing_time', 0):.2f} seconds\n")
                    if result.get('status') == 'success':
                        f.write(f"  Output: {os.path.basename(result.get('output', 'N/A'))}\n")
                    else:
                        f.write(f"  Error: {result.get('error', 'N/A')}\n")
                    f.write("\n")
        
        register_output_file(report_path, category='report')
        logger.info(f"Consolidated report saved to: {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Error creating consolidated report: {e}")
        return None

def main():
    """Main function to run the movement analysis on videos."""
    global generated_files
    generated_files = []  # Reset global list for each run

    # Display script banner
    print("\n" + "=" * 80 + f"\n{'MOVEMENT ANOMALY ANALYSIS SCRIPT':^80}\n" + "=" * 80 + "\n")

    # Ensure project root is in path
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir  # Assuming analyze.py is in the project root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"Added project root to sys.path: {project_root}")

    try:
        # --- Parse Arguments ---
        args = parse_args()
        
        # Set logging level based on verbose flag
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Verbose logging enabled")
        
        # --- Configuration ---
        logger.info("--- STEP 1: Initializing Configuration ---")
        from config import Config
        config = Config()

        # Override config with command-line arguments
        config.input_path_arg = Path(args.input).resolve()
        config.output_dir = Path(args.output_dir).resolve()
        config.model_save_path = Path(args.model_path).resolve()
        
        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log system information
        log_system_info()
        
        # Log configuration settings
        logger.info(f"Input Path: {config.input_path_arg}")
        logger.info(f"Output Dir: {config.output_dir}")
        logger.info(f"LSTM Model Path: {config.model_save_path}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Sample Rate: {config.sample_rate} Hz")
        logger.info(f"Job Type: {args.job_type}")
        if args.skip_frames > 0: 
            logger.info(f"Manual Frame Skip: {args.skip_frames}")
        if args.fps: 
            logger.info(f"FPS Override: {args.fps}")
        if args.workers:
            logger.info(f"Worker Threads: {args.workers}")

        # --- Verify inputs ---
        if not config.input_path_arg.exists():
            logger.error(f"Input path does not exist: {config.input_path_arg}")
            return

        # --- Initialize Analyzer ---
        logger.info("--- STEP 2: Initializing Video Analyzer ---")
        try:
            analyzer = setup_analyzer(config, args)
            logger.info("VideoAnalyzer initialized successfully.")
        except Exception as init_err:
            logger.error(f"Failed to initialize VideoAnalyzer: {init_err}", exc_info=True)
            return

        # --- Perform Analysis ---
        logger.info("--- STEP 3: Starting Analysis ---")
        
        start_time = time.time()
        
        if config.input_path_arg.is_file():
            # Single video mode
            logger.info(f"Analyzing single video: {config.input_path_arg.name}")
            
            # Create a video-specific output directory for consistent organization
            video_output_dir = ensure_directory(config.output_dir / config.input_path_arg.stem)
            output_path = video_output_dir / f"{config.input_path_arg.stem}_analyzed.mp4"
            
            # Process the video
            try:
                result = analyzer.analyze_video(
                    video_path=config.input_path_arg,
                    job_type=args.job_type,
                    output_path=output_path,
                    skip_frames=args.skip_frames,
                    fps_override=args.fps
                )
                
                if result:
                    register_output_file(result, category='video')
                    logger.info(f"Analysis completed successfully. Output: {result}")
                    
                    # Generate additional plots if available
                    if hasattr(analyzer, 'last_movement_data') and analyzer.last_movement_data:
                        # Create plot directory
                        plot_dir = ensure_directory(video_output_dir / 'plots')
                        
                        # Save data in requested format
                        save_analysis_data(
                            analyzer.last_movement_data, 
                            video_output_dir / 'data' / f"{config.input_path_arg.stem}_analysis.{args.report_format}",
                            args.report_format
                        )
                    
                    # Open results if requested
                    if args.show_results:
                        open_file(result)
                        
                else:
                    logger.error(f"Analysis failed for {config.input_path_arg.name}")
            
            except Exception as e:
                logger.error(f"Error analyzing video: {config.input_path_arg.name}")
                logger.error(traceback.format_exc())
                
        elif config.input_path_arg.is_dir():
            # Batch mode
            logger.info(f"Analyzing directory (Batch Mode): {config.input_path_arg}")
            
            # Process all videos in the directory
            batch_results = process_batch_with_analyzer(config, analyzer, args)
            
            # Log results
            success_count = sum(1 for r in batch_results if r.get('status') == 'success')
            logger.info(f"Batch processing complete. Processed: {len(batch_results)}, Succeeded: {success_count}, Failed: {len(batch_results) - success_count}")
            
            # Open results if requested
            if args.show_results and generated_files:
                # Open main output directory
                open_file(config.output_dir)
                
                # Open up to 3 key files
                for category in ['batch_summary', 'video', 'report']:
                    category_files = [f for f in generated_files if f['category'] == category]
                    if category_files:
                        open_file(category_files[0]['path'])
        
        else:
            logger.error(f"Input path is neither a file nor a directory: {config.input_path_arg}")
        
        # --- Calculate total time ---
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")

        # --- Final Summary ---
        print("\n" + "=" * 80 + f"\n{'ANALYSIS COMPLETE':^80}\n" + "=" * 80)
        
        # Display summary of generated files
        if generated_files:
            print(f"\nGenerated {len(generated_files)} output files:")
            
            # Group files by category
            categorized_files = {}
            for file_info in generated_files:
                category = file_info['category']
                if category not in categorized_files:
                    categorized_files[category] = []
                categorized_files[category].append(file_info)
            
            # Display files by category
            category_order = ['video', 'plot', 'data', 'summary', 'batch_summary', 'report', 'other']
            for category in category_order:
                if category in categorized_files:
                    print(f"\n{category.upper()} FILES:")
                    for i, file_info in enumerate(categorized_files[category], 1):
                        file_size_mb = file_info['size'] / (1024 * 1024)
                        print(f"  {i}. {file_info['name']} ({file_size_mb:.1f} MB)")
            
            print(f"\nAll files saved to: {config.output_dir}")
        else:
            print("\nNo output files were generated.")

    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user.")
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        print(f"\nERROR: {e}")
    finally:
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
        logger.info("--- Script finished ---")

def log_system_info():
    """Log system information including GPU, CPU, memory, etc."""
    logger.info("--- SYSTEM INFO ---")
    logger.info(f"Python: {sys.version.split()[0]}")
    
    # CPU info
    import platform
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"CPU count: {os.cpu_count()}")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Total RAM: {memory.total / (1024**3):.2f} GB")
        logger.info(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    except ImportError:
        logger.info("psutil not available, skipping memory info")
    
    # GPU info
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Memory: {props.total_memory / (1024**3):.2f} GB")
    
    logger.info("-" * 30)

def save_analysis_data(movement_data, output_path, format_type='csv'):
    """Save analysis data to a file in the specified format."""
    try:
        # Ensure parent directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'json':
            # Convert analysis data to serializable format
            serializable_data = {}
            for person_id, data in movement_data.items():
                serializable_data[str(person_id)] = {
                    'times': data.get('times', []),
                    'reconstruction_errors': data.get('reconstruction_errors', []),
                    'anomaly_levels': data.get('anomaly_levels', []),
                    'direct_scores': data.get('direct_scores', [])
                }
            with open(output_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        
        elif format_type == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['person_id', 'time', 'reconstruction_error', 'anomaly_level', 'direct_score'])
                # Write data for each person
                for person_id, data in movement_data.items():
                    times = data.get('times', [])
                    errors = data.get('reconstruction_errors', [])
                    levels = data.get('anomaly_levels', [])
                    direct_scores = data.get('direct_scores', [])
                    
                    # Ensure all arrays have the same length by using the minimum length
                    min_length = min(len(times), len(errors), len(levels), len(direct_scores) if direct_scores else len(times))
                    
                    for i in range(min_length):
                        direct_score = direct_scores[i] if i < len(direct_scores) else 0.0
                        writer.writerow([person_id, times[i], errors[i], levels[i], direct_score])
        
        else:  # txt format
            with open(output_path, 'w') as f:
                f.write(f"Analysis Results\n")
                f.write(f"================\n\n")
                
                for person_id, data in movement_data.items():
                    f.write(f"Person ID: {person_id}\n")
                    times = data.get('times', [])
                    errors = data.get('reconstruction_errors', [])
                    levels = data.get('anomaly_levels', [])
                    direct_scores = data.get('direct_scores', [])
                    
                    if times:
                        f.write(f"Analysis Points: {len(times)}\n")
                        
                        # Calculate statistics
                        if errors:
                            f.write(f"Mean Error: {np.mean(errors):.4f}\n")
                            f.write(f"Max Error: {np.max(errors):.4f}\n")
                            
                            # Add direct score statistics if available
                            if direct_scores:
                                valid_scores = [s for s in direct_scores if np.isfinite(s)]
                                if valid_scores:
                                    f.write(f"Mean Direct Score: {np.mean(valid_scores):.4f}\n")
                                    f.write(f"Max Direct Score: {np.max(valid_scores):.4f}\n")
                            
                            # Count anomaly levels
                            level_counts = {}
                            for level in levels:
                                level_counts[level] = level_counts.get(level, 0) + 1
                            
                            f.write("Anomaly Distributions:\n")
                            for level, count in level_counts.items():
                                percentage = count / len(levels) * 100
                                f.write(f"  {level}: {count} ({percentage:.1f}%)\n")
                    f.write("\n")
        
        register_output_file(output_path, category='data')
        logger.info(f"Analysis data saved to: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error saving analysis data: {e}")
        return None

if __name__ == "__main__":
    main()