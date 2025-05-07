import torch
from pathlib import Path
import argparse
import sys
import os
import logging
import json
import traceback
from typing import Optional
import subprocess
import platform

# Ensure project root is in path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import necessary modules
try:
    from config import Config
    from training.trainer import MovementTrainer
    from processors.data_processor import MovementDataProcessor
    from processors.video_analyzer import VideoAnalyzer
    # Import evaluator and model loader here for clarity
    from training.evaluator import AnomalyEvaluator
    from utils.model_loader import load_movement_classifier
except ImportError as e:
    print(f"ERROR: Failed to import modules. Check if the script is run from the project root and dependencies are installed. Details: {e}", file=sys.stderr)
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO, # Changed default level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Train and analyze movement with an LSTM Autoencoder')
    parser.add_argument('--video_dir', type=str, default='data/videos', help='Directory containing training videos')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory for output files')
    parser.add_argument('--model_path', type=str, default='models/saved/lstm_autoencoder.pth', help='Path to save/load the LSTM Autoencoder model')
    parser.add_argument('--batch_size', type=int, default=None, help='Training batch size (overrides config if set)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (overrides config if set)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config if set)')
    parser.add_argument('--analyze', action='store_true', help='Run analysis on videos after training (requires --input_path)')
    parser.add_argument('--input_path', type=str, default=None, help='Path to video file or directory to analyze (required if --analyze is set)')
    parser.add_argument('--job_type', type=str, default='masonry', help='Job type for analysis context (used if --analyze is set)')
    parser.add_argument('--show_results', action='store_true', help='Open generated output video and visualizations after processing')
    parser.add_argument('--skip_frames', type=int, default=0, help='Number of frames to skip during analysis')
    parser.add_argument('--fps', type=float, default=None, help='Override video FPS for analysis')

    # Add validation for analyze and input_path
    args = parser.parse_args()
    if args.analyze and not args.input_path:
        parser.error("--analyze requires --input_path to be set.")

    return args

def log_system_info() -> None:
    """Logs system information (device, Python version, PyTorch, etc.)."""
    logger.info("--- SYSTEM INFO ---")
    logger.info(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"  Python: {sys.version.split()[0]}")
    logger.info(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA: {torch.version.cuda}")
    logger.info("-" * 20)

def verify_directories(config: Config) -> None:
    """Verifies the existence of necessary directories."""
    logger.info("--- Verifying Directories ---")
    # Check training video directory
    if not config.video_dir.is_dir():
         raise NotADirectoryError(f"Training video directory not found or not a directory: {config.video_dir}")
    logger.info(f"  Verified Training Video Dir: {config.video_dir}")

    # Check and create output directory and model save directory
    for path_obj in [config.output_dir, config.model_save_path.parent]:
        if not path_obj.exists():
            logger.warning(f"  Directory does not exist: {path_obj}. Creating it.")
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"  Created directory: {path_obj}")
            except OSError as e:
                logger.error(f"  Failed to create directory {path_obj}: {e}")
                raise
        elif not path_obj.is_dir():
             raise NotADirectoryError(f"Path exists but is not a directory: {path_obj}")
        logger.info(f"  Verified Output/Model Dir Parent: {path_obj}")

    # Verify input path for analysis if provided
    if hasattr(config, 'input_path_arg') and config.input_path_arg:
        if not config.input_path_arg.exists():
             raise FileNotFoundError(f"Input path for analysis does not exist: {config.input_path_arg}")
        logger.info(f"  Verified Analysis Input Path: {config.input_path_arg}")

    logger.info("--- Directories verified ---")

def open_file(file_path):
    """Opens a file using the default system application."""
    abs_file_path = Path(file_path).resolve()
    logger.info(f"Attempting to open: {abs_file_path}")
    try:
        if not abs_file_path.exists(): logger.error(f"File not found: {abs_file_path}"); return False
        if platform.system() == 'Windows': os.startfile(abs_file_path)
        elif platform.system() == 'Darwin': subprocess.run(['open', abs_file_path], check=True)
        else: subprocess.run(['xdg-open', abs_file_path], check=True)
        logger.info(f"Successfully requested to open: {abs_file_path}"); return True
    except Exception as e: logger.error(f"Could not open file {abs_file_path}: {e}"); return False

def main():
    """Main function to run the anomaly detection training and analysis pipeline."""
    log_system_info()
    args = parse_arguments()
    config = Config()

    # Override config from command line args
    config.video_dir = Path(args.video_dir).resolve()
    config.output_dir = Path(args.output_dir).resolve()
    config.model_save_path = Path(args.model_path).resolve()
    if args.batch_size: config.batch_size = args.batch_size
    if args.epochs: config.num_epochs = args.epochs
    if args.lr: config.learning_rate = args.lr
    if args.input_path: config.input_path_arg = Path(args.input_path).resolve() # Store analysis path

    logger.info("--- Initialized Arguments ---")
    logger.info(f"  Video Directory: {config.video_dir}")
    logger.info(f"  Output Directory: {config.output_dir}")
    logger.info(f"  Model Path: {config.model_save_path}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Analyze after training: {args.analyze}")
    if args.analyze:
        logger.info(f"  Analysis Input Path: {config.input_path_arg}")
        logger.info(f"  Analysis Job Type: {args.job_type}")
    logger.info("--- Arguments Initialized ---")

    try:
        verify_directories(config)

        # --- 1. Data Preparation ---
        logger.info("--- 1. Preparing Data ---")
        data_processor = MovementDataProcessor(config)
        # prepare_data now only returns loaders
        train_loader, val_loader = data_processor.prepare_data()

        if train_loader is None or val_loader is None:
            logger.error("  Data preparation failed. Exiting.")
            return
        logger.info(f"  Training set size: {len(train_loader.dataset)}")
        logger.info(f"  Validation set size: {len(val_loader.dataset)}")
        logger.info("--- Data preparation complete ---")

        # --- 2. Model Training ---
        logger.info("--- 2. Initializing and Training Model ---")
        trainer = MovementTrainer(config)
        # Train the model; best model is saved by the trainer
        trainer.train(train_loader, val_loader)
        logger.info("--- Model Training Complete ---")

        # --- 3. Model Evaluation ---
        logger.info("--- 3. Evaluating Trained Model ---")
        # Load the BEST saved model and its info for evaluation
        logger.info(f"Loading best model from {config.model_save_path} for evaluation...")
        best_model, model_info = load_movement_classifier(config.model_save_path, config)

        if best_model is None or model_info is None:
             logger.error("Failed to load the trained model or model info for evaluation. Exiting.")
             return

        # Create the anomaly evaluator with the loaded model and info
        evaluator = AnomalyEvaluator(best_model, config, model_info)

        # Evaluate the model to calculate reconstruction error statistics and save thresholds
        evaluation_results = evaluator.evaluate_model(val_loader)

        if evaluation_results:
            logger.info("--- Model Evaluation Complete ---")
            # Anomaly statistics and thresholds should be logged within evaluator.evaluate_model
            # Example: Find and visualize anomalies (uses thresholds calculated in evaluate_model)
            anomalies = evaluator.find_anomalies(val_loader, output_limit=10)
            if anomalies:
                logger.info(f"Found {len(anomalies)} potential anomalies in validation data based on calculated thresholds")
                for i, anomaly in enumerate(anomalies[:5]):
                    logger.info(f"  Anomaly {i+1}: Error={anomaly['error']:.4f}, Job={anomaly['job_type']}")
            # Analyze error distribution
            error_distribution = evaluator.analyze_error_distribution(val_loader)
            if error_distribution: logger.info("Error distribution analysis complete")
        else:
            logger.error("Model evaluation failed.")
            # Optionally, try to proceed with analysis using default thresholds if needed
            logger.warning("Evaluation failed, analysis might use default/missing thresholds.")


        # --- 4. Optionally Analyze Videos if --analyze flag is set ---
        if args.analyze:
            logger.info("--- 4. Analyzing Videos ---")
            # Re-initialize VideoAnalyzer or BatchProcessor with the *loaded* best model/info
            # Note: VideoAnalyzer loads the model itself using config.model_save_path,
            # which should now point to the best trained model. It will also load model_info.json.
            # We need to ensure model_info.json was correctly updated by the evaluator.

            if config.input_path_arg is None:
                logger.error("Analysis requested (--analyze) but no input path provided (--input_path).")
                return

            try:
                analyzer = VideoAnalyzer(config) # Re-init to load the latest saved model/info
                logger.info("VideoAnalyzer initialized for analysis.")
            except Exception as e:
                 logger.error(f"Failed to initialize VideoAnalyzer for analysis: {e}", exc_info=True)
                 return # Cannot proceed with analysis

            analysis_func_args = {
                "job_type": args.job_type,
                "skip_frames": args.skip_frames,
                "fps_override": args.fps
            }

            if config.input_path_arg.is_file():
                logger.info(f"  Analyzing single video: {config.input_path_arg.name}")
                try:
                    output_path = analyzer.analyze_video(video_path=config.input_path_arg, **analysis_func_args)
                    if output_path: logger.info(f"  Analysis results (video) saved to: {output_path}"); open_file(output_path) if args.show_results else None
                    else: logger.warning(f"  Analysis failed for {config.input_path_arg.name}")
                except Exception as e: logger.error(f"  Error analyzing video: {config.input_path_arg.name} Error: {e}", exc_info=True)
            elif config.input_path_arg.is_dir():
                logger.info(f"  Analyzing directory of videos: {config.input_path_arg}")
                from processors.batch_processor import AnalysisBatchProcessor
                batch_processor = AnalysisBatchProcessor(config, analyzer) # Pass analyzer instance
                try:
                    results = batch_processor.batch_analyze_videos(directory_path=config.input_path_arg, **analysis_func_args)
                    if results:
                        logger.info("  Batch analysis complete.")
                        for result in results:
                            if result['status'] == 'success' and result['output']: logger.info(f"  Analyzed: {result['input']} Output: {result['output']}"); open_file(result['output']) if args.show_results else None
                            else: logger.warning(f"  Analysis failed for: {result['input']} Error: {result['error']}")
                        summary_path = batch_processor.create_batch_summary(results, job_type=args.job_type)
                        if summary_path: logger.info(f"  Batch summary plot saved to: {summary_path}"); open_file(summary_path) if args.show_results else None
                    else: logger.warning("  Batch analysis returned no results.")
                except Exception as e: logger.error(f"  Error during batch analysis: {e}", exc_info=True)
            else: logger.error(f"  Input path is not a file or directory: {config.input_path_arg}")
        else: logger.info("Skipping video analysis as --analyze was not set.")

        logger.info("--- Main script finished ---")

    except (FileNotFoundError, NotADirectoryError) as e: logger.error(f"File/Directory Error: {e}", exc_info=True); print(f"\nERROR: {e}", file=sys.stderr)
    except ValueError as e: logger.error(f"Data or Configuration Error: {e}", exc_info=True); print(f"\nERROR: {e}", file=sys.stderr)
    except Exception as e: logger.critical("An unexpected error occurred:", exc_info=True); print(f"\nFATAL ERROR: {e}", file=sys.stderr)
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache(); logger.debug("CUDA cache cleared.")

if __name__ == "__main__":
    main()