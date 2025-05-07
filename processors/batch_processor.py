import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import gc
import torch
from .video_analyzer import VideoAnalyzer 
import os
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track CUDA devices to prevent race conditions
cuda_lock = threading.Lock()

class AnalysisBatchProcessor:
    """Batch processing of multiple videos for movement analysis."""

    def __init__(self, config, analyzer):
        """
        Initialize analysis batch processor.

        Args:
            config: Configuration object
            analyzer: VideoAnalyzer instance (expected to have analyze_video method)
        """
        self.config = config
        self.analyzer = analyzer
        # Create a thread-local storage for analyzers
        self.thread_local = threading.local()
        # Store batch results
        self.batch_results = []

    def _get_thread_analyzer(self):
        """Get or create analyzer instance for current thread with improved device management."""
        if not hasattr(self.thread_local, 'analyzer'):
            # Import copy only when needed
            import copy
            thread_config = copy.deepcopy(self.config)
            
            # IMPROVED GPU MANAGEMENT: Use more sophisticated device assignment
            if torch.cuda.is_available():
                with cuda_lock:
                    try:
                        # Get memory info for all devices
                        available_devices = []
                        for i in range(torch.cuda.device_count()):
                            # Get memory stats
                            total_mem = torch.cuda.get_device_properties(i).total_memory
                            used_mem = torch.cuda.memory_allocated(i)
                            free_mem = total_mem - used_mem
                            available_devices.append((i, free_mem))
                        
                        if available_devices:
                            # Sort by free memory (descending)
                            available_devices.sort(key=lambda x: x[1], reverse=True)
                            # Get device with most free memory
                            best_device_idx = available_devices[0][0]
                            thread_config.device = torch.device(f'cuda:{best_device_idx}')
                            logger.info(f"Thread {threading.get_ident()} assigned to CUDA device {best_device_idx} with most free memory")
                            
                            # Create thread-specific CUDA stream
                            with torch.cuda.device(best_device_idx):
                                self.thread_local.cuda_stream = torch.cuda.Stream()
                        else:
                            logger.warning("No CUDA devices available. Using CPU.")
                            thread_config.device = torch.device('cpu')
                    except Exception as e:
                        logger.warning(f"Error setting CUDA for thread: {e}. Using CPU.")
                        thread_config.device = torch.device('cpu')
            
            try:
                self.thread_local.analyzer = VideoAnalyzer(thread_config)
                logger.debug(f"Created analyzer for thread {threading.get_ident()} on device {thread_config.device}")
            except Exception as e:
                logger.error(f"Failed to create analyzer for thread {threading.get_ident()}: {e}")
                raise RuntimeError(f"Failed to initialize analyzer: {e}")
        
        return self.thread_local.analyzer

    def _analyze_single_video_wrapper(self, video_path, job_type, skip_frames, fps_override):
        """Improved video analysis wrapper with better memory management."""
        video_path = Path(video_path)
        try:
            # Get memory stats before processing
            if torch.cuda.is_available():
                before_mem = torch.cuda.memory_allocated() / (1024**2)
                logger.debug(f"GPU memory before processing {video_path.name}: {before_mem:.1f} MB")
            
            thread_analyzer = self._get_thread_analyzer()
            
            # IMPROVED MEMORY MANAGEMENT: Set max CUDA memory fragment size
            if hasattr(thread_analyzer, 'config') and torch.cuda.is_available():
                # Limit max split size for better memory fragmentation handling
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of available memory
            
            # Use CUDA stream if available
            if hasattr(self.thread_local, 'cuda_stream') and self.thread_local.cuda_stream is not None:
                with torch.cuda.stream(self.thread_local.cuda_stream):
                    output_path = thread_analyzer.analyze_video(
                        video_path, job_type, None, skip_frames, fps_override
                    )
            else:
                output_path = thread_analyzer.analyze_video(
                    video_path, job_type, None, skip_frames, fps_override
                )
            
            # Log memory usage after processing
            if torch.cuda.is_available():
                after_mem = torch.cuda.memory_allocated() / (1024**2)
                logger.debug(f"GPU memory after processing {video_path.name}: {after_mem:.1f} MB (Delta: {after_mem - before_mem:.1f} MB)")
            
            return output_path
            
        except torch.cuda.OutOfMemoryError as oom_error:
            logger.error(f"CUDA out of memory error processing {video_path.name}: {oom_error}")
            # Aggressive cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            # Try again with CPU fallback
            logger.info(f"Retrying {video_path.name} on CPU")
            try:
                # Create a CPU-only config
                import copy
                cpu_config = copy.deepcopy(self.config)
                cpu_config.device = torch.device('cpu')
                
                # Create a CPU analyzer
                cpu_analyzer = VideoAnalyzer(cpu_config)
                
                # Process on CPU
                output_path = cpu_analyzer.analyze_video(
                    video_path, job_type, None, skip_frames, fps_override
                )
                
                return output_path
            except Exception as retry_error:
                logger.error(f"CPU retry failed for {video_path.name}: {retry_error}")
                raise
        except Exception as e:
            logger.error(f"Error analyzing {video_path.name}: {e}", exc_info=True)
            raise
        finally:
            # Always cleanup after each video
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Add a method to manage overall batch memory
    def _batch_memory_check(self):
        """Monitor and manage memory during batch processing."""
        if not torch.cuda.is_available():
            return
            
        try:
            # Log current memory state
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                
                logger.info(f"CUDA:{i} - Total: {total:.2f}GB, Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                self._batch_memory_check()
                # If memory usage is too high, trigger aggressive cleanup
                if allocated / total > 0.85:  # Over 85% usage
                    logger.warning(f"High memory usage detected on CUDA:{i} - Triggering cleanup")
                    torch.cuda.empty_cache()
                    gc.collect()
        except Exception as e:
            logger.warning(f"Error during memory check: {e}")
        
    def batch_analyze_videos(self, directory_path, job_type=None, skip_frames=0, fps_override=None, timeout=3600):
        """
        Analyze all videos in a directory with timeout.

        Args:
            directory_path: Path to directory containing videos
            job_type: Optional job category name
            skip_frames: Number of frames to skip between processing
            fps_override: Optional FPS override for all videos
            timeout: Maximum time in seconds to process a single video (default: 1 hour)

        Returns:
            List of dictionaries, each containing results for a processed video
        """
        # Ensure Path object
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise FileNotFoundError(f"Directory not found or is not a directory: {directory_path}")

        # Get all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(directory_path.glob(f"*{ext}")))
        
        # Remove duplicates and sort
        video_files = sorted(list(set(video_files)))

        if not video_files:
            logger.warning(f"No video files found with extensions {video_extensions} in {directory_path}")
            return []

        logger.info(f"Found {len(video_files)} videos to analyze in {directory_path}")

        # Reset results for this batch
        self.batch_results = []
        
        # Determine optimal worker count based on available resources
        cpu_count = os.cpu_count() or 1
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Default worker calculation considering CPU and GPU constraints
        default_workers = min(cpu_count, max(1, gpu_count * 2)) if gpu_count > 0 else max(1, cpu_count // 2)
        max_workers = getattr(self.config, 'analysis_workers', default_workers)
        
        logger.info(f"Using {max_workers} workers for batch analysis (CPU: {cpu_count}, GPU: {gpu_count})")

        # Create a semaphore to limit concurrent resource usage
        # This is especially important for CUDA operations
        resource_semaphore = threading.Semaphore(max_workers)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use a dictionary to track futures
            futures = {}
            
            for video_path in video_files:
                def submit_with_semaphore(video_path=video_path):
                    # Acquire semaphore before submitting
                    with resource_semaphore:
                        logger.info(f"Submitting {video_path.name} for analysis...")
                        return executor.submit(
                            self._analyze_single_video_wrapper,
                            video_path,
                            job_type,
                            skip_frames,
                            fps_override
                        )
                
                # Submit each job with semaphore control
                future = submit_with_semaphore()
                futures[future] = video_path

            # Process completed futures as they finish
            processed_count = 0
            for future in tqdm(futures.keys(), total=len(futures), desc="Batch Analysis Progress"):
                video_path = futures[future]
                processed_count += 1
                start_time = time.time()
                
                try:
                    # Wait for the result with timeout
                    output_path = future.result(timeout=timeout)
                    processing_time = time.time() - start_time
                    
                    result = {
                        'input': str(video_path),
                        'output': str(output_path) if output_path else None,
                        'status': 'success',
                        'error': None,
                        'processing_time': processing_time
                    }
                    self.batch_results.append(result)
                    logger.info(f"Successfully processed {video_path.name} in {processing_time:.1f}s")
                    
                except TimeoutError:
                    processing_time = time.time() - start_time
                    error_msg = f"Processing timed out after {processing_time:.1f} seconds (limit: {timeout}s)"
                    logger.error(f"Timeout processing {video_path.name}")
                    
                    result = {
                        'input': str(video_path),
                        'output': None,
                        'status': 'error',
                        'error': error_msg,
                        'processing_time': processing_time
                    }
                    self.batch_results.append(result)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    logger.error(f"Error processing {video_path.name}: {e}")
                    logger.error(traceback.format_exc())
                    
                    result = {
                        'input': str(video_path),
                        'output': None,
                        'status': 'error',
                        'error': str(e),
                        'processing_time': processing_time
                    }
                    self.batch_results.append(result)
                
                # Force cleanup after each video
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Create summary report
        success_count = sum(1 for r in self.batch_results if r['status'] == 'success')
        fail_count = len(self.batch_results) - success_count
        logger.info(f"Batch analysis complete. Processed: {len(self.batch_results)}, Succeeded: {success_count}, Failed: {fail_count}")

        # Write detailed results to CSV file
        report_path = Path(self.config.output_dir) / "batch_analysis_results.csv"
        try:
            import csv
            with open(report_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(["input", "output", "status", "error", "processing_time_seconds"])
                # Write data rows
                for r in self.batch_results:
                    writer.writerow([
                        r.get('input', ''),
                        r.get('output', ''),
                        r.get('status', 'unknown'),
                        r.get('error', ''),
                        f"{r.get('processing_time', ''):.2f}"
                    ])
            logger.info(f"Batch analysis report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save batch analysis report: {e}")

        return self.batch_results

    def create_batch_summary(self, results, job_type=None):
        """
        Create summary visualizations for all videos processed in a batch.

        Args:
            results: List of processing results from batch_analyze_videos
            job_type: Optional job category name

        Returns:
            Path to the generated summary visualization, or None if failed
        """
        if not results:
            logger.warning("No results provided for batch summary.")
            return None

        try:
            # Ensure Matplotlib is imported and backend is set
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            successful_results = [r for r in results if r['status'] == 'success']
            failed_results = [r for r in results if r['status'] != 'success']

            if not successful_results and not failed_results:
                 logger.warning("No successful or failed results found to summarize.")
                 return None

            plt.figure(figsize=(10, 6))

            # Plot success rate pie chart
            success_count = len(successful_results)
            fail_count = len(failed_results)

            if success_count > 0 or fail_count > 0:
                 plt.pie([success_count, fail_count],
                        labels=['Success', 'Failed/Timeout'],
                        autopct='%1.1f%%',
                        colors=['#4CAF50', '#F44336'],
                        startangle=90,
                        explode=(0.1 if success_count > 0 and fail_count > 0 else 0, 0)) # Explode success slice
                 plt.title(f'Batch Analysis Results ({success_count+fail_count} Videos)', fontsize=16)
            else:
                 plt.text(0.5, 0.5, "No results to display", ha='center', va='center')
                 plt.title('Batch Analysis Results', fontsize=16)


            # Ensure output directory exists
            summary_output_dir = self.config.output_dir / 'summaries'
            summary_output_dir.mkdir(parents=True, exist_ok=True)

            # Save plots
            job_suffix = f"_{job_type}" if job_type else "_all"
            output_path = summary_output_dir / f"batch_summary{job_suffix}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close() # Close the figure

            logger.info(f"Batch summary visualization saved to: {output_path}")
            return output_path

        except ImportError:
             logger.error("Matplotlib not found. Cannot create batch summary visualization. Please install matplotlib.")
             return None
        except Exception as e:
            logger.error(f"Error creating batch summary visualization: {str(e)}")
            logger.error(traceback.format_exc())
            plt.close() # Ensure plot is closed on error
            return None