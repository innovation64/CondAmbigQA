#!/usr/bin/env python3
"""
Batch processor for running annotations on large datasets
This script helps process large datasets by splitting them into batches
and optionally running them in parallel or distributed modes.
"""

import os
import json
import argparse
import logging
import subprocess
import time
import signal
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"batch_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Global flag to indicate if we're shutting down
is_shutting_down = False

# Global list to track child processes
child_processes = []

class ProcessingStatus:
    """Track the status of batch processing"""
    
    def __init__(self, total_batches, total_items):
        self.total_batches = total_batches
        self.total_items = total_items
        self.completed_batches = 0
        self.completed_items = 0
        self.start_time = time.time()
        self.current_batch_start = time.time()
        self.successful_items = 0
        self.failed_items = 0
        
    def update_batch(self, successful_items, failed_items):
        """Update batch completion status"""
        self.completed_batches += 1
        self.successful_items += successful_items
        self.failed_items += failed_items
        self.completed_items += successful_items + failed_items
        self.current_batch_start = time.time()
        
    def get_progress_report(self):
        """Get detailed progress report"""
        elapsed = time.time() - self.start_time
        if self.completed_batches > 0:
            avg_batch_time = elapsed / self.completed_batches
            remaining_batches = self.total_batches - self.completed_batches
            estimated_remaining = avg_batch_time * remaining_batches
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
        else:
            avg_batch_time = 0
            estimated_remaining = 0
            eta = "Unknown"
            
        if self.completed_items > 0:
            avg_item_time = elapsed / self.completed_items
            remaining_items = self.total_items - self.completed_items
            item_estimated_remaining = avg_item_time * remaining_items
            item_eta = datetime.now() + timedelta(seconds=item_estimated_remaining)
        else:
            avg_item_time = 0
            item_estimated_remaining = 0
            item_eta = "Unknown"
        
        return f"""
=== Processing Progress Report ===
Time elapsed: {timedelta(seconds=int(elapsed))}
Batch progress: {self.completed_batches}/{self.total_batches} ({self.completed_batches/self.total_batches*100:.1f}%)
Item progress: {self.completed_items}/{self.total_items} ({self.completed_items/self.total_items*100:.1f}%)
Successful items: {self.successful_items} ({self.successful_items/max(1, self.completed_items)*100:.1f}% success rate)
Failed items: {self.failed_items}
Average time per batch: {timedelta(seconds=int(avg_batch_time))}
Average time per item: {timedelta(seconds=int(avg_item_time))}
Estimated time remaining: {timedelta(seconds=int(estimated_remaining))}
Estimated completion: {eta if isinstance(eta, str) else eta.strftime('%Y-%m-%d %H:%M:%S')}
"""

def signal_handler(sig, frame):
    """
    Handle interruption signals gracefully
    """
    global is_shutting_down
    if is_shutting_down:
        logger.info("Forced shutdown requested. Exiting immediately.")
        # Kill all child processes
        for proc in child_processes:
            try:
                if proc.poll() is None:  # If process is still running
                    if os.name == 'nt':  # Windows
                        proc.kill()  # More direct killing method
                    else:  # Unix/Linux
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # Stronger kill signal
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
        sys.exit(1)
    
    logger.info("Interrupt received. Completing current batch and saving progress...")
    is_shutting_down = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
if hasattr(signal, 'SIGBREAK'):  # Windows Ctrl+Break
    signal.signal(signal.SIGBREAK, signal_handler)

def check_kill_switch():
    """Check if kill switch file exists"""
    if os.path.exists("STOP.txt"):
        logger.info("STOP.txt file detected. Graceful shutdown requested.")
        return True
    return False

def check_existing_batches(batch_dir: str, input_file: str, batch_size: int) -> List[str]:
    """
    Check if batch files already exist and use them if valid
    
    Args:
        batch_dir: Directory for batch files
        input_file: Path to the input dataset file
        batch_size: Number of items per batch
        
    Returns:
        List of paths to the batch files
    """
    if os.path.exists(batch_dir) and os.listdir(batch_dir):
        logger.info(f"Detected existing batch files, verifying integrity...")
        
        # List all batch files
        batch_files = [os.path.join(batch_dir, f) for f in sorted(os.listdir(batch_dir)) 
                       if f.startswith("batch_") and f.endswith(".json")]
        
        if not batch_files:
            logger.info(f"No valid batch files found, will create new batches")
            return split_dataset(input_file, batch_dir, batch_size)
        
        # Verify batch file integrity
        total_items = 0
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    total_items += len(batch_data)
            except Exception as e:
                logger.error(f"Batch file {batch_file} is corrupted: {e}")
                logger.info("Will recreate all batch files")
                return split_dataset(input_file, batch_dir, batch_size)
        
        # Verify total item count matches the input file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
                if len(input_data) != total_items:
                    logger.warning(f"Batch total item count ({total_items}) does not match input file ({len(input_data)})")
                    logger.info("Will recreate all batch files")
                    return split_dataset(input_file, batch_dir, batch_size)
        except Exception as e:
            logger.error(f"Cannot read input file: {e}")
            # If input file can't be read but batch files are valid, still use existing batches
            logger.warning("Will use existing batch files, but cannot verify their integrity")
            
        logger.info(f"Verification passed, will use {len(batch_files)} existing batch files")
        return batch_files
    else:
        logger.info(f"No existing batch files detected, will create new batches")
        return split_dataset(input_file, batch_dir, batch_size)

def split_dataset(input_file: str, output_dir: str, batch_size: int = 100) -> List[str]:
    """
    Split a large dataset into smaller batches
    
    Args:
        input_file: Path to the input dataset file
        output_dir: Directory to save the batch files
        batch_size: Number of items per batch
        
    Returns:
        List of paths to the batch files
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Calculate number of batches
        num_items = len(data)
        num_batches = (num_items + batch_size - 1) // batch_size  # Ceiling division
        
        logger.info(f"Splitting dataset with {num_items} items into {num_batches} batches of approximately {batch_size} items each")
        
        batch_files = []
        
        # Create batch files
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_items)
            
            batch_data = data[start_idx:end_idx]
            batch_file = os.path.join(output_dir, f"batch_{i+1:04d}.json")
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            batch_files.append(batch_file)
            logger.info(f"Created batch {i+1}/{num_batches}: {batch_file} with {len(batch_data)} items")
        
        return batch_files
        
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        return []

def create_batch_status_map(output_dir: str, batch_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Create a mapping showing the processing status of each batch
    
    Args:
        output_dir: Output directory
        batch_files: List of batch files
        
    Returns:
        Batch status mapping
    """
    progress_file = os.path.join(output_dir, "progress.json")
    status_map = {}
    
    # Load existing status from progress file
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                
            # Convert progress data to status map
            for batch_file in batch_files:
                batch_name = os.path.basename(batch_file)
                output_file = os.path.join(output_dir, f"annotated_{batch_name}")
                
                if batch_name in progress:
                    batch_info = progress[batch_name]
                    # Verify output file actually exists and is valid
                    if batch_info.get("completed", False) and os.path.exists(output_file):
                        try:
                            with open(output_file, 'r', encoding='utf-8') as f:
                                output_data = json.load(f)
                                if len(output_data) > 0:
                                    # Output file is valid
                                    status_map[batch_file] = {
                                        "status": "completed",
                                        "output_file": output_file,
                                        "items_count": batch_info.get("items_count", 0),
                                        "successful_items": batch_info.get("successful_items", 0),
                                        "failed_items": batch_info.get("failed_items", 0)
                                    }
                                    continue
                        except Exception as e:
                            logger.warning(f"Output file {output_file} is corrupted: {e}")
                
                # If we get here, the batch is not completed or output file is invalid
                status_map[batch_file] = {
                    "status": "pending",
                    "output_file": output_file,
                    "items_count": 0,
                    "successful_items": 0,
                    "failed_items": 0
                }
        except Exception as e:
            logger.warning(f"Cannot load progress file: {e}")
            # Create empty status map
            for batch_file in batch_files:
                batch_name = os.path.basename(batch_file)
                output_file = os.path.join(output_dir, f"annotated_{batch_name}")
                status_map[batch_file] = {
                    "status": "pending",
                    "output_file": output_file,
                    "items_count": 0,
                    "successful_items": 0,
                    "failed_items": 0
                }
    else:
        # If progress file doesn't exist, create new status map
        for batch_file in batch_files:
            batch_name = os.path.basename(batch_file)
            output_file = os.path.join(output_dir, f"annotated_{batch_name}")
            # Check if output file already exists (might be result of previous run)
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        output_data = json.load(f)
                    with open(batch_file, 'r', encoding='utf-8') as f:
                        batch_data = json.load(f)
                    
                    status_map[batch_file] = {
                        "status": "completed",
                        "output_file": output_file,
                        "items_count": len(batch_data),
                        "successful_items": len(output_data),
                        "failed_items": len(batch_data) - len(output_data)
                    }
                    logger.info(f"Detected previously processed batch: {batch_file}")
                except Exception as e:
                    logger.warning(f"Detected output file {output_file} but cannot verify: {e}")
                    status_map[batch_file] = {
                        "status": "pending",
                        "output_file": output_file,
                        "items_count": 0,
                        "successful_items": 0,
                        "failed_items": 0
                    }
            else:
                status_map[batch_file] = {
                    "status": "pending",
                    "output_file": output_file,
                    "items_count": 0,
                    "successful_items": 0,
                    "failed_items": 0
                }
    
    # Count completed and pending batches
    completed = sum(1 for info in status_map.values() if info["status"] == "completed")
    pending = len(status_map) - completed
    logger.info(f"Batch status: {completed} completed, {pending} pending")
    
    return status_map

def merge_results(batch_outputs: List[str], final_output: str) -> bool:
    """
    Merge multiple batch outputs into a single file
    
    Args:
        batch_outputs: List of batch output files
        final_output: Path to save the merged output
        
    Returns:
        True if merging was successful, False otherwise
    """
    try:
        logger.info(f"Merging {len(batch_outputs)} batch outputs")
        
        merged_data = []
        
        # Load and merge each batch output
        for batch_file in batch_outputs:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                
                # Handle different data formats
                if isinstance(batch_data, dict):
                    # If the data is a dictionary, convert to list
                    batch_list = list(batch_data.values())
                else:
                    batch_list = batch_data
                
                merged_data.extend(batch_list)
                logger.info(f"Added {len(batch_list)} items from {batch_file}")
            except Exception as e:
                logger.error(f"Error processing batch file {batch_file}: {str(e)}")
        
        # Save the merged data
        with open(final_output, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully merged {len(merged_data)} total items to {final_output}")
        return True
    
    except Exception as e:
        logger.error(f"Error merging batch outputs: {str(e)}")
        return False

def process_batch(batch_file: str, output_file: str, parallel: int = 1, convert: bool = False, stats: bool = False, timeout: int = 3600) -> bool:
    """Process a single batch with better output handling"""
    global is_shutting_down
    global child_processes
    
    try:
        logger.info(f"Processing batch: {batch_file}")
        
        # Prepare the command
        cmd = ["python", "main.py", "--input", batch_file, "--output", output_file, "--parallel", "1"]
        
        if convert:
            cmd.append("--convert")
        
        if stats:
            cmd.append("--stats")
        
        # Run the command with better output handling
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Create process
        import subprocess
        import threading
        import queue
        import os
        import time
        import platform
        import re
        
        # Check if we're on Windows
        is_windows = os.name == 'nt' or platform.system() == 'Windows'
        
        # Use subprocess.PIPE for capturing output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            # Windows does not support preexec_fn
            preexec_fn=None if is_windows else os.setsid
        )
        
        # Add to global process list
        child_processes.append(process)
        
        # Start timeout timer and heartbeat
        start_time = time.time()
        last_output_time = time.time()
        
        # Create queues for output
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        
        # Regex patterns to identify log levels
        info_pattern = re.compile(r'INFO|INFO\s-')
        warning_pattern = re.compile(r'WARNING|WARN\s-')
        error_pattern = re.compile(r'ERROR|CRITICAL|EXCEPTION|Traceback|Error:|Exception:')
        
        # Function to read output and put it in queue
        def read_output(pipe, queue):
            for line in iter(pipe.readline, ''):
                queue.put(line)
            pipe.close()
        
        # Start threads to read output
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_queue))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_queue))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Helper function to determine appropriate log level
        def determine_log_level(line):
            if error_pattern.search(line):
                return "error"
            elif warning_pattern.search(line):
                return "warning"
            elif info_pattern.search(line):
                return "info"
            else:
                return "info"  # Default to info for unclassified lines
        
        # Monitor and process output until completion
        is_finished = False
        while not is_finished:
            # Check if process has finished
            exit_code = process.poll()
            if exit_code is not None:
                is_finished = True
            
            # Check if we should shut down
            if is_shutting_down or check_kill_switch():
                logger.info("Graceful shutdown requested. Terminating process...")
                try:
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    time.sleep(2)
                    # Force kill if still running
                    if process.poll() is None:
                        logger.warning("Process did not terminate gracefully. Forcing termination...")
                        process.kill()
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")
                return False
            
            # Check timeout
            current_time = time.time()
            if timeout > 0 and current_time - start_time > timeout:
                logger.warning(f"Process timed out after {timeout} seconds")
                try:
                    process.terminate()
                    # Force kill after a short wait if still running
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                except Exception as e:
                    logger.error(f"Error terminating process after timeout: {e}")
                return False
            
            # Check for output stalling
            if current_time - last_output_time > 300:  # 5 minutes with no output
                logger.warning("No output received for 5 minutes, process may be stalled")
                # Optional: add code to terminate stalled processes
            
            # Process any available stdout
            try:
                while True:
                    try:
                        line = stdout_queue.get_nowait().strip()
                        if line:
                            # Use appropriate log level based on content
                            log_level = determine_log_level(line)
                            if log_level == "error":
                                logger.error(f"MAIN: {line}")
                            elif log_level == "warning":
                                logger.warning(f"MAIN: {line}")
                            else:
                                logger.info(f"MAIN: {line}")
                            last_output_time = current_time
                    except queue.Empty:
                        break
            except Exception as e:
                logger.error(f"Error processing stdout: {e}")
            
            # Process any available stderr
            try:
                while True:
                    try:
                        line = stderr_queue.get_nowait().strip()
                        if line:
                            # Use appropriate log level based on content
                            log_level = determine_log_level(line)
                            if log_level == "error":
                                logger.error(f"MAIN ERROR: {line}")
                            elif log_level == "warning":
                                logger.warning(f"MAIN WARNING: {line}")
                            else:
                                logger.info(f"MAIN OUTPUT: {line}")
                            last_output_time = current_time
                    except queue.Empty:
                        break
            except Exception as e:
                logger.error(f"Error processing stderr: {e}")
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.1)
        
        # Wait for output threads to finish
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        # Process any remaining output
        try:
            while not stdout_queue.empty():
                line = stdout_queue.get_nowait().strip()
                if line:
                    log_level = determine_log_level(line)
                    if log_level == "error":
                        logger.error(f"MAIN FINAL ERROR: {line}")
                    elif log_level == "warning":
                        logger.warning(f"MAIN FINAL WARNING: {line}")
                    else:
                        logger.info(f"MAIN FINAL: {line}")
            
            while not stderr_queue.empty():
                line = stderr_queue.get_nowait().strip()
                if line:
                    log_level = determine_log_level(line)
                    if log_level == "error":
                        logger.error(f"MAIN FINAL ERROR: {line}")
                    elif log_level == "warning":
                        logger.warning(f"MAIN FINAL WARNING: {line}")
                    else:
                        logger.info(f"MAIN FINAL OUTPUT: {line}")
        except Exception as e:
            logger.error(f"Error processing remaining output: {e}")
        
        # Remove from global process list
        if process in child_processes:
            child_processes.remove(process)
        
        # Check if the process completed successfully
        if process.returncode != 0:
            logger.error(f"Batch processing failed with exit code {process.returncode}")
            return False
        
        # Check if output file was created
        if not os.path.exists(output_file):
            logger.error(f"Output file {output_file} was not created")
            return False
            
        # Validate output file
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            logger.info(f"Successfully processed batch: {batch_file} ({len(output_data)} items)")
            return True
        except Exception as e:
            logger.error(f"Error validating output file: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}", exc_info=True)
        return False

def run_sequential(batch_files: List[str], output_dir: str, status_map: Dict[str, Dict[str, Any]], 
                   parallel: int = 1, convert: bool = False, stats: bool = False, 
                   timeout: int = 3600, status_update_interval: int = 300) -> List[str]:
    """
    Process batches sequentially
    
    Args:
        batch_files: List of batch files to process
        output_dir: Directory to save the output files
        status_map: Batch status mapping
        parallel: Number of parallel processes for each batch
        convert: Whether to convert to MCAQAR format
        stats: Whether to show statistics
        timeout: Maximum timeout per batch in seconds
        status_update_interval: Status update interval in seconds
        
    Returns:
        List of output files
    """
    global is_shutting_down
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = []
    
    # Calculate total items
    total_items = 0
    for batch_file in batch_files:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                total_items += len(batch_data)
        except Exception as e:
            logger.warning(f"Unable to read item count from {batch_file}: {e}")
    
    # Initialize status tracker
    status = ProcessingStatus(len(batch_files), total_items)
    
    # Update status for already completed batches
    completed_count = 0
    completed_items = 0
    successful_items = 0
    failed_items = 0
    
    for batch_file, info in status_map.items():
        if info["status"] == "completed":
            output_files.append(info["output_file"])
            completed_count += 1
            completed_items += info["items_count"]
            successful_items += info["successful_items"]
            failed_items += info["failed_items"]
    
    status.completed_batches = completed_count
    status.completed_items = completed_items
    status.successful_items = successful_items
    status.failed_items = failed_items
    
    # Create progress bars
    batch_pbar = tqdm(total=len(batch_files), desc="Batch progress", position=0)
    items_pbar = tqdm(total=total_items, desc="Item progress", position=1)
    
    # Update progress bars to reflect already completed batches
    batch_pbar.update(completed_count)
    items_pbar.update(completed_items)
    
    # Create progress tracking file
    progress_file = os.path.join(output_dir, "progress.json")
    progress = {}
    
    # Load progress if file exists
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        except:
            logger.warning("Failed to load progress file. Starting fresh.")
            progress = {}
    
    # Last status update time
    last_status_update = time.time()
    
    # Only process pending batches
    pending_batch_files = [batch_file for batch_file in batch_files 
                          if status_map[batch_file]["status"] == "pending"]
    
    for i, batch_file in enumerate(pending_batch_files):
        # Check if we should shut down
        if is_shutting_down or check_kill_switch():
            logger.info("Graceful shutdown in progress. Saving current state...")
            break
            
        batch_name = os.path.basename(batch_file)
        output_file = os.path.join(output_dir, f"annotated_{batch_name}")
        
        # Get item count in batch
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                batch_data_count = len(batch_data)
        except Exception as e:
            logger.error(f"Unable to read batch file {batch_file}: {e}")
            batch_data_count = 0
        
        # Update progress to in-progress
        progress[batch_name] = {
            "started": True, 
            "completed": False, 
            "start_time": time.time(),
            "items_count": batch_data_count
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        current_position = completed_count + i + 1
        total_batches = len(batch_files)
        logger.info(f"Processing batch {current_position}/{total_batches}: {batch_file}")
        
        start_time = time.time()
        
        success = process_batch(batch_file, output_file, parallel, convert, stats, timeout)
        
        processing_time = time.time() - start_time
        
        # Get successful/failed item counts
        successful_items = 0
        failed_items = 0
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                    successful_items = len(output_data)
                    failed_items = batch_data_count - successful_items
            except Exception as e:
                logger.error(f"Unable to read output file {output_file}: {e}")
        
        # Update progress
        progress[batch_name] = {
            "started": True,
            "completed": success,
            "start_time": start_time,
            "end_time": time.time(),
            "processing_time": processing_time,
            "items_count": batch_data_count,
            "successful_items": successful_items,
            "failed_items": failed_items
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        if success:
            output_files.append(output_file)
            logger.info(f"Batch {current_position}/{total_batches} completed successfully in {processing_time:.2f} seconds")
        else:
            logger.error(f"Batch {current_position}/{total_batches} failed after {processing_time:.2f} seconds")
        
        # Update progress bars
        batch_pbar.update(1)
        items_pbar.update(batch_data_count)
        
        # Update status
        status.update_batch(successful_items, failed_items)
        
        # Periodically display status updates
        if time.time() - last_status_update > status_update_interval:
            print(status.get_progress_report())
            last_status_update = time.time()
        
        # Brief pause between batches to avoid API rate limits
        time.sleep(2)
    
    # Close progress bars
    batch_pbar.close()
    items_pbar.close()
    
    # Final progress report
    print(status.get_progress_report())
    
    logger.info(f"Sequential processing completed: {status.successful_items}/{status.completed_items} items successful")
    return output_files

def main():
    parser = argparse.ArgumentParser(description="Batch processor for large dataset annotation")
    
    parser.add_argument("--input", "-i", type=str, required=True, help="Input dataset file")
    parser.add_argument("--output", "-o", type=str, default="annotated_dataset.json", help="Final output file")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Number of items per batch")
    parser.add_argument("--batch-dir", "-d", type=str, default="batches", help="Directory for batch files")
    parser.add_argument("--output-dir", "-od", type=str, default="outputs", help="Directory for batch outputs")
    parser.add_argument("--parallel", "-p", type=int, default=1, help="Parallel processes per batch")
    parser.add_argument("--convert", "-c", action="store_true", help="Convert to MCAQAR format")
    parser.add_argument("--stats", "-s", action="store_true", help="Show statistics")
    parser.add_argument("--timeout", "-t", type=int, default=3600, help="Timeout per batch in seconds (0 for no timeout)")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from last successful batch")
    parser.add_argument("--status-interval", "-si", type=int, default=300, help="Status update interval in seconds")
    parser.add_argument("--force-resplit", "-fr", action="store_true", help="Force resplit of dataset, ignoring existing batches")
    
    # Additional modes
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--split-only", action="store_true", help="Only split the dataset, don't process")
    group.add_argument("--merge-only", action="store_true", help="Only merge existing outputs")
    group.add_argument("--process-batch", type=str, help="Process a single batch file")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Ensure output directories exist
    os.makedirs(args.batch_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.merge_only:
            # Only merge existing outputs
            logger.info("Merge-only mode selected")
            
            # Find all output files
            output_files = []
            for filename in os.listdir(args.output_dir):
                if filename.startswith("annotated_batch_") and filename.endswith(".json"):
                    output_files.append(os.path.join(args.output_dir, filename))
            
            if not output_files:
                logger.error(f"No output files found in {args.output_dir}")
                return
            
            logger.info(f"Found {len(output_files)} output files to merge")
            
            # Merge the outputs
            success = merge_results(output_files, args.output)
            
            if success:
                logger.info(f"Successfully merged outputs to {args.output}")
            else:
                logger.error("Failed to merge outputs")
            
            return
        
        if args.process_batch:
            # Process a single batch file
            batch_file = args.process_batch
            output_file = os.path.join(args.output_dir, f"annotated_{os.path.basename(batch_file)}")
            
            logger.info(f"Processing single batch: {batch_file}")
            success = process_batch(
                batch_file, 
                output_file, 
                args.parallel, 
                args.convert, 
                args.stats,
                args.timeout
            )
            
            if success:
                logger.info(f"Successfully processed batch: {batch_file}")
            else:
                logger.error(f"Failed to process batch: {batch_file}")
            
            return
        
        # Use improved batch verification, check or create batches
        if args.force_resplit:
            logger.info("Forcing resplit of dataset...")
            batch_files = split_dataset(args.input, args.batch_dir, args.batch_size)
        else:
            batch_files = check_existing_batches(args.batch_dir, args.input, args.batch_size)
        
        if not batch_files:
            logger.error("Failed to split dataset")
            return
        
        if args.split_only:
            logger.info("Split-only mode selected. Dataset split into batches. Exiting.")
            return
        
        # Create status map, identify already processed batches
        status_map = create_batch_status_map(args.output_dir, batch_files)
        
        # Process batches
        output_files = run_sequential(
            batch_files, 
            args.output_dir,
            status_map,
            args.parallel, 
            args.convert, 
            args.stats,
            args.timeout,
            args.status_interval
        )
        
        # Check if we're shutting down
        if is_shutting_down or check_kill_switch():
            logger.info("Graceful shutdown completed. Merging processed outputs.")
        
        if not output_files:
            logger.error("No batches were processed successfully")
            return
        
        # Merge the outputs
        success = merge_results(output_files, args.output)
        
        if success:
            logger.info(f"Successfully merged outputs to {args.output}")
        else:
            logger.error("Failed to merge outputs")
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Unhandled exception in main process: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()