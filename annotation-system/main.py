"""
Main entry point for the AI-AI annotation system.
Coordinates the entire annotation process for multiple items.
"""

import os
import json
import logging
import argparse
import time
import concurrent.futures
import signal
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm

import config
from data_processor import DataProcessor
from conversation import AnnotationConversation
from models import AnnotatorAgent, ReviewerAgent, FacilitatorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(config.DEFAULT_LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Global variable declarations
is_shutting_down = False
child_processes = []

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
                if hasattr(proc, 'terminate'):
                    proc.terminate()
                    time.sleep(0.5)
                    if hasattr(proc, 'kill') and (not hasattr(proc, 'poll') or proc.poll() is None):
                        proc.kill()
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
        exit(1)
    
    logger.info("Interrupt received. Completing current processing and saving progress...")
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

def process_item(item: Dict[str, Any], annotator: AnnotatorAgent, reviewer: ReviewerAgent, facilitator: FacilitatorAgent) -> Dict[str, Any]:
    """
    Process a single item using the AI-AI annotation conversation
    
    Args:
        item: The item to annotate
        annotator: The annotator agent
        reviewer: The reviewer agent
        facilitator: The facilitator agent
        
    Returns:
        The annotated item
    """
    item_id = item.get("id", "")
    
    # Fix: Check for question in multiple possible locations
    question = item.get("question", "")
    if not question and "data" in item and "ambiguous_question" in item["data"]:
        question = item["data"]["ambiguous_question"]
    
    fragments = item.get("ctxs", [])
    if not fragments and "retrieval_ctxs" in item:
        fragments = item["retrieval_ctxs"]
    
    logger.info(f"Processing item {item_id}: {question[:50]}...")
    
    # Add timeout handling
    start_time = time.time()
    max_execution_time = getattr(config, "MAX_EXECUTION_TIME", 300)  # 5 minutes default timeout
    
    try:
        # Create and run the conversation
        conversation = AnnotationConversation(
            item_id=item_id,
            question=question,
            fragments=fragments,
            annotator=annotator,
            reviewer=reviewer,
            facilitator=facilitator
        )
        
        # Run the conversation with timeout monitoring
        result = None
        
        # Use a thread with timeout to run the conversation
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(conversation.run_conversation)
            
            # Wait for the result with timeout
            try:
                result = future.result(timeout=max_execution_time)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Processing for item {item_id} timed out after {max_execution_time} seconds")
                # Return minimal valid result on timeout
                return {
                    "id": item_id,
                    "question": question,
                    "properties": [],
                    "error": f"Processing timed out after {max_execution_time} seconds"
                }
        
        if result is None:
            # This shouldn't happen with the timeout, but just in case
            logger.error(f"No result obtained for item {item_id}")
            return {
                "id": item_id,
                "question": question,
                "properties": [],
                "error": "No result obtained from conversation"
            }
        
        logger.info(f"Completed annotation for item {item_id} in {result.get('metadata', {}).get('rounds', 0)} rounds")
        return result
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error processing item {item_id} after {elapsed_time:.2f} seconds: {str(e)}", exc_info=True)
        # Return minimal valid result on error
        return {
            "id": item_id,
            "question": question,
            "properties": [],
            "error": str(e)
        }

def process_batch(
    items: List[Dict[str, Any]], 
    data_processor: DataProcessor, 
    max_parallel: int = 1
) -> int:
    """
    Process a batch of items in parallel
    
    Args:
        items: List of items to process
        data_processor: Data processor for saving results
        max_parallel: Maximum number of items to process in parallel
        
    Returns:
        Number of successfully processed items
    """
    global is_shutting_down
    global child_processes
    
    # Create agents (shared across all items)
    annotator = AnnotatorAgent()
    reviewer = ReviewerAgent()
    facilitator = FacilitatorAgent()
    
    successful = 0
    
    # Create progress bar
    pbar = tqdm(total=len(items), desc="Item progress")
    
    # Define a heartbeat function to ensure we're making progress
    last_progress_time = time.time()
    
    def update_progress():
        nonlocal last_progress_time
        last_progress_time = time.time()
        # Save interim results periodically
        data_processor.save_interim_annotations()
    
    # Process items sequentially if max_parallel <= 1
    if max_parallel <= 1:
        for i, item in enumerate(items):
            # Check if we should shut down
            if is_shutting_down or check_kill_switch():
                logger.info("Shutdown requested. Stopping processing.")
                break
                
            item_id = item.get("id", "")
            question = item.get("question", "")
            if not question and "data" in item and "ambiguous_question" in item["data"]:
                question = item["data"]["ambiguous_question"]
                
            logger.info(f"Processing item {i+1}/{len(items)}: {item_id} - {question[:50]}...")
            pbar.set_description(f"Processing: {question[:30]}...")
            
            try:
                result = process_item(item, annotator, reviewer, facilitator)
                
                # Add annotation to data processor
                if data_processor.add_annotation(result):
                    successful += 1
                    
                # Update progress
                update_progress()
                    
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Completed: {question[:30]}... ({result.get('metadata', {}).get('rounds', 0)} rounds)")
                
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
                pbar.update(1)
                pbar.set_description(f"Failed: {question[:30]}...")
            
            # Short pause between items to avoid API rate limits
            time.sleep(2)
        
        # Save interim results after processing all items
        data_processor.save_interim_annotations()
        pbar.close()
        return successful
    
    # Use ThreadPoolExecutor for parallel processing
    active_tasks = set()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all tasks to the executor
        future_to_item = {}
        for i, item in enumerate(items):
            # Check if we should shut down
            if is_shutting_down or check_kill_switch():
                logger.info("Shutdown requested. Stopping submission of new tasks.")
                break
                
            item_id = item.get("id", "")
            question = item.get("question", "")
            if not question and "data" in item and "ambiguous_question" in item["data"]:
                question = item["data"]["ambiguous_question"]
                
            logger.info(f"Submitting item {i+1}/{len(items)}: {item_id} - {question[:50]}...")
            
            future = executor.submit(process_item, item, annotator, reviewer, facilitator)
            future_to_item[future] = (item, question)
            active_tasks.add(future)
        
        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_item):
            # Check if we should shut down before processing result
            if is_shutting_down:
                for f in list(future_to_item.keys()):
                    if f != future and not f.done():
                        f.cancel()
                break
                
            item, question = future_to_item[future]
            active_tasks.discard(future)
            
            try:
                result = future.result()
                
                # Add annotation to data processor
                if data_processor.add_annotation(result):
                    successful += 1
                    
                # Update progress
                update_progress()
                    
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Completed: {question[:30]}... ({result.get('metadata', {}).get('rounds', 0)} rounds)")
                
            except concurrent.futures.CancelledError:
                logger.warning(f"Task for item with question '{question[:30]}...' was cancelled")
                pbar.update(1)
                pbar.set_description(f"Cancelled: {question[:30]}...")
                
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
                pbar.update(1)
                pbar.set_description(f"Failed: {question[:30]}...")
        
        # Save interim results after all tasks are completed
        data_processor.save_interim_annotations()
    
    pbar.close()
    return successful

def main():
    """Entry point for the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI-AI Annotation System")
    parser.add_argument("--input", "-i", type=str, default=config.DEFAULT_INPUT_PATH, help="Input dataset file")
    parser.add_argument("--output", "-o", type=str, default=config.DEFAULT_OUTPUT_PATH, help="Output file")
    parser.add_argument("--interim", type=str, default=config.DEFAULT_INTERIM_PATH, help="Interim results file")
    parser.add_argument("--batch-size", "-b", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--parallel", "-p", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Maximum number of items to process")
    parser.add_argument("--convert", "-c", action="store_true", help="Convert final output to MCAQAR format")
    parser.add_argument("--stats", "-s", action="store_true", help="Show statistics after processing")
    parser.add_argument("--timeout", "-t", type=int, default=0, help="Timeout per item in seconds (0 for no timeout)")
    parser.add_argument("--resume-from", "-rf", type=int, default=0, help="Resume from specified index")
    
    args = parser.parse_args()
    
    # Check for pre-existing kill switch and remove it
    if os.path.exists("STOP.txt"):
        os.remove("STOP.txt")
        logger.info("Removed pre-existing STOP.txt file")
    
    # Initialize data processor
    data_processor = DataProcessor(
        input_file=args.input,
        output_file=args.output,
        interim_file=args.interim
    )
    
    # Get unannotated items
    unannotated_items = data_processor.get_unannotated_items(args.limit)
    
    if not unannotated_items:
        logger.info("No unannotated items found. Exiting.")
        if args.stats:
            stats = data_processor.get_annotation_statistics()
            print("\n=== Annotation Statistics ===")
            print(f"Total items: {stats['total_items']}")
            print(f"Annotated items: {stats['annotated_items']}")
            print(f"Completion percentage: {stats['completion_percentage']:.2f}%")
            print(f"Average conditions per item: {stats['avg_conditions']:.2f}")
            print(f"Average condition citations: {stats['avg_condition_citations']:.2f}")
            print(f"Average answer citations: {stats['avg_answer_citations']:.2f}")
        return
    
    logger.info(f"Found {len(unannotated_items)} unannotated items.")
    
    # Handle resume from index
    if args.resume_from > 0:
        if args.resume_from < len(unannotated_items):
            logger.info(f"Resuming from index {args.resume_from}")
            unannotated_items = unannotated_items[args.resume_from:]
        else:
            logger.warning(f"Resume index {args.resume_from} is beyond the available items. Starting from the beginning.")
    
    # Process in batches
    total_processed = 0
    batch_size = args.batch_size
    max_parallel = min(args.parallel, config.MAX_PARALLEL_PROCESSES)
    
    start_time = time.time()
    
    for i in range(0, len(unannotated_items), batch_size):
        # Check if we should shut down
        if is_shutting_down or check_kill_switch():
            logger.info("Shutdown requested. Stopping batch processing.")
            break
            
        batch = unannotated_items[i:i + batch_size]
        batch_start_time = time.time()
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(unannotated_items)+batch_size-1)//batch_size} ({len(batch)} items)")
        
        # Process the batch
        successful = process_batch(
            batch,
            data_processor,
            max_parallel
        )
        
        total_processed += successful
        batch_time = time.time() - batch_start_time
        
        logger.info(f"Batch completed: {successful}/{len(batch)} items processed successfully in {batch_time:.2f} seconds")
        logger.info(f"Total progress: {total_processed}/{len(unannotated_items)} items ({(total_processed/len(unannotated_items)*100):.2f}%)")
        
        # Save interim results
        data_processor.save_interim_annotations()
        
        # Short pause between batches to avoid API rate limits
        time.sleep(5)
    
    # Save final results
    if args.convert:
        # Load annotations
        with open(args.interim, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        
        # Convert to list format
        annotation_list = []
        for item_id, annotation in annotations.items():
            annotation_list.append(annotation)
        
        # Convert to MCAQAR format
        mcaqar_format = DataProcessor.convert_to_mcaqar_format(annotation_list)
        
        # Save to output file
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(mcaqar_format, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(mcaqar_format)} annotations in MCAQAR format to {args.output}")
    else:
        data_processor.save_final_results()
    
    total_time = time.time() - start_time
    
    logger.info(f"Annotation process completed in {total_time:.2f} seconds")
    logger.info(f"Processed {total_processed}/{len(unannotated_items)} items successfully")
    
    # Show statistics
    if args.stats:
        stats = data_processor.get_annotation_statistics()
        print("\n=== Annotation Statistics ===")
        print(f"Total items: {stats['total_items']}")
        print(f"Annotated items: {stats['annotated_items']}")
        print(f"Completion percentage: {stats['completion_percentage']:.2f}%")
        print(f"Average conditions per item: {stats['avg_conditions']:.2f}")
        print(f"Average condition citations: {stats['avg_condition_citations']:.2f}")
        print(f"Average answer citations: {stats['avg_answer_citations']:.2f}")
        print("\nConditions per item distribution:")
        for count, occurrences in sorted(stats['conditions_per_item'].items()):
            print(f"  {count} conditions: {occurrences} items ({(occurrences/stats['annotated_items']*100):.2f}%)")

if __name__ == "__main__":
    main()