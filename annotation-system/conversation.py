"""
Conversation management for the AI-AI annotation system.
This module handles the conversational flow between AI agents.
"""

import os
import json
import logging
import time
import signal
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import config
from models import AnnotatorAgent, ReviewerAgent, FacilitatorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(config.DEFAULT_LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class AnnotationConversation:
    """Manages the conversation flow between AI agents for annotation"""
    
    def __init__(
        self,
        item_id: str,
        question: str,
        fragments: List[Dict[str, str]],
        annotator: Optional[AnnotatorAgent] = None,
        reviewer: Optional[ReviewerAgent] = None,
        facilitator: Optional[FacilitatorAgent] = None
    ):
        """
        Initialize an annotation conversation
        
        Args:
            item_id: Unique identifier for the item being annotated
            question: The question to annotate
            fragments: Retrieved fragments for the question
            annotator: Annotator agent (will be created if None)
            reviewer: Reviewer agent (will be created if None)
            facilitator: Facilitator agent (will be created if None)
        """
        self.item_id = item_id
        self.question = question
        self.fragments = self._format_fragments(fragments)  # Format fragments to ensure consistent structure
        
        # Initialize AI agents if not provided
        self.annotator = annotator or AnnotatorAgent()
        self.reviewer = reviewer or ReviewerAgent()
        self.facilitator = facilitator or FacilitatorAgent()
        
        # Initialize conversation state
        self.conversation_history = []
        self.current_round = 0
        self.annotations_history = []
        self.reviews_history = []
        self.final_annotation = None
        self.final_properties = None
        
        # Initialize counters
        self.started_at = datetime.now()
        self.completed_at = None
        
        # Flags for controlling execution
        self.is_shutting_down = False
        self.timeout_occurred = False
        
        # Heartbeat timer to detect stalls
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 60  # seconds
        self._setup_heartbeat_monitor()
    
    def _setup_heartbeat_monitor(self):
        """Setup a thread to monitor heartbeats and detect stalls"""
        self.heartbeat_thread = threading.Thread(target=self._monitor_heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
    
    def _monitor_heartbeat(self):
        """Monitor heartbeats to detect stalled conversations"""
        while not self.is_shutting_down and not self.timeout_occurred:
            time.sleep(10)  # Check every 10 seconds
            
            current_time = time.time()
            time_since_heartbeat = current_time - self.last_heartbeat
            
            # If no heartbeat for 5 minutes, log warning
            if time_since_heartbeat > 300:
                logger.warning(f"No heartbeat for {time_since_heartbeat:.1f} seconds in conversation for item {self.item_id}")
            
            # If no heartbeat for 10 minutes, consider it a timeout
            if time_since_heartbeat > 600:
                logger.error(f"Heartbeat timeout detected for item {self.item_id} after {time_since_heartbeat:.1f} seconds")
                self.timeout_occurred = True
                break
    
    def update_heartbeat(self):
        """Update the heartbeat timestamp"""
        self.last_heartbeat = time.time()
        
    def _add_to_history(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        # Update heartbeat
        self.update_heartbeat()
    
    def _parse_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from text with improved handling"""
        try:
            import json
            import re
            
            # Try to find JSON between triple backticks
            json_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Try to find JSON enclosed by curly braces if no backticks found
            if not json_match:
                json_pattern = r'(\{[\s\S]*\})'
                json_match = re.search(json_pattern, text)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        return json.loads(json_str)
                    except:
                        pass
            
            # Try to parse the entire text
            return json.loads(text)
        except Exception as e:
            logger.warning(f"Failed to parse JSON from text for item {self.item_id}: {str(e)}")
            return None
    
    # Following are modifications to the _extract_final_properties method in AnnotationConversation class

    def _extract_final_properties(self, annotation_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract final properties from the annotation JSON with support for comprehensive format
        
        Args:
            annotation_json: The annotation in JSON format
            
        Returns:
            List of properties in the format required for the output
        """
        properties = []
        
        # Extract conditions from the annotation
        conditions = annotation_json.get("conditions", [])
        if not conditions and "final_conditions" in annotation_json:
            conditions = annotation_json.get("final_conditions", [])
        
        for condition in conditions:
            # Extract the basic properties
            condition_text = condition.get("condition", "")
            groundtruth = condition.get("ground_truth", "")
            
            # Extract citations
            condition_citations = condition.get("condition_citations", [])
            answer_citations = condition.get("answer_citations", [])
            
            # Format citations
            formatted_condition_citations = [
                {
                    "title": citation.get("fragment_title", ""),
                    "text": citation.get("fragment_text", "")
                }
                for citation in condition_citations
            ]
            
            formatted_answer_citations = [
                {
                    "title": citation.get("fragment_title", ""),
                    "text": citation.get("fragment_text", "")
                }
                for citation in answer_citations
            ]
            
            # Create property entry
            property_entry = {
                "condition": condition_text,
                "groundtruth": groundtruth,
                "condition_citations": formatted_condition_citations,
                "answer_citations": formatted_answer_citations
            }
            
            properties.append(property_entry)
        
        return properties
    
    def run_conversation(self, max_rounds: int = config.MAX_ROUNDS) -> Dict[str, Any]:
        """
        Run the full annotation conversation (synchronous version)
        
        Args:
            max_rounds: Maximum number of conversation rounds
            
        Returns:
            The final annotation result
        """
        logger.info(f"Starting annotation conversation for item {self.item_id}")
        
        # Add timeout handling
        start_time = time.time()
        max_execution_time = getattr(config, "MAX_EXECUTION_TIME", 300)  # 5 minutes timeout default
        
        try:
            # Check for timeout in periodic intervals
            def check_timeout():
                if time.time() - start_time > max_execution_time:
                    logger.warning(f"Timeout reached for item {self.item_id} after {self.current_round} rounds")
                    self.timeout_occurred = True
                    return True
                return False
            
            # Update heartbeat
            self.update_heartbeat()
            
            # Round 0: Facilitator initiates the process
            if check_timeout() or self.timeout_occurred:
                return self._create_timeout_result()
                
            facilitator_initiation = self.facilitator.initiate_annotation(self.question, self.fragments)
            self._add_to_history("facilitator", facilitator_initiation)
            logger.info(f"Facilitator initiated annotation for item {self.item_id}")
            
            # Round 1: Initial annotation
            if check_timeout() or self.timeout_occurred:
                return self._create_timeout_result()
                
            self.current_round = 1
            initial_annotation = self.annotator.generate_initial_annotations(
                self.question, self.fragments
            )
            self._add_to_history("annotator", initial_annotation)
            self.annotations_history.append(initial_annotation)
            logger.info(f"Completed round {self.current_round} - Initial annotation")
            
            initial_annotation_json = self._parse_json_from_text(initial_annotation)
            if not initial_annotation_json:
                logger.error(f"Failed to parse initial annotation for item {self.item_id}")
                # Create a minimal valid structure to allow the process to continue
                initial_annotation_json = {"conditions": []}
            
            # Continue with review and refinement rounds
            continue_annotation = True
            while continue_annotation and self.current_round < max_rounds:
                # Check for timeout
                if check_timeout() or self.timeout_occurred:
                    return self._create_timeout_result()
                
                # Review the current annotation
                review = self.reviewer.evaluate_annotation(
                    self.question, 
                    self.fragments,
                    self.annotations_history[-1],
                    self.current_round
                )
                self._add_to_history("reviewer", review)
                self.reviews_history.append(review)
                logger.info(f"Completed review for round {self.current_round}")
                
                # Check for timeout
                if check_timeout() or self.timeout_occurred:
                    return self._create_timeout_result()
                
                # Parse the review
                review_json = self._parse_json_from_text(review)
                if not review_json:
                    logger.warning(f"Failed to parse review for round {self.current_round}")
                    review_json = {
                        "overall_score": 0.0,
                        "annotation_approved": False,
                        "needs_another_round": self.current_round < config.MIN_ROUNDS
                    }
                
                # Check if we should continue
                if ((review_json.get("annotation_approved", False) and self.current_round >= config.MIN_ROUNDS) 
                    or self.current_round >= max_rounds):
                    # Annotation is approved or we've reached the maximum number of rounds
                    continue_annotation = False
                    logger.info(f"Annotation process complete after {self.current_round} rounds")
                    
                    # Use the last annotation as the final one
                    self.final_annotation = self.annotations_history[-1]
                    final_json = self._parse_json_from_text(self.final_annotation)
                    if final_json:
                        self.final_properties = self._extract_final_properties(final_json)
                    
                    break
                
                # Check for timeout
                if check_timeout() or self.timeout_occurred:
                    return self._create_timeout_result()
                
                # Get decision from facilitator
                decision = self.facilitator.decide_if_continue(
                    self.question,
                    self.current_round,
                    review
                )
                self._add_to_history("facilitator", json.dumps(decision))
                
                if not decision.get("continue", True) and self.current_round >= config.MIN_ROUNDS:
                    # Facilitator decided to stop
                    continue_annotation = False
                    logger.info(f"Facilitator decided to stop after round {self.current_round}: {decision.get('reason', '')}")
                    
                    # Use the last annotation as the final one
                    self.final_annotation = self.annotations_history[-1]
                    final_json = self._parse_json_from_text(self.final_annotation)
                    if final_json:
                        self.final_properties = self._extract_final_properties(final_json)
                    
                    break
                
                # Check for timeout
                if check_timeout() or self.timeout_occurred:
                    return self._create_timeout_result()
                
                # Move to the next round
                self.current_round += 1
                
                # Refine the annotation
                refined_annotation = self.annotator.refine_annotations(
                    self.question,
                    self.fragments,
                    self.annotations_history[-1],
                    review,
                    self.current_round
                )
                self._add_to_history("annotator", refined_annotation)
                self.annotations_history.append(refined_annotation)
                logger.info(f"Completed round {self.current_round} - Refined annotation")
            
            # Check for timeout
            if check_timeout() or self.timeout_occurred:
                return self._create_timeout_result()
            
            # Final summary by facilitator
            if not self.final_annotation:
                self.final_annotation = self.annotations_history[-1]
                final_json = self._parse_json_from_text(self.final_annotation)
                if final_json:
                    self.final_properties = self._extract_final_properties(final_json)
            
            final_summary = self.facilitator.summarize_conversation(
                self.question,
                self.conversation_history,
                self.final_annotation
            )
            self._add_to_history("facilitator", final_summary)
            logger.info(f"Facilitator provided final summary for item {self.item_id}")
            
            # Mark as completed
            self.completed_at = datetime.now()
            elapsed_time = (self.completed_at - self.started_at).total_seconds()
            logger.info(f"Annotation process completed in {elapsed_time:.2f} seconds after {self.current_round} rounds")
            
            # Save conversation history
            self._save_conversation_history()
            
            # Indicate we're shutting down properly
            self.is_shutting_down = True
            
            # Return the final result
            return {
                "id": self.item_id,
                "question": self.question,
                "properties": self.final_properties or [],
                "metadata": {
                    "rounds": self.current_round,
                    "duration_seconds": elapsed_time,
                    "conversation_length": len(self.conversation_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in conversation for item {self.item_id}: {str(e)}", exc_info=True)
            # Return a minimal valid result
            return {
                "id": self.item_id,
                "question": self.question,
                "properties": self.final_properties or [],
                "error": str(e)
            }
        finally:
            # Ensure we're marked as shutting down
            self.is_shutting_down = True
    
    def _create_timeout_result(self) -> Dict[str, Any]:
        """Create a result object for timeout scenarios"""
        # Use any partial results if available
        if self.annotations_history and not self.final_properties:
            try:
                latest_annotation = self.annotations_history[-1]
                latest_json = self._parse_json_from_text(latest_annotation)
                if latest_json:
                    self.final_properties = self._extract_final_properties(latest_json)
            except Exception as e:
                logger.error(f"Error extracting properties from latest annotation: {str(e)}")
        
        elapsed_time = (datetime.now() - self.started_at).total_seconds()
        logger.warning(f"Returning timeout result for item {self.item_id} after {elapsed_time:.2f} seconds")
        
        # Save what we have so far
        self._save_conversation_history(is_timeout=True)
        
        return {
            "id": self.item_id,
            "question": self.question,
            "properties": self.final_properties or [],
            "metadata": {
                "rounds": self.current_round,
                "duration_seconds": elapsed_time,
                "conversation_length": len(self.conversation_history),
                "timeout": True
            },
            "error": f"Timeout occurred after {elapsed_time:.2f} seconds during round {self.current_round}"
        }
    
    def _save_conversation_history(self, is_timeout=False):
        """Save the conversation history to a file"""
        try:
            # Create a filename with item ID and timestamp
            status = "timeout" if is_timeout else "complete"
            filename = f"{self.item_id}_{int(time.time())}_{status}_conversation.json"
            filepath = os.path.join(config.CONVERSATION_LOG_DIR, filename)
            
            # Ensure directory exists
            os.makedirs(config.CONVERSATION_LOG_DIR, exist_ok=True)
            
            # Create the conversation log
            conversation_log = {
                "item_id": self.item_id,
                "question": self.question,
                "started_at": self.started_at.isoformat(),
                "completed_at": datetime.now().isoformat(),
                "rounds": self.current_round,
                "timeout": is_timeout,
                "conversation": self.conversation_history,
                "final_properties": self.final_properties
            }
            
            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(conversation_log, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved conversation history for item {self.item_id} to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save conversation history for item {self.item_id}: {str(e)}")
    
    def _format_fragments(self, fragments: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format fragments to ensure they have the expected structure
        
        Args:
            fragments: The fragments to format
            
        Returns:
            Formatted fragments
        """
        formatted_fragments = []
        
        for i, fragment in enumerate(fragments):
            # Check if this is a retrieval_ctxs format
            if "title" in fragment and "text" in fragment and "score" in fragment:
                formatted_fragments.append({
                    "title": fragment.get("title", ""),
                    "text": fragment.get("text", "")
                })
            # Check if this is already in the expected format
            elif "title" in fragment and "text" in fragment:
                formatted_fragments.append(fragment)
            # Try to handle other formats
            else:
                title = fragment.get("title", "") or fragment.get("fragment_title", "") or f"Fragment {i+1}"
                text = fragment.get("text", "") or fragment.get("fragment_text", "") or fragment.get("content", "")
                
                formatted_fragments.append({
                    "title": title,
                    "text": text
                })
        
        return formatted_fragments