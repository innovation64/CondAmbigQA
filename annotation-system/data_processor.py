"""
Data processing utilities for the AI-AI annotation system.
Handles loading, saving, and processing dataset files.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(config.DEFAULT_LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles dataset loading, saving, and processing"""
    
    def __init__(self, input_file: str = config.DEFAULT_INPUT_PATH, output_file: str = config.DEFAULT_OUTPUT_PATH, interim_file: str = config.DEFAULT_INTERIM_PATH):
        """
        Initialize the data processor
        
        Args:
            input_file: Path to the input dataset file
            output_file: Path to save the output dataset
            interim_file: Path to save/load interim results
        """
        self.input_file = input_file
        self.output_file = output_file
        self.interim_file = interim_file
        
        # Initialize data containers
        self.dataset = []
        self.annotations = {}
        
        # Load the dataset
        self._load_dataset()
        
        # Preprocess the dataset
        self.preprocess_dataset()
        
        # Load any existing annotations
        self._load_interim_annotations()
    
    def _load_dataset(self) -> bool:
        """
        Load the input dataset
        
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
            
            logger.info(f"Loaded {len(self.dataset)} items from {self.input_file}")
            return True
        except FileNotFoundError:
            logger.error(f"Input file {self.input_file} not found")
            return False
        except json.JSONDecodeError:
            logger.error(f"Input file {self.input_file} is not valid JSON")
            return False
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False
    
    def _load_interim_annotations(self) -> bool:
        """
        Load any existing interim annotations
        
        Returns:
            True if loading was successful or file didn't exist, False on error
        """
        try:
            if os.path.exists(self.interim_file):
                with open(self.interim_file, "r", encoding="utf-8") as f:
                    self.annotations = json.load(f)
                
                logger.info(f"Loaded {len(self.annotations)} interim annotations from {self.interim_file}")
            else:
                logger.info(f"No interim annotations file found at {self.interim_file}")
                self.annotations = {}
            
            return True
        except json.JSONDecodeError:
            logger.error(f"Interim file {self.interim_file} is not valid JSON")
            self.annotations = {}
            return False
        except Exception as e:
            logger.error(f"Error loading interim annotations: {str(e)}")
            self.annotations = {}
            return False
    
    def save_interim_annotations(self) -> bool:
        """
        Save interim annotations to file
        
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            with open(self.interim_file, "w", encoding="utf-8") as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(self.annotations)} interim annotations to {self.interim_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving interim annotations: {str(e)}")
            return False
    
    def save_final_results(self) -> bool:
        """
        Save final annotation results to output file
        
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            # Convert annotations dictionary to list format
            annotated_dataset = []
            
            for item in self.dataset:
                item_id = item.get("id", "")
                
                if item_id in self.annotations:
                    # Use the annotated version
                    annotated_dataset.append(self.annotations[item_id])
                else:
                    # Skip unannotated items
                    continue
            
            # Save to output file
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(annotated_dataset, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(annotated_dataset)} annotated items to {self.output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving final results: {str(e)}")
            return False
        
    def preprocess_dataset(self) -> None:
        """
        Preprocess the dataset to ensure it's in the format expected by the annotation system
        """
        preprocessed_dataset = []
        
        for item in self.dataset:
            preprocessed_item = {}
            
            # Extract ID
            if "id" in item:
                preprocessed_item["id"] = item["id"]
            
            # Extract question (handle the specific format in paste.txt)
            question = ""
            if "question" in item:
                question = item["question"]
            elif "data" in item and "ambiguous_question" in item["data"]:
                question = item["data"]["ambiguous_question"]
            
            preprocessed_item["question"] = question
            
            # Extract contexts (specifically for the retrieval_ctxs format in paste.txt)
            contexts = []
            if "retrieval_ctxs" in item:
                contexts = item["retrieval_ctxs"]
            elif "ctxs" in item:
                contexts = item["ctxs"]
            
            preprocessed_item["ctxs"] = contexts
            
            # Add the preprocessed item
            preprocessed_dataset.append(preprocessed_item)
        
        self.dataset = preprocessed_dataset
        logger.info(f"Preprocessed {len(preprocessed_dataset)} items for annotation")
    
    def add_annotation(self, annotation: Dict[str, Any]) -> bool:
        """
        Add an annotation to the collection
        
        Args:
            annotation: The annotation to add
            
        Returns:
            True if adding was successful, False otherwise
        """
        try:
            item_id = annotation.get("id", "")
            if not item_id:
                logger.error("Annotation missing 'id' field")
                return False
            
            # Ensure the question field is populated
            if not annotation.get("question", ""):
                original_item = self.get_item_by_id(item_id)
                if original_item:
                    # Try to get the question from multiple possible locations
                    question = original_item.get("question", "")
                    if not question and "data" in original_item and "ambiguous_question" in original_item["data"]:
                        question = original_item["data"]["ambiguous_question"]
                    
                    # Update the annotation with the question
                    annotation["question"] = question
                    logger.info(f"Updated annotation for item {item_id} with question: {question[:50]}...")
            
            # Add to annotations dictionary
            self.annotations[item_id] = annotation
            
            # Save interim results after each addition
            self.save_interim_annotations()
            
            return True
        except Exception as e:
            logger.error(f"Error adding annotation: {str(e)}")
            return False
    
    def get_unannotated_items(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get items that haven't been annotated yet
        
        Args:
            limit: Maximum number of items to return (None for all)
            
        Returns:
            List of unannotated items
        """
        unannotated = []
        
        for item in self.dataset:
            item_id = item.get("id", "")
            
            if item_id not in self.annotations:
                unannotated.append(item)
            
            if limit is not None and len(unannotated) >= limit:
                break
        
        return unannotated
    
    def get_item_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an item by its ID
        
        Args:
            item_id: The ID of the item to get
            
        Returns:
            The item if found, None otherwise
        """
        for item in self.dataset:
            if item.get("id", "") == item_id:
                return item
            # Handle nested structure where id might be inside an item
            elif isinstance(item, dict) and "id" in item and item["id"] == item_id:
                return item
        
        return None
    
    def get_annotation_statistics(self) -> Dict[str, Any]:
        """
        Generate statistics about the annotations
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_items": len(self.dataset),
            "annotated_items": len(self.annotations),
            "completion_percentage": (len(self.annotations) / len(self.dataset) * 100) if self.dataset else 0,
            "conditions_per_item": {},
            "avg_conditions": 0,
            "avg_condition_citations": 0,
            "avg_answer_citations": 0,
        }
        
        # Count conditions per item
        condition_counts = {}
        total_conditions = 0
        total_condition_citations = 0
        total_answer_citations = 0
        
        for item_id, annotation in self.annotations.items():
            properties = annotation.get("properties", [])
            num_conditions = len(properties)
            total_conditions += num_conditions
            
            # Update condition count
            condition_counts[num_conditions] = condition_counts.get(num_conditions, 0) + 1
            
            # Count citations
            for prop in properties:
                total_condition_citations += len(prop.get("condition_citations", []))
                total_answer_citations += len(prop.get("answer_citations", []))
        
        # Calculate averages
        stats["conditions_per_item"] = condition_counts
        stats["avg_conditions"] = total_conditions / len(self.annotations) if self.annotations else 0
        stats["avg_condition_citations"] = total_condition_citations / total_conditions if total_conditions else 0
        stats["avg_answer_citations"] = total_answer_citations / total_conditions if total_conditions else 0
        
        return stats

    @staticmethod
    def convert_to_mcaqar_format(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert annotations to the MCAQAR format
        
        Args:
            annotations: List of annotations with differentiated citations
            
        Returns:
            List of annotations in MCAQAR format
        """
        mcaqar_formatted = []
        
        for item in annotations:
            item_id = item.get("id", "")
            question = item.get("question", "")
            properties = item.get("properties", [])
            
            # Create the MCAQAR format item
            mcaqar_item = {
                "id": item_id,
                "question": question,
                "properties": []
            }
            
            # Process each property
            for prop in properties:
                condition = prop.get("condition", "")
                groundtruth = prop.get("groundtruth", "")
                
                # Combine condition and answer citations
                condition_citations = prop.get("condition_citations", [])
                answer_citations = prop.get("answer_citations", [])
                
                # Ensure no duplicate citations
                combined_citations = []
                seen_titles = set()
                
                for citation in condition_citations + answer_citations:
                    title = citation.get("title", "")
                    if title and title not in seen_titles:
                        combined_citations.append(citation)
                        seen_titles.add(title)
                
                # Add to MCAQAR properties
                mcaqar_item["properties"].append({
                    "condition": condition,
                    "groundtruth": groundtruth,
                    "citations": combined_citations
                })
            
            mcaqar_formatted.append(mcaqar_item)
        
        return mcaqar_formatted
    
    @staticmethod
    def convert_mcaqar_to_enhanced(mcaqar_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert MCAQAR format to enhanced_questions.json format
        
        Args:
            mcaqar_data: List of items in MCAQAR format
            
        Returns:
            List of items in enhanced_questions.json format
        """
        enhanced_questions = []
        
        for item in mcaqar_data:
            item_id = item.get("id", "")
            question = item.get("question", "")
            ctxs = item.get("ctxs", [])
            
            enhanced_item = {
                "id": item_id,
                "question": question,
                "ctxs": ctxs,
                "properties": []  # Will be populated by annotation process
            }
            
            enhanced_questions.append(enhanced_item)
        
        return enhanced_questions
    
    @staticmethod
    def split_dataset(dataset: List[Dict[str, Any]], num_parts: int) -> List[List[Dict[str, Any]]]:
        """
        Split a dataset into multiple parts
        
        Args:
            dataset: The dataset to split
            num_parts: Number of parts to split into
            
        Returns:
            List of dataset parts
        """
        parts = []
        items_per_part = len(dataset) // num_parts
        
        for i in range(num_parts):
            start_idx = i * items_per_part
            end_idx = (i + 1) * items_per_part if i < num_parts - 1 else len(dataset)
            parts.append(dataset[start_idx:end_idx])
        
        return parts