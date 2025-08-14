"""
Utility functions for the AI-AI annotation system.
"""

import os
import json
import re
import logging
import time
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

def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from text with improved handling
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
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
        
        # Try to parse the entire response as JSON as a last resort
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Failed to parse JSON from text: {str(e)}")
        return None

def extract_citations_from_fragments(fragments: List[Dict[str, str]], condition_text: str, answer_text: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Extract relevant citations from fragments for a condition and answer
    
    Args:
        fragments: List of fragments to extract citations from
        condition_text: The condition text
        answer_text: The answer text
        
    Returns:
        Dictionary with keys:
        - condition_citations: List of citations for the condition
        - answer_citations: List of citations for the answer
    """
    condition_keywords = extract_keywords(condition_text)
    answer_keywords = extract_keywords(answer_text)
    
    condition_citations = []
    answer_citations = []
    
    for i, fragment in enumerate(fragments):
        fragment_text = fragment.get("text", "")
        fragment_keywords = extract_keywords(fragment_text)
        
        # Calculate relevance to condition and answer
        condition_relevance = calculate_relevance(fragment_keywords, condition_keywords)
        answer_relevance = calculate_relevance(fragment_keywords, answer_keywords)
        
        # Add to citations if relevant
        if condition_relevance > 0.3:  # Threshold for relevance
            condition_citations.append({
                "fragment_number": i + 1,
                "fragment_title": fragment.get("title", ""),
                "fragment_text": fragment_text,
                "relevance": condition_relevance
            })
        
        if answer_relevance > 0.3:  # Threshold for relevance
            answer_citations.append({
                "fragment_number": i + 1,
                "fragment_title": fragment.get("title", ""),
                "fragment_text": fragment_text,
                "relevance": answer_relevance
            })
    
    # Sort by relevance and limit the number of citations
    condition_citations.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    answer_citations.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    
    # Limit to the most relevant citations
    condition_citations = condition_citations[:3]
    answer_citations = answer_citations[:3]
    
    # Remove the relevance score from the final citations
    for citation in condition_citations + answer_citations:
        if "relevance" in citation:
            del citation["relevance"]
    
    return {
        "condition_citations": condition_citations,
        "answer_citations": answer_citations
    }

def extract_keywords(text: str) -> List[str]:
    """
    Extract keywords from text
    
    Args:
        text: Text to extract keywords from
        
    Returns:
        List of keywords
    """
    # A simple implementation that removes stopwords
    # A real implementation would use more sophisticated NLP
    stopwords = [
        "the", "is", "in", "on", "at", "and", "or", "but", "if", "because", 
        "therefore", "a", "an", "of", "to", "for", "with", "by", "about", 
        "as", "that", "this", "these", "those", "it", "they", "them", "he", 
        "she", "his", "her", "their", "its", "our", "we", "us", "you", "your"
    ]
    
    # Convert to lowercase and split
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stopwords and short words
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    
    return keywords

def calculate_relevance(source_keywords: List[str], target_keywords: List[str]) -> float:
    """
    Calculate the relevance of source keywords to target keywords
    
    Args:
        source_keywords: Source keywords
        target_keywords: Target keywords
        
    Returns:
        Relevance score between 0 and 1
    """
    if not source_keywords or not target_keywords:
        return 0.0
    
    # Count overlapping keywords
    overlap = len(set(source_keywords) & set(target_keywords))
    
    # Calculate Jaccard similarity
    union = len(set(source_keywords) | set(target_keywords))
    
    return overlap / union if union > 0 else 0.0

def estimate_hierarchy_level(condition: str) -> int:
    """
    Estimate the hierarchy complexity of a condition
    
    Args:
        condition: The condition text
        
    Returns:
        Estimated hierarchy level (1, 2, or 3+)
    """
    # Check for hierarchy indicators
    level_indicators = ["if", "when", "while", "but", "unless", "otherwise", "also", "and", "or", "assuming"]
    
    # Count indicators
    indicator_count = sum(1 for indicator in level_indicators if indicator.lower() in condition.lower())
    
    # Check for complex sentence structures
    clauses = len([s for s in condition.split(",") if len(s) > 5])
    
    # Estimate hierarchy: base level + indicator bonus + complex sentence bonus
    return 1 + min(3, indicator_count) + (1 if clauses >= 3 else 0)

def format_conversation_for_display(conversation_history: List[Dict[str, str]]) -> str:
    """
    Format a conversation history for display
    
    Args:
        conversation_history: List of conversation turns
        
    Returns:
        Formatted conversation as a string
    """
    formatted = []
    
    for turn in conversation_history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        
        role_display = {
            "annotator": "👨‍🔬 Annotator",
            "reviewer": "👨‍🏫 Reviewer",
            "facilitator": "👨‍💼 Facilitator",
            "system": "🤖 System",
            "user": "👤 User"
        }.get(role, role.capitalize())
        
        formatted.append(f"## {role_display}\n\n{content}\n")
    
    return "\n".join(formatted)

def create_batch_from_dataset(dataset_path: str, output_path: str, size: int = 100, start_idx: int = 0) -> str:
    """
    Create a batch file from a larger dataset
    
    Args:
        dataset_path: Path to the dataset file
        output_path: Path to save the batch file
        size: Number of items to include in the batch
        start_idx: Starting index in the dataset
        
    Returns:
        Path to the created batch file
    """
    try:
        # Load the dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        # Extract the batch
        batch = dataset[start_idx:start_idx + size]
        
        # Save the batch
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created batch with {len(batch)} items from {start_idx} to {start_idx + len(batch) - 1}")
        return output_path
    except Exception as e:
        logger.error(f"Error creating batch: {str(e)}")
        return ""

def merge_annotation_files(file_paths: List[str], output_path: str) -> bool:
    """
    Merge multiple annotation files into one
    
    Args:
        file_paths: List of annotation file paths
        output_path: Path to save the merged file
        
    Returns:
        True if merging was successful, False otherwise
    """
    try:
        merged = {}
        total_items = 0
        
        # Load and merge each file
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            
            for item_id, annotation in annotations.items():
                merged[item_id] = annotation
                total_items += 1
        
        # Convert to list if needed
        if all(isinstance(key, str) for key in merged.keys()):
            # Dictionary format, save as is
            output_data = merged
        else:
            # Convert to list
            output_data = list(merged.values())
        
        # Save to output file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Merged {len(file_paths)} files with {total_items} total items into {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error merging annotation files: {str(e)}")
        return False