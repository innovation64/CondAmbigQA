import json
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("format_handler.log"),
        logging.StreamHandler(),
    ],
)

class FormatHandler:
    """Process different format data files and extract relevant metrics"""
    
    def __init__(self, output_dir="visualizations"):
        """Initialize handler"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store raw data
        self.data = {
            "ground_truth": {},
            "main": {},
            "compare": {}
        }
        
        # Store processed metrics at example level (new)
        self.example_metrics = {
            "ground_truth": {},
            "main": {},
            "compare": {}
        }
        
        # Store processed metrics at model level
        self.metrics = {
            "ground_truth": {},
            "main": {},
            "compare": {}
        }
        
        # Store expected answer counts for each example
        self.expected_answer_counts = {}
        
        # Model name standardization - aligned with EnhancedVisualizer
        self.model_mapping = {
            "gpt-4o": "gpt-4o",
            "glm-4-plus": "glm4-plus",
            "glm4": "glm4",
            "llama3.1": "llama3.1",  # Modified to match EnhancedVisualizer
            "mistral": "mistral", 
            "gemma2": "gemma2",
            "qwen2.5": "qwen2.5",
            "deepseekr1": "deepseek-r1"    # Modified to match EnhancedVisualizer
        }
        
        # Set highlight models - same as EnhancedVisualizer
        self.highlight_models = ['gpt-4o', 'glm4-plus']
        
        # Color scheme - aligned with EnhancedVisualizer
        self.colors = {
            "gpt-4o": "#1f77b4",      # Strong blue
            "qwen2.5": "#17becf",       # Teal
            "glm4-plus": "#9467bd",   # Purple
            "glm4": "#d62728",       # Red
            "llama3.1": "#ff7f0e",      # Orange
            "mistral": "#2ca02c",    # Green
            "gemma2": "#e377c2",
            "deepseek-r1": "#8c564b"      # Pink
        }
        
        # API models and Local models - same as EnhancedVisualizer
        self.api_models = ["gpt-4o", "glm4-plus"]
        self.local_models = ["llama3.1", "mistral", "qwen2.5", "gemma2", "glm4", "deepseek-r1"]
        
        # Set line styles - same as EnhancedVisualizer
        self.line_styles = {model: 'dashed' if model in self.api_models else 'solid' 
                          for model in self.colors.keys()}
        
        # Set line widths - thicker for highlighted models
        self.line_widths = {model: 3 if model in self.highlight_models else 2
                           for model in self.colors.keys()}
        
        # Set high-quality plotting style - same as EnhancedVisualizer
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'axes.labelpad': 10,
            'figure.dpi': 300,
        })

    def extract_model_name(self, filename: str) -> str:
        """Extract standardized model name from filename"""
        # Remove common prefixes/suffixes
        clean_name = filename.replace("results_", "").replace("_ground_truth", "").replace("_compare", "").replace("_main", "")
        
        # Check for standard model names - use exact mapping for consistency
        for model_key, std_name in self.model_mapping.items():
            if model_key in clean_name:
                return std_name
                
        # If no match found, return cleaned name
        logging.warning(f"Could not extract standard model name from filename: {filename}")
        return clean_name

    def load_json_file(self, file_path: Path) -> Any:
        """Safely load JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def extract_example_id(self, example: Dict) -> str:
        """Extract ID from example"""
        if "id" in example:
            return example["id"]
        elif "sample_id" in example:
            return example["sample_id"]
        else:
            # No clear ID, use question as ID
            return example.get("question", "unknown_id")
        
    def calculate_balanced_scores(self, 
                                condition_score: float, 
                                answer_score: float, 
                                citation_score: float,
                                answer_count_difference: int,
                                expected_answer_count: int,
                                citation_count: float,
                                model: str) -> Tuple[float, float, float]:
            """
            Calculate balanced scores considering answer quantity, answer quality, and citation quality.
            
            Args:
                condition_score: Original condition score
                answer_score: Original answer score
                citation_score: Original citation score
                answer_count_difference: Difference between actual and expected answers
                expected_answer_count: Number of expected answers
                citation_count: Average number of citations per answer
                model: Model name
                
            Returns:
                Tuple of balanced scores (condition, answer, citation)
            """
            # Constants definition, improving readability and making adjustments easier
            HIGH_QUALITY_THRESHOLD = 0.5
            MAX_CITATION_BOOST_COUNT = 3
            CITATION_BOOST_FACTOR = 0.3
            SINGLE_ANSWER_PENALTY = 0.9
            MISSING_ANSWER_MAX_PENALTY = 0.2
            EXTRA_ANSWER_MAX_PENALTY = 0.15
            
            # Model-specific parameter configuration
            model_config = {
                "default": {
                    "quality_weight": 0.6,      # Weight of quality factor
                    "quantity_weight": 0.4,     # Weight of quantity factor
                    "citation_weight": 0.5,     # Weight of citation factor
                    "missing_severity": 0.6,    # Severity of missing answers
                    "extra_severity": 0.3       # Severity of extra answers
                }
            }
            
            # Get configuration for current model, use default if no specific configuration exists
            config = model_config.get(model, model_config["default"])
            
            # Prevent division by zero error
            safe_expected_count = max(expected_answer_count, 1)
            
            # Calculate actual answer count
            actual_answer_count = safe_expected_count + answer_count_difference
            
            # 1. Quality factor calculation
            # Base quality score: weighted average of condition score and answer score
            base_quality = (condition_score + answer_score) / 2
            
            # Citation quality: boost citation score based on citation count
            citation_quality = citation_score * (1 + CITATION_BOOST_FACTOR * min(citation_count, MAX_CITATION_BOOST_COUNT))
            
            # Comprehensive quality score: combine base quality and citation quality
            quality_score = (base_quality * (1 - config["citation_weight"]) + 
                            citation_quality * config["citation_weight"])
            
            # 2. Quantity adjustment calculation
            quantity_adjustment = 1.0  # Default no adjustment
            
            if actual_answer_count == 1 and safe_expected_count > 1:
                # Special case for single answer
                quantity_adjustment = 1.0 - SINGLE_ANSWER_PENALTY
            elif answer_count_difference < 0:
                # Case for missing answers
                missing_ratio = abs(answer_count_difference) / safe_expected_count
                
                # Adjust missing answer penalty based on answer quality: higher quality answers receive lighter penalties
                quality_factor = max(0.5, min(1.0, answer_score / HIGH_QUALITY_THRESHOLD))
                penalty = min(missing_ratio * config["missing_severity"] * (2 - quality_factor), 
                            MISSING_ANSWER_MAX_PENALTY)
                
                quantity_adjustment = 1.0 - penalty
            elif answer_count_difference > 0:
                # Case for extra answers
                extra_ratio = answer_count_difference / safe_expected_count
                
                # Adjust extra answer penalty based on answer quality: higher quality extra answers receive lighter penalties
                quality_factor = max(0.5, min(1.0, answer_score / HIGH_QUALITY_THRESHOLD))
                
                if extra_ratio > 1.0:  # Significantly exceeds expectations
                    penalty = min(extra_ratio * config["extra_severity"], EXTRA_ANSWER_MAX_PENALTY)
                else:
                    penalty = min(extra_ratio * config["extra_severity"] * 0.5, EXTRA_ANSWER_MAX_PENALTY * 0.5)
                    
                # High-quality answers reduce the penalty for extra answers
                penalty = penalty * (2 - quality_factor)
                
                quantity_adjustment = 1.0 - penalty
            
            # 3. Calculate final balanced score
            # Weighted combination of quality and quantity factors
            quality_component = quality_score * config["quality_weight"]
            quantity_component = quantity_adjustment * config["quantity_weight"]
            
            # Combined scoring factor
            combined_factor = quality_component + quantity_component
            
            # Calculate final balanced scores
            balanced_condition_score = condition_score * combined_factor
            balanced_answer_score = answer_score * combined_factor
            balanced_citation_score = citation_quality * combined_factor
            
            # Ensure scores are within valid range [0, 1]
            balanced_condition_score = min(max(balanced_condition_score, 0), 1)
            balanced_answer_score = min(max(balanced_answer_score, 0), 1)
            balanced_citation_score = min(max(balanced_citation_score, 0), 1)
            
            # Log debug information
            logging.debug(
                f"Model {model}, Original: C={condition_score:.3f}, A={answer_score:.3f}, Cit={citation_score:.3f} | "
                f"Quality={quality_score:.3f}, Quantity={quantity_adjustment:.3f}, Combined={combined_factor:.3f} | "
                f"Balanced: C={balanced_condition_score:.3f}, A={balanced_answer_score:.3f}, Cit={balanced_citation_score:.3f} | "
                f"Diff={answer_count_difference}, CitCnt={citation_count:.2f}"
            )
            
            return balanced_condition_score, balanced_answer_score, balanced_citation_score

    
    def process_ground_truth_file(self, file_path: Path) -> Tuple[str, Dict, List]:
        """Process ground_truth format file and calculate metrics at example level"""
        model_name = self.extract_model_name(file_path.stem)
        data = self.load_json_file(file_path)
        if data is None:
            return model_name, {}, []
            
        examples = []
        
        # Handle different formats
        if isinstance(data, dict):
            # Check if format is {model_name: [...examples...]}
            for key in data.keys():
                if key in self.model_mapping.values() or key in self.model_mapping.keys():
                    if isinstance(data[key], list):
                        examples = data[key]
                        break
                    
            # Check if format is {example_id: example}
            if not examples and all(isinstance(key, str) and isinstance(data[key], dict) for key in data.keys()):
                examples = list(data.values())
                
        elif isinstance(data, list):
            # Already a list of examples
            examples = data
        
        # Initialize example-level metrics container for this model
        self.example_metrics["ground_truth"][model_name] = {}
        
        # Process each example individually
        for example in examples:
            example_id = self.extract_example_id(example)
            
            # Record expected answer count - from ground_truth conditions count
            if "conditions" in example:
                self.expected_answer_counts[example_id] = len(example["conditions"])
            
            # Extract scores for this example
            example_answer_scores = []
            example_citation_scores = []
            example_condition_scores = []
            
            # Check for evaluations
            if "evaluations" in example:
                for eval_item in example["evaluations"]:
                    # Process answer scores
                    if "answer_score" in eval_item:
                        example_answer_scores.append(eval_item["answer_score"])
                    elif isinstance(eval_item, dict) and "answer_evaluation" in eval_item:
                        if isinstance(eval_item["answer_evaluation"], dict) and "score" in eval_item["answer_evaluation"]:
                            example_answer_scores.append(eval_item["answer_evaluation"]["score"])
                    
                    # Process citation scores
                    if "citation_score" in eval_item:
                        example_citation_scores.append(eval_item["citation_score"])
                    elif isinstance(eval_item, dict) and "citation_evaluation" in eval_item:
                        if isinstance(eval_item["citation_evaluation"], dict) and "score" in eval_item["citation_evaluation"]:
                            example_citation_scores.append(eval_item["citation_evaluation"]["score"])
                    
                    # Process condition scores
                    if "condition_score" in eval_item:
                        example_condition_scores.append(eval_item["condition_score"])
                    elif isinstance(eval_item, dict) and "condition_evaluation" in eval_item:
                        if isinstance(eval_item["condition_evaluation"], dict) and "score" in eval_item["condition_evaluation"]:
                            example_condition_scores.append(eval_item["condition_evaluation"]["score"])
            
            # Calculate average scores for this example
            avg_answer_score = np.mean(example_answer_scores) if example_answer_scores else 0
            avg_citation_score = np.mean(example_citation_scores) if example_citation_scores else 0
            avg_condition_score = np.mean(example_condition_scores) if example_condition_scores else 0
            
            # Get actual and expected answer counts
            actual_answer_count = len(example.get("conditions", []))
            expected_answer_count = self.expected_answer_counts.get(example_id, actual_answer_count)
            answer_count_difference = actual_answer_count - expected_answer_count
            
            # Calculate citation count
            citation_count = 0
            if "conditions" in example:
                citation_counts = [len(condition.get("citations", [])) for condition in example["conditions"]]
                citation_count = np.mean(citation_counts) if citation_counts else 0
            
            # Calculate balanced scores for this example
            balanced_condition_score, balanced_answer_score, balanced_citation_score = self.calculate_balanced_scores(
                avg_condition_score, avg_answer_score, avg_citation_score,
                answer_count_difference, expected_answer_count, 
                citation_count, model_name
            )
            
            # Store example metrics
            self.example_metrics["ground_truth"][model_name][example_id] = {
                "answer_score": avg_answer_score,
                "citation_score": avg_citation_score,
                "condition_score": avg_condition_score,
                "balanced_answer_score": balanced_answer_score,
                "balanced_citation_score": balanced_citation_score,
                "balanced_condition_score": balanced_condition_score,
                "answer_count": actual_answer_count,
                "expected_answer_count": expected_answer_count,
                "answer_count_difference": answer_count_difference,
                "citation_count": citation_count
            }
        
        # Now calculate model-level metrics by averaging across examples
        model_metrics = self.aggregate_example_metrics("ground_truth", model_name)
        
        return model_name, model_metrics, examples

    def process_main_file(self, file_path: Path) -> Tuple[str, Dict, List]:
        """Process main experiment format file and calculate metrics at example level"""
        model_name = self.extract_model_name(file_path.stem)
        data = self.load_json_file(file_path)
        if data is None:
            return model_name, {}, []
            
        examples = []
        
        # Handle different formats
        if isinstance(data, dict):
            # Try to find model name key
            for key in data.keys():
                if key in self.model_mapping.values() or key in self.model_mapping.keys():
                    if isinstance(data[key], list):
                        examples = data[key]
                        break
            
            # If no model name key found, check if it's example ID format
            if not examples and len(data) > 0:
                # Check key structure - if long ID strings, might be example dictionary
                first_key = next(iter(data))
                if isinstance(first_key, str) and len(first_key) > 30:
                    examples = list(data.values())
        elif isinstance(data, list):
            # Already a list of examples
            examples = data
        
        # Initialize example-level metrics container for this model
        self.example_metrics["main"][model_name] = {}
        
        # Process each example individually
        for example in examples:
            example_id = self.extract_example_id(example)
            
            # Extract scores for this example
            example_answer_scores = []
            example_citation_scores = []
            example_condition_scores = []
            
            # Check for evaluations
            if "evaluations" in example:
                for eval_item in example["evaluations"]:
                    # Process answer scores
                    if "answer_score" in eval_item:
                        example_answer_scores.append(eval_item["answer_score"])
                    elif isinstance(eval_item, dict) and "answer_evaluation" in eval_item:
                        if isinstance(eval_item["answer_evaluation"], dict) and "score" in eval_item["answer_evaluation"]:
                            example_answer_scores.append(eval_item["answer_evaluation"]["score"])
                    
                    # Process citation scores
                    if "citation_score" in eval_item:
                        example_citation_scores.append(eval_item["citation_score"])
                    elif isinstance(eval_item, dict) and "citation_evaluation" in eval_item:
                        if isinstance(eval_item["citation_evaluation"], dict) and "score" in eval_item["citation_evaluation"]:
                            example_citation_scores.append(eval_item["citation_evaluation"]["score"])
                    
                    # Process condition scores
                    if "condition_score" in eval_item:
                        example_condition_scores.append(eval_item["condition_score"])
                    elif isinstance(eval_item, dict) and "condition_evaluation" in eval_item:
                        if isinstance(eval_item["condition_evaluation"], dict) and "score" in eval_item["condition_evaluation"]:
                            example_condition_scores.append(eval_item["condition_evaluation"]["score"])
            
            # Calculate average scores for this example
            avg_answer_score = np.mean(example_answer_scores) if example_answer_scores else 0
            avg_citation_score = np.mean(example_citation_scores) if example_citation_scores else 0
            avg_condition_score = np.mean(example_condition_scores) if example_condition_scores else 0
            
            # Get actual answer count
            actual_answer_count = len(example.get("conditions", []))
            
            # Get expected answer count from ground truth or the difference field
            expected_answer_count = None
            if "answer_count_difference" in example:
                answer_count_difference = example["answer_count_difference"]
                expected_answer_count = actual_answer_count - answer_count_difference
            else:
                # Try to get from self.expected_answer_counts
                expected_answer_count = self.expected_answer_counts.get(example_id, actual_answer_count)
                answer_count_difference = actual_answer_count - expected_answer_count
            
            # Calculate citation count
            citation_count = 0
            if "conditions" in example:
                citation_counts = [len(condition.get("citations", [])) for condition in example["conditions"]]
                citation_count = np.mean(citation_counts) if citation_counts else 0
            
            # Calculate balanced scores for this example
            balanced_condition_score, balanced_answer_score, balanced_citation_score = self.calculate_balanced_scores(
                avg_condition_score, avg_answer_score, avg_citation_score,
                answer_count_difference, expected_answer_count, 
                citation_count, model_name
            )
            
            # Store example metrics
            self.example_metrics["main"][model_name][example_id] = {
                "answer_score": avg_answer_score,
                "citation_score": avg_citation_score,
                "condition_score": avg_condition_score,
                "balanced_answer_score": balanced_answer_score,
                "balanced_citation_score": balanced_citation_score,
                "balanced_condition_score": balanced_condition_score,
                "answer_count": actual_answer_count,
                "expected_answer_count": expected_answer_count,
                "answer_count_difference": answer_count_difference,
                "citation_count": citation_count
            }
        
        # Now calculate model-level metrics by averaging across examples
        model_metrics = self.aggregate_example_metrics("main", model_name)
        
        return model_name, model_metrics, examples

    def process_compare_file(self, file_path: Path) -> Tuple[str, Dict, List]:
        """Process compare experiment format file and calculate metrics at example level"""
        model_name = self.extract_model_name(file_path.stem)
        data = self.load_json_file(file_path)
        if data is None:
            return model_name, {}, []
            
        examples = []
        
        # Handle different formats
        if isinstance(data, dict):
            # Try to find model name key
            for key in data.keys():
                if key in self.model_mapping.values() or key in self.model_mapping.keys():
                    if isinstance(data[key], list):
                        examples = data[key]
                        break
                        
            # If no model name key found, might be example ID format
            if not examples and len(data) > 0:
                first_key = next(iter(data))
                if isinstance(first_key, str) and len(first_key) > 30:
                    examples = list(data.values())
        elif isinstance(data, list):
            # Already a list of examples
            examples = data
        
        # Initialize example-level metrics container for this model
        self.example_metrics["compare"][model_name] = {}
        
        # Process each example individually
        for example in examples:
            example_id = self.extract_example_id(example)
            
            # Extract scores for this example
            example_answer_scores = []
            example_citation_scores = []
            
            # Check for evaluations
            if "evaluations" in example:
                for eval_item in example["evaluations"]:
                    # Process answer scores
                    if "answer_evaluation" in eval_item and isinstance(eval_item["answer_evaluation"], dict):
                        if "score" in eval_item["answer_evaluation"]:
                            example_answer_scores.append(eval_item["answer_evaluation"]["score"])
                    
                    # Process citation scores
                    if "citation_evaluation" in eval_item and isinstance(eval_item["citation_evaluation"], dict):
                        if "score" in eval_item["citation_evaluation"]:
                            example_citation_scores.append(eval_item["citation_evaluation"]["score"])
            
            # Calculate average scores for this example
            avg_answer_score = np.mean(example_answer_scores) if example_answer_scores else 0
            avg_citation_score = np.mean(example_citation_scores) if example_citation_scores else 0
            avg_condition_score = 0  # Usually not applicable for compare experiments
            
            # Get actual answer count
            actual_answer_count = 0
            if "generated_answers" in example:
                actual_answer_count = len(example["generated_answers"])
            elif "answers" in example:
                actual_answer_count = len(example["answers"])
            
            # Get expected answer count from ground truth or the difference field
            expected_answer_count = None
            if "answer_count_difference" in example:
                answer_count_difference = example["answer_count_difference"]
                expected_answer_count = actual_answer_count - answer_count_difference
            else:
                # Try to get from self.expected_answer_counts
                expected_answer_count = self.expected_answer_counts.get(example_id, actual_answer_count)
                answer_count_difference = actual_answer_count - expected_answer_count
            
            # Calculate citation count
            citation_count = 0
            if "citations" in example:
                citation_count = len(example["citations"])
            elif "answers" in example:
                citation_counts = [len(answer.get("citations", [])) for answer in example.get("answers", [])]
                citation_count = np.mean(citation_counts) if citation_counts else 0
            
            # Calculate balanced scores for this example
            balanced_condition_score, balanced_answer_score, balanced_citation_score = self.calculate_balanced_scores(
                avg_condition_score, avg_answer_score, avg_citation_score,
                answer_count_difference, expected_answer_count, 
                citation_count, model_name
            )
            
            # Store example metrics
            self.example_metrics["compare"][model_name][example_id] = {
                "answer_score": avg_answer_score,
                "citation_score": avg_citation_score,
                "condition_score": avg_condition_score,
                "balanced_answer_score": balanced_answer_score,
                "balanced_citation_score": balanced_citation_score,
                "balanced_condition_score": balanced_condition_score,
                "answer_count": actual_answer_count,
                "expected_answer_count": expected_answer_count,
                "answer_count_difference": answer_count_difference,
                "citation_count": citation_count
            }
        
        # Now calculate model-level metrics by averaging across examples
        model_metrics = self.aggregate_example_metrics("compare", model_name)
        
        return model_name, model_metrics, examples

    def aggregate_example_metrics(self, exp_type, model_name):
        """
        Aggregate example-level metrics into model-level metrics
        by averaging across all examples for a given model
        """
        examples = self.example_metrics[exp_type][model_name]
        if not examples:
            return {}
        
        # Initialize aggregation variables
        answer_scores = []
        citation_scores = []
        condition_scores = []
        balanced_answer_scores = []
        balanced_citation_scores = []
        balanced_condition_scores = []
        answer_counts = []
        answer_count_differences = []
        citation_counts = []
        
        # Collect metrics from all examples
        for example_id, metrics in examples.items():
            answer_scores.append(metrics["answer_score"])
            citation_scores.append(metrics["citation_score"])
            condition_scores.append(metrics["condition_score"])
            balanced_answer_scores.append(metrics["balanced_answer_score"])
            balanced_citation_scores.append(metrics["balanced_citation_score"])
            balanced_condition_scores.append(metrics["balanced_condition_score"])
            answer_counts.append(metrics["answer_count"])
            answer_count_differences.append(metrics["answer_count_difference"])
            citation_counts.append(metrics["citation_count"])
        
        # Calculate average metrics
        model_metrics = {
            "answer_score": np.mean(answer_scores),
            "citation_score": np.mean(citation_scores),
            "condition_score": np.mean(condition_scores),
            "balanced_answer_score": np.mean(balanced_answer_scores),
            "balanced_citation_score": np.mean(balanced_citation_scores),
            "balanced_condition_score": np.mean(balanced_condition_scores),
            "balanced_combined_score": np.mean([np.mean(balanced_condition_scores), 
                                               np.mean(balanced_answer_scores), 
                                               np.mean(balanced_citation_scores)]),
            "answer_count": np.mean(answer_counts),
            "answer_count_difference": np.mean(answer_count_differences),
            "citation_count": np.mean(citation_counts),
            "count": len(examples)
        }
        
        # Store these metrics
        self.metrics[exp_type][model_name] = model_metrics
        
        return model_metrics

    def load_data(self, data_dir="."):
        """Load all data files and extract metrics"""
        data_dir = Path(data_dir)
        
        # Process ground_truth files first to get expected answer counts
        ground_truth_files = list(data_dir.glob("*ground_truth*.json"))
        logging.info(f"Found {len(ground_truth_files)} ground_truth files")
        
        for file_path in ground_truth_files:
            model_name, metrics, examples = self.process_ground_truth_file(file_path)
            if model_name and examples:
                self.data["ground_truth"][model_name] = examples
                logging.info(f"Processed ground_truth file: {model_name}, example count: {len(examples)}")
            else:
                logging.warning(f"Could not process ground_truth file: {file_path}")
        
        # Process main files
        main_files = list(data_dir.glob("*_main*.json"))
        logging.info(f"Found {len(main_files)} main files")
        
        for file_path in main_files:
            model_name, metrics, examples = self.process_main_file(file_path)
            if model_name and examples:
                self.data["main"][model_name] = examples
                logging.info(f"Processed main file: {model_name}, example count: {len(examples)}")
            else:
                logging.warning(f"Could not process main file: {file_path}")
        
        # Process compare files
        compare_files = list(data_dir.glob("*compare*.json"))
        logging.info(f"Found {len(compare_files)} compare files")
        
        for file_path in compare_files:
            model_name, metrics, examples = self.process_compare_file(file_path)
            if model_name and examples:
                self.data["compare"][model_name] = examples
                logging.info(f"Processed compare file: {model_name}, example count: {len(examples)}")
            else:
                logging.warning(f"Could not process compare file: {file_path}")
        
        # Print expected answer count statistics
        logging.info(f"Collected expected answer counts for {len(self.expected_answer_counts)} examples")
        
        # Print model statistics
        for exp_type in self.metrics:
            for model, metrics in self.metrics[exp_type].items():
                logging.info(f"Model {model} in {exp_type} experiment:")
                logging.info(f"  Example count: {metrics.get('count', 0)}")
                logging.info(f"  Balanced answer score: {metrics.get('balanced_answer_score', 0):.4f}")
                logging.info(f"  Balanced citation score: {metrics.get('balanced_citation_score', 0):.4f}")
                logging.info(f"  Balanced condition score: {metrics.get('balanced_condition_score', 0):.4f}")
                logging.info(f"  Balanced combined score: {metrics.get('balanced_combined_score', 0):.4f}")
                logging.info(f"  Average answer count: {metrics.get('answer_count', 0):.2f}")
                logging.info(f"  Average answer count difference: {metrics.get('answer_count_difference', 0):.2f}")
        
        return self.metrics

    def create_comparison_visualization(self, balanced_metrics=None):
        """Create comparison visualization charts with the same styling as EnhancedVisualizer"""
        # If no balanced_metrics provided, use self.metrics
        if balanced_metrics is None:
            balanced_metrics = self.metrics
        
        # Get all available models
        all_models = set()
        for exp_type in balanced_metrics:
            all_models.update(balanced_metrics[exp_type].keys())
        
        # Sort models by balanced combined score from main experiment (descending)
        sorted_models = []
        model_scores = {}
        
        # Use main experiment's balanced scores for sorting
        for model in all_models:
            if model in balanced_metrics["main"]:
                score = balanced_metrics["main"][model].get("balanced_combined_score", 0)
                model_scores[model] = score
        
        # Sort by score
        sorted_models = [model for model, _ in sorted(model_scores.items(), key=lambda x: x[1], reverse=True)]
        
        # Create figure with 2 subplots (same layout as EnhancedVisualizer)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1.5, 1])
        
        # --- Top half: Balanced score chart ---
        ax1 = axes[0]
        
        # Bar width and positions
        width = 0.2  # Same as EnhancedVisualizer
        x = np.arange(len(sorted_models))
        
        # Get API vs. Local model masks for visual distinction
        api_models_mask = [model in self.api_models for model in sorted_models]
        
        # Create gradient color scheme - same as EnhancedVisualizer
        condition_colors = ['#1f77b4' if is_api else '#2ca02c' for is_api in api_models_mask]
        answer_colors = ['#5798c9' if is_api else '#5dba5d' for is_api in api_models_mask]
        citation_colors = ['#9bbade' if is_api else '#8ecc8e' for is_api in api_models_mask]
        
        # Highlight specific models with stronger colors
        for i, model in enumerate(sorted_models):
            if model in self.highlight_models:
                # Make highlighted models more saturated - same as EnhancedVisualizer
                condition_colors[i] = '#0a4a8f' if api_models_mask[i] else '#166616'
                answer_colors[i] = '#1a64a3' if api_models_mask[i] else '#237b23'
                citation_colors[i] = '#3b7eb8' if api_models_mask[i] else '#3a9c3a'
        
        # Extract score data
        condition_scores = [balanced_metrics["main"].get(model, {}).get("balanced_condition_score", 0) for model in sorted_models]
        answer_scores = [balanced_metrics["main"].get(model, {}).get("balanced_answer_score", 0) for model in sorted_models]
        citation_scores = [balanced_metrics["main"].get(model, {}).get("balanced_citation_score", 0) for model in sorted_models]
        combined_scores = [balanced_metrics["main"].get(model, {}).get("balanced_combined_score", 0) for model in sorted_models]
        
        # Plot bars with hatching patterns - same as EnhancedVisualizer
        condition_bars = ax1.bar(
            x - width, 
            condition_scores, 
            width, 
            label='Condition Score',
            color=condition_colors,
            edgecolor='black',
            hatch='//'
        )
        
        answer_bars = ax1.bar(
            x, 
            answer_scores, 
            width, 
            label='Answer Score',
            color=answer_colors,
            edgecolor='black',
            hatch='\\'
        )
        
        citation_bars = ax1.bar(
            x + width, 
            citation_scores, 
            width, 
            label='Citation Score',
            color=citation_colors,
            edgecolor='black',
            hatch='..'
        )
        
        # Add combined score line - same as EnhancedVisualizer
        ax1_twin = ax1.twinx()
        combined_line = ax1_twin.plot(
            x, 
            combined_scores,
            'o-',
            label='Combined Score',
            color='darkred',
            linewidth=2,
            markersize=8
        )
        
        # Add value labels for bars
        def add_labels(bars, scores):
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width()/2,
                    height,
                    f'{score:.2f}',
                    ha='center', 
                    va='bottom', 
                    fontsize=8,
                    rotation=0
                )
        
        add_labels(condition_bars, condition_scores)
        add_labels(answer_bars, answer_scores)
        add_labels(citation_bars, citation_scores)
        
        # Add labels for combined score line
        for i, score in enumerate(combined_scores):
            ax1_twin.text(
                x[i],
                score + 0.01,
                f'{score:.2f}',
                ha='center',
                va='bottom',
                fontsize=9,
                color='darkred',
                weight='bold'
            )
        
        # Configure axes
        ax1.set_ylim(0, 1)  # Balanced scores up to 1.0
        ax1_twin.set_ylim(0, 0.6)  # Combined score up to 0.6
        
        ax1.set_ylabel('Individual Scores')
        ax1_twin.set_ylabel('Combined Score', color='darkred')
        ax1_twin.tick_params(axis='y', colors='darkred')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([])  # Hide x-axis labels (shown in bottom plot)
        
        ax1.set_title('Balanced Performance Scores on CondAmbigQA')
        
        # Add legends - separate legends for bar chart and line
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax1_twin.get_legend_handles_labels()
        
        ax1.legend(
            handles1, 
            labels1, 
            loc='upper right',
            bbox_to_anchor=(1, 1)
        )
        
        ax1_twin.legend(
            handles2, 
            labels2, 
            loc='upper right',
            bbox_to_anchor=(1, 0.8)
        )
        
        # Add background distinction between API and Local models
        for i, is_api in enumerate(api_models_mask):
            x_start = i - 0.4
            x_end = i + 0.4
            if is_api:
                # Light blue background for API models
                ax1.axvspan(x_start, x_end, color='#e6f2ff', alpha=0.3, zorder=-1)
            else:
                # Light green background for Local models
                ax1.axvspan(x_start, x_end, color='#e6ffe6', alpha=0.3, zorder=-1)
        
        # --- Bottom half: Answer count statistics ---
        ax2 = axes[1]
        
        # Extract answer count data
        answer_counts = [balanced_metrics["main"].get(model, {}).get("answer_count", 0) for model in sorted_models]
        count_differences = [balanced_metrics["main"].get(model, {}).get("answer_count_difference", 0) for model in sorted_models]
        
        # Create color schemes for answer count - same as EnhancedVisualizer
        answer_count_colors = ['#aed6f1' if is_api else '#a3e4a3' for is_api in api_models_mask]
        diff_colors = []
        
        # Determine difference colors based on positive/negative and API/Local
        for i, diff in enumerate(count_differences):
            is_api = api_models_mask[i]
            if diff >= 0:
                # Positive difference - lighter color
                diff_colors.append('#5dade2' if is_api else '#58d258')
            else:
                # Negative difference - darker color
                diff_colors.append('#2874a6' if is_api else '#2e8b57')
        
        # Highlight specific models
        for i, model in enumerate(sorted_models):
            if model in self.highlight_models:
                answer_count_colors[i] = '#6baed6' if api_models_mask[i] else '#74c476'
        
        # Plot answer count
        width = 0.3  # Wider bars for lower plot
        count_bars = ax2.bar(
            x - width/2, 
            answer_counts, 
            width, 
            label='Answer Count',
            color=answer_count_colors,
            edgecolor='black',
            alpha=0.8
        )
        
        # Plot answer count difference
        diff_bars = ax2.bar(
            x + width/2, 
            count_differences, 
            width, 
            label='Count Difference',
            color=diff_colors,
            edgecolor='black',
            alpha=0.8
        )
        
        # Add zero line for difference
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        
        # Add value labels
        for bar in count_bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.05,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        for bar in diff_bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 0.05 if height >= 0 else -0.15
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                height + offset,
                f'{height:.2f}',
                ha='center',
                va=va,
                fontsize=9
            )
        
        # Configure axes
        ax2.set_ylabel('Count / Difference')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sorted_models, rotation=0, ha='center')
        ax2.set_ylim(-1.5, 4)
        ax2.legend(loc='upper right')
        ax2.set_title('Answer Count Statistics by Model')
        
        # Apply the same background distinction as top chart
        for i, is_api in enumerate(api_models_mask):
            x_start = i - 0.4
            x_end = i + 0.4
            if is_api:
                # Light blue background for API models
                ax2.axvspan(x_start, x_end, color='#e6f2ff', alpha=0.3, zorder=-1)
            else:
                # Light green background for Local models
                ax2.axvspan(x_start, x_end, color='#e6ffe6', alpha=0.3, zorder=-1)
        
        # Add explanation text
        plt.figtext(0.5, 0.01, 
                   "Note: Blue background indicates API models (GPT-4o, GLM4Plus), Green indicates Local models\n"
                   "Balanced scores account for both answer quality and consistency with expected count",
                   ha='center', fontsize=10, style='italic')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save to PDF and PNG
        plt.savefig(self.output_dir / 'comprehensive_performance.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'comprehensive_performance.pdf', bbox_inches='tight')
        
        logging.info("Saved comprehensive performance chart")
        return fig
    def create_experiment_comparison(self, balanced_metrics=None):
        """Create charts comparing different experiments with the same styling as EnhancedVisualizer"""
        # Remove default grid and use a clean white background
        sns.set_style('white')
        plt.rcParams['axes.grid'] = False

        # If no balanced_metrics provided, use self.metrics
        if balanced_metrics is None:
            balanced_metrics = self.metrics

        # Get all available models
        all_models = set()
        for exp_type in balanced_metrics:
            all_models.update(balanced_metrics[exp_type].keys())

        # Only select models that have data in all experiments
        common_models = [
            model for model in all_models
            if all(model in balanced_metrics[exp] for exp in balanced_metrics)
        ]
        if not common_models:
            logging.warning("No models found with data in all experiments")
            common_models = list(balanced_metrics["main"].keys())

        # Sort models by main experiment's balanced combined score (descending)
        model_scores = {
            model: balanced_metrics["main"].get(model, {}).get("balanced_combined_score", 0)
            for model in common_models
        }
        sorted_models = [
            model for model, _ in
            sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # Create figure with 1×2 subplot layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Experiment labels and colors
        experiment_labels = {
            "ground_truth": "Answering with Ground Truth Conditions",
            "main":         "Answering with Model-generated Conditions",
            "compare":      "Answering without Conditions"
        }
        experiment_colors = {
            "ground_truth": "#77c2a2",
            "main":         "#f79a76",
            "compare":      "#a4b6de"
        }

        # Bar width and positions
        width = 0.25
        x = np.arange(len(sorted_models))

        # --- Plot answer scores comparison ---
        answer_scores = {
            key: [
                balanced_metrics[key].get(model, {}).get("balanced_answer_score", 0)
                for model in sorted_models
            ]
            for key in ("ground_truth", "main", "compare")
        }

        bars_gt = ax1.bar(
            x - width, answer_scores["ground_truth"], width,
            label=experiment_labels["ground_truth"],
            color=experiment_colors["ground_truth"],
            edgecolor='black', hatch='//'
        )
        bars_main = ax1.bar(
            x, answer_scores["main"], width,
            label=experiment_labels["main"],
            color=experiment_colors["main"],
            edgecolor='black', hatch='\\'
        )
        bars_comp = ax1.bar(
            x + width, answer_scores["compare"], width,
            label=experiment_labels["compare"],
            color=experiment_colors["compare"],
            edgecolor='black', hatch='..'
        )

        # Add labels above bars
        for i in range(len(sorted_models)):
            for dx, scores in zip(
                (-width, 0, width),
                (answer_scores["ground_truth"], answer_scores["main"], answer_scores["compare"])
            ):
                ax1.text(
                    i + dx, scores[i] + 0.01,
                    f'{scores[i]:.2f}',
                    ha='center', va='bottom', fontsize=9
                )

        ax1.set_title('Answer Score Comparison', fontsize=16)
        ax1.set_ylabel('Average Answer Score', fontsize=14)
        ax1.set_ylim(0, 0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(sorted_models, rotation=0, ha='center', fontsize=12)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.grid(False)

        # --- Plot citation scores comparison ---
        citation_scores = {
            key: [
                balanced_metrics[key].get(model, {}).get("balanced_citation_score", 0)
                for model in sorted_models
            ]
            for key in ("ground_truth", "main", "compare")
        }

        ax2.bar(
            x - width, citation_scores["ground_truth"], width,
            label=experiment_labels["ground_truth"],
            color=experiment_colors["ground_truth"],
            edgecolor='black', hatch='//'
        )
        ax2.bar(
            x, citation_scores["main"], width,
            label=experiment_labels["main"],
            color=experiment_colors["main"],
            edgecolor='black', hatch='\\'
        )
        ax2.bar(
            x + width, citation_scores["compare"], width,
            label=experiment_labels["compare"],
            color=experiment_colors["compare"],
            edgecolor='black', hatch='..'
        )

        # Add labels above bars
        for i in range(len(sorted_models)):
            for dx, scores in zip(
                (-width, 0, width),
                (citation_scores["ground_truth"], citation_scores["main"], citation_scores["compare"])
            ):
                ax2.text(
                    i + dx, scores[i] + 0.01,
                    f'{scores[i]:.2f}',
                    ha='center', va='bottom', fontsize=9
                )

        ax2.set_title('Citation Score Comparison', fontsize=16)
        ax2.set_ylabel('Average Citation Score', fontsize=14)
        ax2.set_ylim(0, 1)
        ax2.set_xticks(x)
        ax2.set_xticklabels(sorted_models, rotation=0, ha='center', fontsize=12)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.grid(False)

        # Single legend below the two plots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1),
            ncol=3,
            frameon=False,
            fontsize=12
        )

        # Overall title and caption
        fig.suptitle(
            'Model Performance Across Different Experimental Conditions',
            fontsize=18, x=0.5, y=1.02
        )
        # caption = (
        #     "Figure: Model performance in Answer Score and Citation Score, comparing answering without conditions,\n"
        #     "answering based on identified conditions (Main Experiment), and answering based on ground truth conditions."
        # )
        # fig.text(0.5, 0.01, caption, ha='center', fontsize=12)

        # Layout adjustments
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(top=0.85, bottom=0.15)

        # Save figures
        fig_path_png = self.output_dir / 'experiment_comparison.png'
        fig_path_pdf = self.output_dir / 'experiment_comparison.pdf'
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, bbox_inches='tight')
        plt.close(fig)

        logging.info(f"Saved experiment comparison chart to {fig_path_png} and {fig_path_pdf}")
        return fig



    def create_horizontal_ranking_chart(self, balanced_metrics=None):
        """Create a horizontal bar chart showing model ranking - matches EnhancedVisualizer"""
        # If no balanced_metrics provided, use self.metrics
        if balanced_metrics is None:
            balanced_metrics = self.metrics
        
        # Get models from main experiment
        if "main" not in balanced_metrics or not balanced_metrics["main"]:
            logging.error("No data available for main experiment, cannot create ranking chart")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort models by combined score for ranking display (ascending for horizontal chart)
        all_models = list(balanced_metrics["main"].keys())
        sorted_data = [(model, balanced_metrics["main"][model].get("balanced_combined_score", 0)) 
                      for model in all_models]
        sorted_data.sort(key=lambda x: x[1])  # Ascending sort for horizontal chart
        
        models = [item[0] for item in sorted_data]
        combined_scores = [item[1] for item in sorted_data]
        y_pos = np.arange(len(models))
        
        # Get API vs. Local model masks for visual distinction
        api_models_mask = [model in self.api_models for model in models]
        
        # Create colors based on model type and highlight status
        bar_colors = []
        for i, model in enumerate(models):
            if model in self.highlight_models:
                # Highlighted models get more saturated colors
                bar_colors.append('#1a5fb4' if model in self.api_models else '#2b7827')
            else:
                # Standard colors for other models
                bar_colors.append('#6baed6' if model in self.api_models else '#74c476')
        
        # Plot horizontal bars
        bars = ax.barh(
            y_pos,
            combined_scores,
            height=0.8,
            color=bar_colors,
            edgecolor='black',
            alpha=0.8
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        # Add rank numbers
        for i, (model, score) in enumerate(zip(models, combined_scores)):
            rank = len(models) - i
            ax.text(
                0.01,
                y_pos[i],
                f'#{rank}',
                ha='left',
                va='center',
                color='white',
                fontweight='bold',
                fontsize=10
            )
        
        # Configure axes
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('Combined Score (Balanced)', fontsize=12)
        ax.set_title('Model Ranking Based on Balanced Combined Score', fontsize=14)
        
        # Add a vertical line for average score
        avg_score = np.mean(combined_scores)
        ax.axvline(x=avg_score, color='gray', linestyle='--', alpha=0.7)
        ax.text(
            avg_score + 0.01,
            -0.5,
            f'Average: {avg_score:.3f}',
            ha='left',
            va='center',
            color='gray',
            fontsize=10,
            fontstyle='italic'
        )
        
        # Add background shading for model types
        for i, is_api in enumerate(api_models_mask):
            y_start = y_pos[i] - 0.3
            y_end = y_pos[i] + 0.3
            if is_api:
                # Light blue background for API models
                ax.axhspan(y_start, y_end, color='#e6f2ff', alpha=0.3, zorder=-1)
            else:
                # Light green background for Local models
                ax.axhspan(y_start, y_end, color='#e6ffe6', alpha=0.3, zorder=-1)
        
        # Add legend for model types
        api_patch = plt.Rectangle((0, 0), 1, 1, fc='#e6f2ff', alpha=0.5, ec='gray')
        local_patch = plt.Rectangle((0, 0), 1, 1, fc='#e6ffe6', alpha=0.5, ec='gray')
        ax.legend([api_patch, local_patch], ['API Models', 'Local Models'], 
                 loc='lower right', framealpha=0.7)
        
        # Add explanation text
        plt.figtext(0.5, 0.01, 
                   "Combined Score = Average of Balanced Condition, Answer, and Citation Scores\n"
                   "Balanced scoring accounts for both quality and quantity of generated answers",
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save to PDF and PNG
        plt.savefig(self.output_dir / 'model_ranking.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'model_ranking.pdf', bbox_inches='tight')
        
        logging.info("Created model ranking chart")
        return fig
    
    def print_metrics_summary(self, balanced_metrics=None):
        """Print metrics summary"""
        # If no balanced_metrics provided, use self.metrics
        if balanced_metrics is None:
            balanced_metrics = self.metrics
            
        logging.info("=== Metrics Summary ===")
        
        # For each experiment type
        for exp_type in balanced_metrics:
            logging.info(f"\n--- {exp_type} Experiment ---")
            
            # For each model
            for model in balanced_metrics[exp_type]:
                metrics = balanced_metrics[exp_type][model]
                
                logging.info(f"Model: {model}")
                logging.info(f"  Example count: {metrics.get('count', 0)}")
                logging.info(f"  Raw answer score: {metrics.get('answer_score', 0):.4f}")
                logging.info(f"  Raw citation score: {metrics.get('citation_score', 0):.4f}")
                if "condition_score" in metrics:
                    logging.info(f"  Raw condition score: {metrics.get('condition_score', 0):.4f}")
                logging.info(f"  Balanced answer score: {metrics.get('balanced_answer_score', 0):.4f}")
                logging.info(f"  Balanced citation score: {metrics.get('balanced_citation_score', 0):.4f}")
                if "balanced_condition_score" in metrics:
                    logging.info(f"  Balanced condition score: {metrics.get('balanced_condition_score', 0):.4f}")
                logging.info(f"  Balanced combined score: {metrics.get('balanced_combined_score', 0):.4f}")
                if "answer_count" in metrics:
                    logging.info(f"  Answer count: {metrics.get('answer_count', 0):.2f}")
                if "answer_count_difference" in metrics:
                    logging.info(f"  Answer count difference: {metrics.get('answer_count_difference', 0):.2f}")
                if "citation_count" in metrics:
                    logging.info(f"  Citation count: {metrics.get('citation_count', 0):.2f}")
                logging.info("")
                
        # Create table comparing main experiment results (if available)
        if "main" in balanced_metrics and balanced_metrics["main"]:
            logging.info("\n=== Main Experiment Results Summary ===")
            # Sort models by balanced combined score
            sorted_models = sorted(
                balanced_metrics["main"].keys(),
                key=lambda m: balanced_metrics["main"][m].get("balanced_combined_score", 0),
                reverse=True
            )
            
            # Print header
            header = f"{'Model':<10} | {'Type':<5} | {'Cond.':<6} | {'Ans.':<6} | {'Cite.':<6} | {'Comb.':<6} | {'Count':<5} | {'Diff':<5} | {'Citations':<9}"
            logging.info(header)
            logging.info("-" * len(header))
            
            # Print each model's stats
            for model in sorted_models:
                metrics = balanced_metrics["main"][model]
                model_type = "API" if model in self.api_models else "Local"
                logging.info(
                    f"{model:<10} | "
                    f"{model_type:<5} | "
                    f"{metrics.get('balanced_condition_score', 0):.4f} | "
                    f"{metrics.get('balanced_answer_score', 0):.4f} | "
                    f"{metrics.get('balanced_citation_score', 0):.4f} | "
                    f"{metrics.get('balanced_combined_score', 0):.4f} | "
                    f"{metrics.get('answer_count', 0):.2f} | "
                    f"{metrics.get('answer_count_difference', 0):.2f} | "
                    f"{metrics.get('citation_count', 0):.2f}"
                )

    def run(self, data_dir="."):
        """Run complete processing and visualization pipeline"""
        # Load data
        self.load_data(data_dir)
        
        # Create performance comparison chart (same as EnhancedVisualizer's comprehensive_bar_chart)
        self.create_comparison_visualization()
        
        # Create horizontal ranking chart (same as EnhancedVisualizer)
        self.create_horizontal_ranking_chart()
        
        # Create experiment comparison chart
        self.create_experiment_comparison()
        
        # Print metrics summary
        self.print_metrics_summary()
        
        logging.info("Processing and visualization pipeline completed")
        
        # Return processed metrics
        return self.metrics

if __name__ == "__main__":
    # Create handler
    handler = FormatHandler(output_dir="visualizations")
    
    # Run processing and visualization pipeline
    metrics = handler.run(data_dir=".")
    
    # Print confirmation message
    print("\nVisualization process completed. Output files:")
    print("- visualizations/comprehensive_performance.pdf")
    print("- visualizations/comprehensive_performance.png")
    print("- visualizations/model_ranking.pdf")
    print("- visualizations/model_ranking.png")
    print("- visualizations/experiment_comparison.pdf")
    print("- visualizations/experiment_comparison.png")