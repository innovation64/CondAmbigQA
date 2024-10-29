import json
import logging
import os
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

# Import custom modules
from evaluator import (
    evaluate_condition_correctness,
    evaluate_answer_correctness,
    evaluate_citation_correctness,
)
from utils import load_dataset  # 假设 utils 中有 load_dataset 函数

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler(),
    ],
)

def load_dataset(filepath: str) -> list:
    """Load JSON format dataset"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File {filepath} not found.")
        return []
    except json.JSONDecodeError:
        logging.error(f"File {filepath} is not valid JSON format.")
        return []

def save_dataset(data: dict, filepath: str) -> None:
    """Save dataset to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def compute_experiment_stats(models: List) -> Dict:
    """Compute experiment statistics"""
    stats = {}
    
    for model in models:
        main_filepath = f"intermediate_results_{model.name}_experiment.json"
        main_result = load_dataset(main_filepath)
        
        if not isinstance(main_result, dict) or model.name not in main_result:
            logging.error(f"Invalid result format or missing data for model {model.name}")
            continue
            
        main_results = main_result[model.name]
        
        if not isinstance(main_results, list):
            logging.error(f"Invalid main_results format, expected list but got {type(main_results)}")
            continue

        stats[model.name] = {
            "total_examples": 0,
            "total_expected_answers": 0,
            "total_generated_answers": 0,
            "answer_count_difference": 0
        }

        for example in tqdm(main_results, desc=f"Evaluating results for {model.name}"):
            if not isinstance(example, dict):
                logging.error(f"Invalid example format, expected dict but got {type(example)}")
                continue
            
            response = example.get("response", {})
            question = example.get("question", "")
            fragment_mapping = example.get("fragment_mapping", {})
            
            actual_conditions = response.get("conditions", [])
            properties = response.get("properties", [])  # 假设 properties 存在于 response 中

            # Update statistics
            stats[model.name]["total_examples"] += 1
            stats[model.name]["total_expected_answers"] += len(properties)
            stats[model.name]["total_generated_answers"] += len(actual_conditions)
            stats[model.name]["answer_count_difference"] += len(actual_conditions) - len(properties)

            evaluations = []
            for idx, actual in enumerate(actual_conditions):
                expected = properties[idx] if idx < len(properties) else {}
                
                # Evaluate condition correctness
                cond_eval = evaluate_condition_correctness(
                    input_text=question,
                    actual_condition=actual.get("condition", ""),
                    expected_condition=expected.get("condition", "")
                )

                # Evaluate answer correctness
                ans_eval = evaluate_answer_correctness(
                    input_text=question,
                    actual_answer=actual.get("answer", ""),
                    expected_answer=expected.get("groundtruth", "")
                )

                # Evaluate citation correctness
                expected_citations = [citation["title"] for citation in expected.get("citations", [])]
                actual_citations = actual.get("citations", [])

                cit_eval = evaluate_citation_correctness(
                    actual_output=actual_citations,
                    expected_output=expected_citations,
                    fragment_mapping=fragment_mapping
                )

                evaluations.append({
                    "condition_evaluation": cond_eval,
                    "answer_evaluation": ans_eval,
                    "citation_evaluation": cit_eval
                })

            example["evaluations"] = evaluations
            example["answer_count_difference"] = len(actual_conditions) - len(properties)

        # Save evaluated results if needed
        evaluated_filepath = f"evaluated_results_{model.name}.json"
        save_dataset(main_result, evaluated_filepath)
        logging.info(f"Saved evaluated results to {evaluated_filepath}")

    return stats

def main():
    """Main function to evaluate results"""
    # Load models
    from models import get_models
    models = get_models()
    if not models:
        logging.error("No models loaded")
        return

    # Compute statistics
    logging.info("Computing experiment statistics...")
    stats = compute_experiment_stats(models)

    # Print statistics
    print("\n=== Experiment Statistics ===")
    for model_name, model_stats in stats.items():
        print(f"\nModel: {model_name}")
        print(f"Total examples: {model_stats['total_examples']}")
        print(f"Total expected answers: {model_stats['total_expected_answers']}")
        print(f"Total generated answers: {model_stats['total_generated_answers']}")
        print(f"Answer count difference: {model_stats['answer_count_difference']}")
        
        avg_difference = (
            model_stats["answer_count_difference"] / model_stats["total_examples"]
            if model_stats["total_examples"] > 0 else 0
        )
        print(f"Average answer count difference: {avg_difference:.2f}")

    # Save statistics
    with open("experiment_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("\nSaved experiment statistics to experiment_statistics.json")

    logging.info("Evaluation completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise