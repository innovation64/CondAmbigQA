import json
import logging
import os
from typing import Dict, List, Set, Tuple
import numpy as np
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import custom modules
from models import get_models
from utils import construct_prompt
from evaluator import (
    evaluate_condition_correctness,
    evaluate_answer_correctness,
    evaluate_citation_correctness,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("main_experiment.log"),
        logging.StreamHandler(),
    ],
)

# Initialize OpenAI client
client = OpenAI()

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

def get_processed_ids(filepath: str, model_name: str) -> Set[str]:
    """Get set of processed example IDs"""
    processed_ids = set()
    if os.path.exists(filepath):
        try:
            results = load_dataset(filepath)
            if isinstance(results, dict) and model_name in results:
                processed_ids.update(
                    res["id"] for res in results[model_name] 
                    if isinstance(res, dict) and "id" in res
                )
                logging.info(f"Loaded {len(processed_ids)} processed example IDs")
        except Exception as e:
            logging.error(f"Error loading processed IDs: {str(e)}")
    return processed_ids

@retry(
    retry=retry_if_exception_type((APIConnectionError, APIError, RateLimitError)),
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True,
)
def safe_generate_response(model, prompt: str) -> Dict:
    """Safely generate model response with retry mechanism"""
    try:
        return model.generate_response(prompt)
    except (APIConnectionError, APIError, RateLimitError) as e:
        logging.error(f"API error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return {}

def process_single_example(
    example: Dict,
    model,
    fragment_mapping: Dict,
    stats: Dict
) -> Tuple[Dict, bool]:
    """Process a single example"""
    try:
        question = example["question"]
        ctxs = example["ctxs"]
        properties = example.get("properties", [])
        example_id = example["id"]

        prompt = construct_prompt(question, ctxs)
        response = safe_generate_response(model, prompt)

        if not response:
            logging.warning(f"Model {model.name} gave no response for example {example_id}")
            return None, False

        actual_conditions = response.get("conditions", [])

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
                input_text=prompt,
                actual_condition=actual.get("condition", ""),
                expected_condition=expected.get("condition", "")
            )

            # Evaluate answer correctness
            ans_eval = evaluate_answer_correctness(
                input_text=prompt,
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

        result_entry = {
            "id": example_id,
            "question": question,
            "conditions": actual_conditions,
            "evaluations": evaluations,
            "answer_count_difference": len(actual_conditions) - len(properties)
        }

        return result_entry, True

    except Exception as e:
        logging.error(f"Error processing example: {str(e)}")
        return None, False

def run_main_experiment(dataset: List[Dict], models: List) -> Dict:
    """Run main experiment"""
    stats = {model.name: {
        "total_examples": 0,
        "total_expected_answers": 0,
        "total_generated_answers": 0,
        "answer_count_difference": 0
    } for model in models}

    for model in models:
        intermediate_filepath = f"intermediate_results_{model.name}_main_experiment.json"
        
        # Load or initialize results
        try:
            results = load_dataset(intermediate_filepath)
            if not isinstance(results, dict) or model.name not in results:
                results = {model.name: []}
        except FileNotFoundError:
            results = {model.name: []}

        model_results = results[model.name]
        processed_ids = get_processed_ids(intermediate_filepath, model.name)

        # Process each example
        for example in tqdm(dataset, desc=f"Processing examples for {model.name}"):
            if example["id"] in processed_ids:
                continue

            fragment_mapping = {f"Fragment {i+1}": ctx for i, ctx in enumerate(example["ctxs"])}
            result_entry, success = process_single_example(
                example=example,
                model=model,
                fragment_mapping=fragment_mapping,
                stats=stats
            )

            if success and result_entry:
                model_results.append(result_entry)
                results[model.name] = model_results
                save_dataset(results, intermediate_filepath)
                logging.info(f"Saved intermediate results for model {model.name}, example ID: {example['id']}")

    return stats

def compute_experiment_stats(models: List) -> Dict:
    """Compute experiment statistics"""
    stats = {}
    
    for model in models:
        main_filepath = f"intermediate_results_{model.name}_main_experiment.json"
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

        for example in main_results:
            if not isinstance(example, dict):
                logging.error(f"Invalid example format, expected dict but got {type(example)}")
                continue
            
            stats[model.name]["total_examples"] += 1
            stats[model.name]["total_expected_answers"] += len(example.get("conditions", []))
            stats[model.name]["total_generated_answers"] += len(example.get("conditions", []))
            stats[model.name]["answer_count_difference"] += example.get("answer_count_difference", 0)

    return stats

def main():
    """Main function"""
    # Load dataset
    dataset = load_dataset("mcaqa.json")
    if not dataset:
        logging.error("Failed to load dataset")
        return

    # Load models
    models = get_models()
    if not models:
        logging.error("No models loaded")
        return

    # Run main experiment
    print("Running main experiment...")
    stats = run_main_experiment(dataset, models)

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

    print("\nExperiment completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error running experiment: {str(e)}")
        raise