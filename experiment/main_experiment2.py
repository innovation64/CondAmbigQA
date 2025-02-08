import json
import logging
import os
from models import get_models
from utils import (
    construct_prompt,
    construct_prompt_no_condition,
)
from evaluator import (
    evaluate_condition_correctness,
    evaluate_answer_correctness,
    evaluate_citation_correctness,
)
from tqdm import tqdm
import numpy as np
from openai import OpenAI
from openai import APIConnectionError, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler(),
    ],
)

# Initialize the OpenAI client
client = OpenAI()

def load_dataset(filepath: str) -> list:
    """Load dataset in JSON format"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File {filepath} not found.")
        return []
    except json.JSONDecodeError:
        logging.error(f"File {filepath} is not valid JSON.")
        return []

def save_dataset(data: list, filepath: str):
    """Save dataset to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_processed_ids(model_name: str, intermediate_filepath: str) -> set:
    """
    Load processed example IDs to avoid reprocessing
    model_name: name of the model
    intermediate_filepath: path to intermediate results file
    """
    processed_ids = set()
    if os.path.exists(intermediate_filepath):
        try:
            intermediate_data = load_dataset(intermediate_filepath)
            processed_ids.update([res["id"] for res in intermediate_data])
            logging.info(f"Loaded {len(intermediate_data)} processed example IDs for model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading intermediate results file {intermediate_filepath}: {str(e)}")
    else:
        logging.info(f"Intermediate results file {intermediate_filepath} does not exist, will create new file.")
    return processed_ids

@retry(
    retry=retry_if_exception_type((APIConnectionError, APIError, RateLimitError)),
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True,
)
def safe_generate_response(model, prompt):
    """Wrapper function for model response generation with retry mechanism"""
    try:
        return model.generate_response(prompt)
    except APIConnectionError as e:
        logging.error(f"API connection error: {str(e)}")
        raise
    except APIError as e:
        logging.error(f"API error: {str(e)}")
        raise
    except RateLimitError as e:
        logging.error(f"Rate limit error: {str(e)}")
        raise

def run_main_experiment(dataset, models):
    """Run main experiment (with conditions)"""
    stats = {model.name: {
        "total_examples": 0,
        "total_expected_answers": 0,
        "total_generated_answers": 0,
        "answer_count_difference": 0
    } for model in models}

    for model in models:
        intermediate_filepath = f"intermediate_results_{model.name}_main_experiment.json"
        results = load_dataset(intermediate_filepath)
        
        if not isinstance(results, dict) or model.name not in results:
            logging.error(f"Invalid result format or missing data for model {model.name}")
            continue
        
        model_results = results[model.name]
        processed_ids = set(res["id"] for res in model_results if isinstance(res, dict) and "id" in res)
        
        for example in tqdm(dataset, desc=f"Processing examples for model {model.name} (Main Experiment)"):
            example_id = example["id"]
            if example_id in processed_ids:
                continue  # Skip processed examples

            try:
                question = example["question"]
                ctxs = example["ctxs"]
                properties = example.get("properties", [])

                prompt = construct_prompt(question, ctxs)
                fragment_mapping = {f"Fragment {i+1}": ctx for i, ctx in enumerate(ctxs)}
                response = safe_generate_response(model, prompt)

                if not response:
                    logging.warning(f"No response from model {model.name} for example {example_id}")
                    continue

                actual_conditions = response.get("conditions", [])
                unique_actual_conditions = actual_conditions

                stats[model.name]["total_examples"] += 1
                stats[model.name]["total_expected_answers"] += len(properties)
                stats[model.name]["total_generated_answers"] += len(unique_actual_conditions)
                stats[model.name]["answer_count_difference"] += len(unique_actual_conditions) - len(properties)

                evaluations = []
                for idx, actual in enumerate(unique_actual_conditions):
                    expected = properties[idx] if idx < len(properties) else {}
                    
                    cond_eval = evaluate_condition_correctness(
                        input_text=prompt,
                        actual_condition=actual.get("condition", ""),
                        expected_condition=expected.get("condition", "")
                    )

                    ans_eval = evaluate_answer_correctness(
                        input_text=prompt,
                        actual_answer=actual.get("answer", ""),
                        expected_answer=expected.get("groundtruth", "")
                    )

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
                    "conditions": unique_actual_conditions,
                    "evaluations": evaluations,
                    "answer_count_difference": len(unique_actual_conditions) - len(properties)
                }

                model_results.append(result_entry)
                results[model.name] = model_results
                save_dataset(results, intermediate_filepath)
                logging.info(f"Saved intermediate results for model {model.name}, example ID: {example_id}")

            except Exception as e:
                logging.error(f"Error processing model {model.name}, example ID {example_id}: {str(e)}")
                continue

    return stats

def run_comparison_experiment(dataset, models):
    """Run comparison experiment (without conditions)"""
    stats = {model.name: {
        "total_examples": 0,
        "total_expected_answers": 0,
        "total_generated_answers": 0,
        "answer_count_difference": 0
    } for model in models}

    for model in models:
        intermediate_filepath = f"intermediate_results_{model.name}_comparison_experiment.json"
        try:
            results = load_dataset(intermediate_filepath)
            if not isinstance(results, dict) or model.name not in results:
                results = {model.name: []}
        except FileNotFoundError:
            results = {model.name: []}
        
        model_results = results[model.name]
        processed_ids = set(res["id"] for res in model_results if isinstance(res, dict) and "id" in res)
        
        for example in tqdm(dataset, desc=f"Processing examples for model {model.name} (Comparison Experiment)"):
            example_id = example["id"]
            if example_id in processed_ids:
                continue

            try:
                question = example["question"]
                ctxs = example["ctxs"]
                properties = example.get("properties", [])
                expected_answers = [prop.get("groundtruth", "") for prop in properties]
                prompt = construct_prompt_no_condition(question, ctxs)
                fragment_mapping = {f"Fragment {i+1}": ctx for i, ctx in enumerate(ctxs)}
                response = safe_generate_response(model, prompt)

                if not response:
                    logging.warning(f"No response from model {model.name} for example {example_id}")
                    continue

                generated_answers = response.get("answers", [])

                stats[model.name]["total_examples"] += 1
                stats[model.name]["total_expected_answers"] += len(expected_answers)
                stats[model.name]["total_generated_answers"] += len(generated_answers)
                stats[model.name]["answer_count_difference"] += len(generated_answers) - len(expected_answers)

                evaluations = []
                for idx, actual in enumerate(generated_answers):
                    expected = properties[idx] if idx < len(properties) else {}

                    ans_eval = evaluate_answer_correctness(
                        input_text=prompt,
                        actual_answer=actual.get("answer", ""),
                        expected_answer=expected.get("groundtruth", "")
                    )

                    expected_citations = [citation["title"] for citation in expected.get("citations", [])]
                    actual_citations = actual.get("citations", [])

                    cit_eval = evaluate_citation_correctness(
                        actual_output=actual_citations,
                        expected_output=expected_citations,
                        fragment_mapping=fragment_mapping
                    )

                    evaluations.append({
                        "answer_evaluation": ans_eval,
                        "citation_evaluation": cit_eval
                    })

                result_entry = {
                    "id": example_id,
                    "question": question,
                    "generated_answers": generated_answers,
                    "expected_answers": expected_answers,
                    "evaluations": evaluations,
                    "answer_count_difference": len(generated_answers) - len(expected_answers)
                }

                model_results.append(result_entry)
                results[model.name] = model_results
                save_dataset(results, intermediate_filepath)
                logging.info(f"Saved comparison experiment results for model {model.name}, example ID: {example_id}")

            except Exception as e:
                logging.error(f"Error processing model {model.name}, example ID {example_id}: {str(e)}")
                continue

    return stats

def compare_answer_scores(models):
    """Compare answer scores between main and comparison experiments"""
    comparison = {}
    for model in models:
        main_filepath = f"intermediate_results_{model.name}_main_experiment.json"
        comparison_filepath = f"intermediate_results_{model.name}_comparison_experiment.json"

        main_results = load_dataset(main_filepath)
        comparison_results = load_dataset(comparison_filepath)

        main_scores = []
        for example in main_results:
            for evaluation in example.get('evaluations', []):
                if 'answer_evaluation' in evaluation and 'score' in evaluation['answer_evaluation']:
                    main_scores.append(evaluation['answer_evaluation']['score'])

        comparison_scores = []
        for example in comparison_results:
            for evaluation in example.get('evaluations', []):
                if 'answer_evaluation' in evaluation and 'score' in evaluation['answer_evaluation']:
                    comparison_scores.append(evaluation['answer_evaluation']['score'])

        comparison[model.name] = {
            "conditional_mean": np.mean(main_scores) if main_scores else 0,
            "conditional_std": np.std(main_scores) if main_scores else 0,
            "no_condition_mean": np.mean(comparison_scores) if comparison_scores else 0,
            "no_condition_std": np.std(comparison_scores) if comparison_scores else 0,
            "difference": (np.mean(main_scores) - np.mean(comparison_scores)) if main_scores and comparison_scores else 0
        }

    return comparison

def compute_stats_from_results(models):
    """Compute experiment statistics"""
    stats = {
        "main_experiment": {model.name: {
            "total_examples": 0,
            "total_expected_answers": 0,
            "total_generated_answers": 0,
            "answer_count_difference": 0
        } for model in models},
        "comparison_experiment": {model.name: {
            "total_examples": 0,
            "total_expected_answers": 0,
            "total_generated_answers": 0,
            "answer_count_difference": 0
        } for model in models}
    }

    for model in models:
        # Main experiment statistics
        main_filepath = f"intermediate_results_{model.name}_main_experiment.json"
        main_results = load_dataset(main_filepath)
        for example in main_results:
            stats["main_experiment"][model.name]["total_examples"] += 1
            stats["main_experiment"][model.name]["total_expected_answers"] += len(example.get("properties", []))
            stats["main_experiment"][model.name]["total_generated_answers"] += len(example.get("conditions", []))
            stats["main_experiment"][model.name]["answer_count_difference"] += example.get("answer_count_difference", 0)

        # Comparison experiment statistics
        comparison_filepath = f"intermediate_results_{model.name}_comparison_experiment.json"
        comparison_results = load_dataset(comparison_filepath)
        for example in comparison_results:
            stats["comparison_experiment"][model.name]["total_examples"] += 1
            stats["comparison_experiment"][model.name]["total_expected_answers"] += len(example.get("expected_answers", []))
            stats["comparison_experiment"][model.name]["total_generated_answers"] += len(example.get("generated_answers", []))
            stats["comparison_experiment"][model.name]["answer_count_difference"] += example.get("answer_count_difference", 0)

    return stats

def main():
    dataset = load_dataset("mcaqar.json")
    if not dataset:
        logging.error("Dataset loading failed. Please check if mcaqar.json exists and is properly formatted.")
        return

    models = get_models()
    if not models:
        logging.error("No models loaded. Please check get_models function.")
        return

    # Run main experiment (with conditions)
    print("Running main experiment (with conditions)")
    main_stats = run_main_experiment(dataset, models)

    # Run comparison experiment (without conditions)
    print("\nRunning comparison experiment (without conditions)")
    comparison_stats = run_comparison_experiment(dataset, models)

    # Calculate statistics
    stats = compute_stats_from_results(models)

    print("\n=== Main Experiment Statistics ===")
    for model in models:
        model_name = model.name
        model_stats = stats["main_experiment"][model_name]
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

    print("\n=== Comparison Experiment Statistics ===")
    for model in models:
        model_name = model.name
        model_stats = stats["comparison_experiment"][model_name]
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

    # Compare answer scores
    print("\nComparing answer scores")
    comparison = compare_answer_scores(models)

    print("\n=== Answer Score Comparison ===")
    for model in models:
        model_name = model.name
        scores = comparison[model_name]
        print(f"\nModel: {model_name}")
        print(f"Conditional RAG - Mean: {scores['conditional_mean']:.4f}, Std: {scores['conditional_std']:.4f}")
        print(f"No-condition RAG - Mean: {scores['no_condition_mean']:.4f}, Std: {scores['no_condition_std']:.4f}")
        print(f"Difference (Conditional - No-condition): {scores['difference']:.4f}")

    # Save comparison results
    with open("answer_score_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print("\nAnswer score comparison results saved to answer_score_comparison.json")

    # Save overall statistics
    with open("experiment_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("\nExperiment statistics saved to experiment_statistics.json")
    print("\nExperiment completed. Results have been saved to JSON files.")

if __name__ == "__main__":
    main()