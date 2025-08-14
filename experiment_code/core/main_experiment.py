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

def sanitize_filename(name: str) -> str:
    """Convert model name to safe filename"""
    # Replace special characters
    safe_name = name.replace(":", "_").replace("/", "_").replace("\\", "_")
    safe_name = safe_name.replace("*", "_").replace("?", "_").replace("\"", "_")
    safe_name = safe_name.replace("<", "_").replace(">", "_").replace("|", "_")
    return safe_name

def load_processed_ids(model_name: str, intermediate_filepath: str) -> set:
    """
    Load processed example IDs to avoid reprocessing
    model_name: name of the model
    intermediate_filepath: path to intermediate results file
    """
    safe_model_name = sanitize_filename(model_name)
    safe_filepath = intermediate_filepath.replace(model_name, safe_model_name)
    
    processed_ids = set()
    if os.path.exists(safe_filepath):
        try:
            intermediate_data = load_dataset(safe_filepath)
            processed_ids.update([res["id"] for res in intermediate_data])
            logging.info(f"Loaded {len(intermediate_data)} processed example IDs for model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading intermediate results file {safe_filepath}: {str(e)}")
    else:
        logging.info(f"Intermediate results file {safe_filepath} does not exist, will create new file.")
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
    """Run main experiment with greedy best‐match evaluation (no threshold)."""
    # Initialize statistics
    stats = {
        model.name: {
            "total_examples": 0,
            "total_expected_answers": 0,
            "total_generated_answers": 0,
            "total_matched_answers": 0,
            "answer_count_difference": 0
        }
        for model in models
    }

    for model in models:
        safe_name = sanitize_filename(model.name)
        intermediate_fp = f"intermediate_results_{safe_name}_main_experiment.json"
        final_fp        = f"results_{safe_name}.json"

        # load or initialize intermediate results
        try:
            results = load_dataset(intermediate_fp)
            if not isinstance(results, dict) or model.name not in results:
                results = {model.name: []}
        except FileNotFoundError:
            results = {model.name: []}

        model_results = results[model.name]
        processed_ids = {r["id"] for r in model_results if isinstance(r, dict) and "id" in r}
        finalized_results = []

        for example in tqdm(dataset, desc=f"Processing examples for model {model.name}"):
            eid = example["id"]
            # if already done, carry over stats & result
            if eid in processed_ids:
                for r in model_results:
                    if r.get("id") == eid:
                        finalized_results.append(r)
                        stats[model.name]["total_examples"] += 1
                        stats[model.name]["total_expected_answers"] += len(r.get("expected_conditions", []))
                        stats[model.name]["total_generated_answers"] += len(r.get("conditions", []))
                        stats[model.name]["total_matched_answers"] += len(r.get("evaluations", []))
                continue

            try:
                # extract inputs
                question   = example["question"]
                ctxs       = example["ctxs"]
                properties = example.get("properties", [])

                prompt = construct_prompt(question, ctxs)
                fragment_mapping = {f"Fragment {i+1}": c for i, c in enumerate(ctxs)}
                resp   = safe_generate_response(model, prompt)
                if not resp:
                    logging.warning(f"No response for model {model.name}, example {eid}")
                    continue

                actual_conditions = resp.get("conditions", [])
                unique_actual_conditions = actual_conditions  # dedupe here if desired

                # update simple stats
                stats[model.name]["total_examples"] += 1
                stats[model.name]["total_expected_answers"] += len(properties)
                stats[model.name]["total_generated_answers"] += len(unique_actual_conditions)
                stats[model.name]["answer_count_difference"] += (
                    len(unique_actual_conditions) - len(properties)
                )

                # greedy best‐match evaluation
                evaluations = []
                used_exp = set()
                for i, actual in enumerate(unique_actual_conditions):
                    best_score = -float("inf")
                    best_j     = None
                    best_eval  = {
                        "condition_evaluation": {},
                        "answer_evaluation": {},
                        "citation_evaluation": {}
                    }

                    for j, exp in enumerate(properties):
                        if j in used_exp:
                            continue
                        # evaluate condition / answer / citation
                        cond_eval = evaluate_condition_correctness(
                            input_text=prompt,
                            actual_condition=actual.get("condition", ""),
                            expected_condition=exp.get("condition", "")
                        )
                        ans_eval = evaluate_answer_correctness(
                            input_text=prompt,
                            actual_answer=actual.get("answer", ""),
                            expected_answer=exp.get("groundtruth", "")
                        )
                        expected_cits = [c["title"] for c in exp.get("citations", [])]
                        cit_eval = evaluate_citation_correctness(
                            actual_output=actual.get("citations", []),
                            expected_output=expected_cits,
                            fragment_mapping=fragment_mapping
                        )
                        # combine scores (adjust weights as needed)
                        score = (
                            cond_eval.get("score", 0) * 0.3 +
                            ans_eval.get("score", 0)  * 0.5 +
                            cit_eval.get("score", 0)  * 0.2
                        )
                        if score > best_score:
                            best_score = score
                            best_j     = j
                            best_eval  = {
                                "condition_evaluation": cond_eval,
                                "answer_evaluation":    ans_eval,
                                "citation_evaluation":  cit_eval
                            }

                    # record the best match—even if score is very low
                    evaluations.append({
                        "actual_idx":        i,
                        "expected_idx":      best_j,
                        "combined_score":    best_score,
                        **best_eval
                    })
                    if best_j is not None:
                        used_exp.add(best_j)

                # update matched count
                stats[model.name]["total_matched_answers"] += len(evaluations)

                # assemble result entry
                result = {
                    "id":                  eid,
                    "question":            question,
                    "expected_conditions": properties,
                    "conditions":   unique_actual_conditions,
                    "evaluations":         evaluations,
                    "answer_count_difference":
                        len(unique_actual_conditions) - len(properties)
                }
                model_results.append(result)
                finalized_results.append(result)

                # save intermediate
                results[model.name] = model_results
                save_dataset(results, intermediate_fp)
                logging.info(f"Saved intermediate for {model.name}, example {eid}")

            except Exception as e:
                logging.error(f"Error in {model.name}, example {eid}: {e}")
                continue

        # write out final results for this model
        save_dataset(finalized_results, final_fp)
        logging.info(f"Saved final results for model {model.name} to {final_fp}")

    return stats


def run_comparison_experiment(dataset, models):
    """Run comparison experiment (without conditions) using greedy best‐match evaluation."""
    # Initialize statistics
    stats = {
        model.name: {
            "total_examples": 0,
            "total_expected_answers": 0,
            "total_generated_answers": 0,
            "total_matched_answers": 0,
            "missing_answers": 0,
            "extra_answers": 0,
            "answer_count_difference": 0
        }
        for model in models
    }

    for model in models:
        safe_name        = sanitize_filename(model.name)
        intermediate_fp  = f"intermediate_results_{safe_name}_comparison_experiment.json"
        final_fp         = f"results_{safe_name}_compare.json"

        # Load or initialize intermediate results
        try:
            results = load_dataset(intermediate_fp)
            if not isinstance(results, dict) or model.name not in results:
                results = {model.name: []}
        except FileNotFoundError:
            results = {model.name: []}

        model_results   = results[model.name]
        processed_ids   = {r["id"] for r in model_results if isinstance(r, dict) and "id" in r}
        finalized_results = []

        for example in tqdm(dataset, desc=f"Processing examples for model {model.name} (Comparison)"):
            eid = example["id"]
            if eid in processed_ids:
                # carry over already‐computed results & stats
                for r in model_results:
                    if r.get("id") == eid:
                        finalized_results.append(r)
                        stats[model.name]["total_examples"]           += 1
                        stats[model.name]["total_expected_answers"]  += len(r.get("expected_answers", []))
                        stats[model.name]["total_generated_answers"] += len(r.get("generated_answers", []))
                        stats[model.name]["total_matched_answers"]   += len(r.get("evaluations", []))
                        stats[model.name]["answer_count_difference"] += r.get("answer_count_difference", 0)
                continue

            try:
                # Extract inputs
                question   = example["question"]
                ctxs       = example["ctxs"]
                properties = example.get("properties", [])
                expected_answers = [p.get("groundtruth", "") for p in properties]

                prompt = construct_prompt_no_condition(question, ctxs)
                fragment_mapping = {f"Fragment {i+1}": c for i, c in enumerate(ctxs)}
                resp = safe_generate_response(model, prompt)
                if not resp:
                    logging.warning(f"No response from model {model.name} for example {eid}")
                    continue

                generated_answers = resp.get("answers", [])

                # Update base stats
                stats[model.name]["total_examples"]           += 1
                stats[model.name]["total_expected_answers"]   += len(expected_answers)
                stats[model.name]["total_generated_answers"]  += len(generated_answers)
                stats[model.name]["answer_count_difference"] += (
                    len(generated_answers) - len(expected_answers)
                )

                # Greedy best‐match evaluation
                evaluations = []
                used_exp    = set()
                for i, actual in enumerate(generated_answers):
                    best_score = -float("inf")
                    best_j     = None
                    best_ans_eval = {}
                    best_cit_eval = {}

                    for j, exp in enumerate(properties):
                        if j in used_exp:
                            continue

                        ans_eval = evaluate_answer_correctness(
                            input_text=prompt,
                            actual_answer=actual.get("answer", ""),
                            expected_answer=exp.get("groundtruth", "")
                        )
                        expected_cits = [c["title"] for c in exp.get("citations", [])]
                        cit_eval = evaluate_citation_correctness(
                            actual_output=actual.get("citations", []),
                            expected_output=expected_cits,
                            fragment_mapping=fragment_mapping
                        )

                        # Combine scores (adjust weights if desired)
                        score = (
                            ans_eval.get("score", 0) * 0.7 +
                            cit_eval.get("score", 0) * 0.3
                        )
                        if score > best_score:
                            best_score    = score
                            best_j        = j
                            best_ans_eval = ans_eval
                            best_cit_eval = cit_eval

                    # Record best match (even if score is low)
                    evaluations.append({
                        "actual_idx":       i,
                        "expected_idx":     best_j,
                        "combined_score":   best_score,
                        "answer_evaluation":   best_ans_eval,
                        "citation_evaluation": best_cit_eval
                    })
                    if best_j is not None:
                        used_exp.add(best_j)

                # Update match/missing/extra stats
                matched = len(evaluations)
                stats[model.name]["total_matched_answers"] += matched
                stats[model.name]["missing_answers"]       += len(expected_answers) - matched
                stats[model.name]["extra_answers"]         += len(generated_answers) - matched

                # Assemble and save this example's result
                result_entry = {
                    "id": eid,
                    "question": question,
                    "expected_answers":   expected_answers,
                    "generated_answers":  generated_answers,
                    "evaluations":        evaluations,
                    "answer_count_difference":
                        len(generated_answers) - len(expected_answers)
                }
                model_results.append(result_entry)
                finalized_results.append(result_entry)

                # Persist intermediate state
                results[model.name] = model_results
                save_dataset(results, intermediate_fp)
                logging.info(f"Saved intermediate comparison results for {model.name}, example {eid}")

            except Exception as e:
                logging.error(f"Error processing model {model.name}, example {eid}: {e}")
                continue

        # Write out final results for this model
        save_dataset(finalized_results, final_fp)
        logging.info(f"Saved final comparison results for {model.name} to {final_fp}")

    return stats


def compute_model_scores(models, experiment_type="main"):
    """Compute model scores for specified experiment type"""
    scores = {}
    
    for model in models:
        model_name = model.name
        safe_model_name = sanitize_filename(model_name)
        
        if experiment_type == "main":
            filepath = f"results_{safe_model_name}.json"
        else:  # comparison experiment
            filepath = f"results_{safe_model_name}_compare.json"
        
        results = load_dataset(filepath)
        if not results:
            logging.warning(f"No results found for {experiment_type} experiment, model: {model_name}")
            continue
            
        answer_scores = []
        citation_scores = []
        total_answers = 0
        
        for example in results:
            # Count answers for this example
            if experiment_type == "main":
                total_answers += len(example.get("conditions", []))
            else:
                total_answers += len(example.get("generated_answers", []))
            
            for evaluation in example.get("evaluations", []):
                if "answer_evaluation" in evaluation and "score" in evaluation["answer_evaluation"]:
                    answer_scores.append(evaluation["answer_evaluation"]["score"])
                    
                if "citation_evaluation" in evaluation and "score" in evaluation["citation_evaluation"]:
                    citation_scores.append(evaluation["citation_evaluation"]["score"])
        
        # Calculate average answer count per example
        average_answer_count = total_answers / len(results) if results else 0
        
        scores[model_name] = {
            "average_answer_score": np.mean(answer_scores) if answer_scores else 0,
            "average_citation_score": np.mean(citation_scores) if citation_scores else 0,
            "average_answer_count": average_answer_count
        }
    
    return scores

def compare_answer_scores(models):
    """Compare answer scores between main and comparison experiments"""
    comparison = {}
    for model in models:
        model_name = model.name
        safe_model_name = sanitize_filename(model_name)
        main_filepath = f"results_{safe_model_name}.json"
        comparison_filepath = f"results_{safe_model_name}_compare.json"

        main_data = load_dataset(main_filepath)
        comparison_data = load_dataset(comparison_filepath)

        main_scores = []
        for example in main_data:
            for evaluation in example.get('evaluations', []):
                if 'answer_evaluation' in evaluation and 'score' in evaluation['answer_evaluation']:
                    main_scores.append(evaluation['answer_evaluation']['score'])

        comparison_scores = []
        for example in comparison_data:
            for evaluation in example.get('evaluations', []):
                if 'answer_evaluation' in evaluation and 'score' in evaluation['answer_evaluation']:
                    comparison_scores.append(evaluation['answer_evaluation']['score'])

        comparison[model_name] = {
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
        safe_model_name = sanitize_filename(model.name)
        main_filepath = f"results_{safe_model_name}.json"
        main_results = load_dataset(main_filepath)
        if not main_results:
            logging.warning(f"No results found for main experiment, model: {model.name}")
            continue
                
        for example in main_results:
            stats["main_experiment"][model.name]["total_examples"] += 1
            
            # Count expected answers based on properties if available
            properties_count = len(example.get("properties", []))
            if properties_count > 0:
                stats["main_experiment"][model.name]["total_expected_answers"] += properties_count
            else:
                # Fallback to evaluations count if properties not available
                stats["main_experiment"][model.name]["total_expected_answers"] += len(example.get("evaluations", []))
                
            stats["main_experiment"][model.name]["total_generated_answers"] += len(example.get("conditions", []))
            stats["main_experiment"][model.name]["answer_count_difference"] += example.get("answer_count_difference", 0)
        
        # Comparison experiment statistics
        comparison_filepath = f"results_{safe_model_name}_compare.json"
        comparison_results = load_dataset(comparison_filepath)
        if not comparison_results:
            logging.warning(f"No results found for comparison experiment, model: {model.name}")
            continue
            
        for example in comparison_results:
            stats["comparison_experiment"][model.name]["total_examples"] += 1
            stats["comparison_experiment"][model.name]["total_expected_answers"] += len(example.get("expected_answers", []))
            stats["comparison_experiment"][model.name]["total_generated_answers"] += len(example.get("generated_answers", []))
            stats["comparison_experiment"][model.name]["answer_count_difference"] += example.get("answer_count_difference", 0)
    
    return stats

def main():
    dataset = load_dataset("cleaned_mcaqa.json")
    if not dataset:
        logging.error("Dataset loading failed. Please check if mcaqar.json exists and is properly formatted.")
        return

    models = get_models()
    if not models:
        logging.error("No models loaded. Please check get_models function.")
        return

    # Run main experiment (with conditions)
    # print("Running main experiment (with conditions)")
    # main_stats = run_main_experiment(dataset, models)

    # # Run comparison experiment (without conditions)
    print("\nRunning comparison experiment (without conditions)")
    comparison_stats = run_comparison_experiment(dataset, models)

    # Calculate statistics
    stats = compute_stats_from_results(models)

    # Compute model scores for both experiments
    print("\nComputing model scores for main experiment")
    model_scores = compute_model_scores(models, experiment_type="main")
    save_dataset(model_scores, "model_scores.json")
    print("Model scores saved to model_scores.json")

    print("\nComputing model scores for comparison experiment")
    model_scores_compare = compute_model_scores(models, experiment_type="comparison")
    save_dataset(model_scores_compare, "model_scores_compare.json")
    print("Comparison model scores saved to model_scores_compare.json")

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