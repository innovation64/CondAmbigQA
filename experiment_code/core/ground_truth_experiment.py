import json
import logging
import os
from tqdm import tqdm
from models import get_models
from utils import construct_modified_prompt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIConnectionError, APIError, RateLimitError
from evaluator import (
    evaluate_answer_correctness,
    evaluate_citation_correctness
)
from statistics import mean

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

@retry(
    retry=retry_if_exception_type((APIConnectionError, APIError, RateLimitError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def generate_response(model, example) -> dict:
    """Generate model response"""
    prompt = construct_modified_prompt(
        question=example["question"],
        ctxs=example["ctxs"],
        properties=example["properties"]
    )
    return model.generate_response(prompt)

def sanitize_filename(name: str) -> str:
    """Convert model name to a safe filename"""
    # Replace special characters
    safe_name = name.replace(":", "_").replace("/", "_").replace("\\", "_")
    safe_name = safe_name.replace("*", "_").replace("?", "_").replace("\"", "_")
    safe_name = safe_name.replace("<", "_").replace(">", "_").replace("|", "_")
    return safe_name

def load_or_create_results(model_name: str) -> dict:
    """Load or create results file"""
    safe_name = sanitize_filename(model_name)
    filepath = f"results_{safe_name}_ground_truth.json"
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                results = json.load(f)
            logging.info(f"Loaded {len(results)} existing results for model {model_name}")
            return results
        except Exception as e:
            logging.error(f"Error loading results: {str(e)}, creating new results file")
            return {}
    else:
        logging.info(f"Creating new results file for model {model_name}")
        return {}

def save_results(results: dict, model_name: str):
    """Save results, add directory creation and better error handling"""
    # Create results directory (if it doesn't exist)
    results_dir = "experiment_results"
    try:
        os.makedirs(results_dir, exist_ok=True)
    except Exception as e:
        logging.warning(f"Unable to create results directory: {str(e)}, will try to save to current directory")
        results_dir = ""
    
    # Use safe filename
    safe_name = sanitize_filename(model_name)
    
    # Try multiple save locations
    save_locations = [
        os.path.join(results_dir, f"results_{safe_name}_ground_truth.json"),
        f"results_{safe_name}_ground_truth.json",  # Current directory
        os.path.join(os.path.expanduser("~"), f"results_{safe_name}_ground_truth.json")  # User home directory
    ]
    
    for filepath in save_locations:
        try:
            # Ensure target directory exists
            dir_path = os.path.dirname(filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                
            # Try to save file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info(f"Successfully saved results to: {filepath}")
            return  # Save successful, exit function
        except PermissionError:
            logging.warning(f"Permission denied: {filepath}, trying next location")
        except Exception as e:
            logging.warning(f"Failed to save to {filepath}: {str(e)}, trying next location")
    
    # If all save locations fail, try using a temporary file
    import tempfile
    import time
    timestamp = int(time.time())
    temp_filepath = os.path.join(tempfile.gettempdir(), f"results_{safe_name}_{timestamp}.json")
    
    try:
        with open(temp_filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved results to temporary location: {temp_filepath}")
    except Exception as e:
        logging.error(f"All save attempts failed, unable to save results: {str(e)}")

def process_single_example(model, example):
    """Process a single example with greedy best-match per generated condition."""
    try:
        # 1. Model generation
        response = generate_response(model, example)
        if not response or "conditions" not in response:
            return None

        ctxs = example["ctxs"]
        fragment_mapping = {f"Fragment {i+1}": ctx for i, ctx in enumerate(ctxs)}

        output = {
            "id": example["id"],
            "question": example["question"],
            "conditions": [],       # Final output pairing list
            "evaluations": []       # Evaluation for each pair
        }

        generated = response["conditions"]
        props     = example.get("properties", [])

        used_exp = set()
        # 2. Greedy best match for each actual
        for i, actual in enumerate(generated):
            best_score = -float("inf")
            best_j     = None
            best_eval  = {
                "answer_score":   0,
                "citation_score": 0
            }

            for j, exp in enumerate(props):
                if j in used_exp:
                    continue
                # Score answers
                ans_eval = evaluate_answer_correctness(
                    input_text=example["question"],
                    actual_answer=actual.get("answer", ""),
                    expected_answer=exp.get("groundtruth", "")
                )["score"]
                # Score citations
                cit_eval = evaluate_citation_correctness(
                    actual_output=actual.get("citations", []),
                    expected_output=[c["title"] for c in exp.get("citations", [])],
                    fragment_mapping=fragment_mapping
                )["score"]
                # Combined score (adjustable weights)
                score = 0.6 * ans_eval + 0.4 * cit_eval

                if score > best_score:
                    best_score = score
                    best_j     = j
                    best_eval  = {
                        "answer_score":   ans_eval,
                        "citation_score": cit_eval
                    }

            # 3. Record this best match
            output["conditions"].append({
                "condition": props[best_j]["condition"] if best_j is not None else None,
                "answer":    actual.get("answer", ""),
                "citations": actual.get("citations", [])
            })
            output["evaluations"].append({
                "actual_idx":      i,
                "expected_idx":    best_j,
                "combined_score":  best_score,
                **best_eval
            })
            if best_j is not None:
                used_exp.add(best_j)

        return output

    except Exception as e:
        logging.error(f"Error processing example: {e}")
        return None


def calculate_model_scores(results: dict) -> dict:
    """Calculate overall model scores"""
    answer_scores = []
    citation_scores = []
    
    for example_output in results.values():
        for eval_data in example_output.get("evaluations", []):
            if "answer_score" in eval_data:
                answer_scores.append(eval_data["answer_score"])
            if "citation_score" in eval_data:
                citation_scores.append(eval_data["citation_score"])
    
    return {
        "average_answer_score": mean(answer_scores) if answer_scores else 0,
        "average_citation_score": mean(citation_scores) if citation_scores else 0
    }

def main():
    # Load dataset
    with open("cleaned_mcaqa.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        if not dataset:
            logging.error("Failed to load dataset")
            return

    # Get model list
    models = get_models()
    
    # Store scores for all models
    all_model_scores = {}
    
    # Generate answers for each model
    for model in models:
        logging.info(f"\nProcessing model: {model.name}")
        
        # Load or create results file
        results = load_or_create_results(model.name)
        
        # Process each example
        progress = tqdm(dataset)
        for example in progress:
            example_id = example["id"]
            
            # Skip already processed examples
            if example_id in results:
                continue
            
            # Process current example
            output = process_single_example(model, example)
            if output:
                results[example_id] = output
                # Save results in real-time
                save_results(results, model.name)
            
            # Update progress bar
            progress.set_description(
                f"Model: {model.name}, "
                f"Processed: {len(results)}"
            )
        
        # Calculate scores for this model
        model_scores = calculate_model_scores(results)
        all_model_scores[model.name] = model_scores
        
        logging.info(f"Model {model.name} processing completed")
        logging.info(f"Average answer score: {model_scores['average_answer_score']:.4f}")
        logging.info(f"Average citation score: {model_scores['average_citation_score']:.4f}")
    
    # Save scores for all models
    with open("model_scores_ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(all_model_scores, f, ensure_ascii=False, indent=2)
    logging.info("\nAll model evaluations completed, results saved to model_scores.json")

if __name__ == "__main__":
    main()