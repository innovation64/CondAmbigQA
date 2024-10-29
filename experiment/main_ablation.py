import json
import logging
import os
from tqdm import tqdm
from models import get_models
from utils import construct_prompt_no_condition
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
    """Generate model response without conditions"""
    prompt = construct_prompt_no_condition(
        question=example["question"],
        ctxs=example["ctxs"]
    )
    return model.generate_response(prompt)

def load_or_create_results(model_name: str) -> dict:
    """Load or create results file"""
    filepath = f"results_{model_name}_ablation.json"
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
    """Save results"""
    filepath = f"results_{model_name}_ablation.json"
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")

def process_single_example(model, example):
    """Process a single example for ablation study"""
    try:
        # Generate response
        response = generate_response(model, example)
        
        # Build fragment mapping
        fragment_mapping = {f"Fragment {i+1}": ctx for i, ctx in enumerate(example["ctxs"])}
        
        # Prepare output structure
        output = {
            "id": example["id"],
            "question": example["question"],
            "answers": [],
            "evaluations": []
        }
        
        # Process answers
        if "answers" in response:
            for i, prop in enumerate(example["properties"]):
                if i < len(response["answers"]):
                    answer_data = response["answers"][i]
                    
                    # Record generated answer and citations
                    answer_output = {
                        "answer": answer_data.get("answer", ""),
                        "citations": answer_data.get("citations", [])
                    }
                    output["answers"].append(answer_output)
                    
                    # Evaluate answer and citations
                    evaluation = {
                        "answer_score": evaluate_answer_correctness(
                            input_text=example["question"],
                            actual_answer=answer_data.get("answer", ""),
                            expected_answer=prop.get("groundtruth", "")
                        )["score"],
                        "citation_score": evaluate_citation_correctness(
                            actual_output=answer_data.get("citations", []),
                            expected_output=[citation["title"] for citation in prop.get("citations", [])],
                            fragment_mapping=fragment_mapping
                        )["score"]
                    }
                    output["evaluations"].append(evaluation)
                else:
                    # Handle case where model generates fewer answers than expected
                    output["answers"].append({
                        "answer": "",
                        "citations": []
                    })
                    output["evaluations"].append({
                        "answer_score": 0,
                        "citation_score": 0
                    })
        
        return output
    except Exception as e:
        logging.error(f"Error processing example: {str(e)}")
        return None

def calculate_model_scores(results: dict) -> dict:
    """Calculate overall model scores"""
    answer_scores = []
    citation_scores = []
    answer_counts = []
    
    for example_output in results.values():
        answer_counts.append(len(example_output.get("answers", [])))
        for eval_data in example_output.get("evaluations", []):
            if "answer_score" in eval_data:
                answer_scores.append(eval_data["answer_score"])
            if "citation_score" in eval_data:
                citation_scores.append(eval_data["citation_score"])
    
    return {
        "average_answer_score": mean(answer_scores) if answer_scores else 0,
        "average_citation_score": mean(citation_scores) if citation_scores else 0,
        "average_answer_count": mean(answer_counts) if answer_counts else 0
    }

def main():
    # Load dataset
    with open("mcaqar.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        if not dataset:
            logging.error("Failed to load dataset")
            return

    # Get models
    models = get_models()
    
    # Store all model scores
    all_model_scores = {}
    
    # Process each model
    for model in models:
        logging.info(f"\nProcessing model: {model.name}")
        
        # Load or create results
        results = load_or_create_results(model.name)
        
        # Process each example
        progress = tqdm(dataset)
        for example in progress:
            example_id = example["id"]
            
            # Skip processed examples
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
        
        # Calculate model scores
        model_scores = calculate_model_scores(results)
        all_model_scores[model.name] = model_scores
        
        logging.info(f"Model {model.name} processing completed")
        logging.info(f"Average answer score: {model_scores['average_answer_score']:.4f}")
        logging.info(f"Average citation score: {model_scores['average_citation_score']:.4f}")
        logging.info(f"Average answer count: {model_scores['average_answer_count']:.2f}")
    
    # Save all model scores
    with open("model_scores_ablation.json", "w", encoding="utf-8") as f:
        json.dump(all_model_scores, f, ensure_ascii=False, indent=2)
    logging.info("\nAll model evaluations completed, results saved to model_scores_ablation.json")

if __name__ == "__main__":
    main()