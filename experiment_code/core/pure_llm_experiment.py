import json
import logging
import os
from tqdm import tqdm
from models import get_models
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIConnectionError, APIError, RateLimitError
from evaluator import evaluate_answer_correctness
from statistics import mean
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def construct_pure_llm_prompt(question: str) -> str:
    """Construct pure LLM prompt for closed-book generation on ambiguous questions"""
    prompt = (
        f"Question: {question}\n\n"
        f"Please answer this question using your general knowledge only. "
        f"Since this question might be ambiguous or have multiple valid answers depending on different contexts, "
        f"provide all relevant answers that address different aspects or interpretations of the question.\n\n"
        f"IMPORTANT: Respond with ONLY the following JSON format, no other text:\n"
        f"{{\n"
        f"  \"answers\": [\n"
        f"    {{\n"
        f"      \"answer\": \"[First relevant answer]\"\n"
        f"    }},\n"
        f"    {{\n"
        f"      \"answer\": \"[Second relevant answer if applicable]\"\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n\n"
        f"Notes:\n"
        f"- Output MUST be valid JSON only\n"
        f"- Do NOT include any explanatory text outside the JSON\n"
        f"- Use your general knowledge to provide detailed answers\n"
        f"- Include multiple answers if the question is ambiguous\n"
        f"- Each answer should address a different aspect or interpretation\n"
    )
    return prompt

@retry(
    retry=retry_if_exception_type((APIConnectionError, APIError, RateLimitError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def generate_pure_llm_response(model, example) -> dict:
    """Generate model response using pure LLM without retrieval"""
    prompt = construct_pure_llm_prompt(question=example["question"])
    return model.generate_response(prompt)

def sanitize_filename(name: str) -> str:
    """Convert model name to safe filename"""
    safe_name = name.replace(":", "_").replace("/", "_").replace("\\", "_")
    safe_name = safe_name.replace("*", "_").replace("?", "_").replace("\"", "_")
    safe_name = safe_name.replace("<", "_").replace(">", "_").replace("|", "_")
    return safe_name

def load_or_create_results(model_name: str) -> dict:
    """Load or create results file for pure LLM experiment"""
    safe_name = sanitize_filename(model_name)
    filepath = f"pure_llm_balanced_results_{safe_name}.json"
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                results = json.load(f)
            logging.info(f"Loaded {len(results)} existing pure LLM balanced results for model {model_name}")
            return results
        except Exception as e:
            logging.error(f"Error loading results: {str(e)}, creating new results file")
            return {}
    else:
        logging.info(f"Creating new pure LLM balanced results file for model {model_name}")
        return {}

def save_results(results: dict, model_name: str):
    """Save pure LLM experiment results"""
    safe_name = sanitize_filename(model_name)
    filepath = f"pure_llm_balanced_results_{safe_name}.json"
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved pure LLM balanced results to: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save pure LLM balanced results: {str(e)}")

def calculate_balanced_scores(answer_score: float, 
                           answer_count_difference: int,
                           expected_answer_count: int) -> float:
    """
    Calculate balanced scores using the same penalty system as main experiment
    """
    # Constants from vmain/vcom
    HIGH_QUALITY_THRESHOLD = 0.5
    SINGLE_ANSWER_PENALTY = 0.9
    MISSING_ANSWER_MAX_PENALTY = 0.2
    EXTRA_ANSWER_MAX_PENALTY = 0.15
    
    # Model config (default)
    quality_weight = 0.6
    quantity_weight = 0.4
    missing_severity = 0.6
    extra_severity = 0.3
    
    # Prevent division by zero
    safe_expected_count = max(expected_answer_count, 1)
    
    # Calculate actual answer count
    actual_answer_count = safe_expected_count + answer_count_difference
    
    # Quality factor
    base_quality = answer_score
    quality_score = base_quality
    
    # Quantity adjustment
    quantity_adjustment = 1.0
    
    if actual_answer_count == 1 and safe_expected_count > 1:
        # Single answer penalty
        quantity_adjustment = 1.0 - SINGLE_ANSWER_PENALTY
    elif answer_count_difference < 0:
        # Missing answers
        missing_ratio = abs(answer_count_difference) / safe_expected_count
        quality_factor = max(0.5, min(1.0, answer_score / HIGH_QUALITY_THRESHOLD))
        penalty = min(missing_ratio * missing_severity * (2 - quality_factor), 
                    MISSING_ANSWER_MAX_PENALTY)
        quantity_adjustment = 1.0 - penalty
    elif answer_count_difference > 0:
        # Extra answers
        extra_ratio = answer_count_difference / safe_expected_count
        quality_factor = max(0.5, min(1.0, answer_score / HIGH_QUALITY_THRESHOLD))
        
        if extra_ratio > 1.0:
            penalty = min(extra_ratio * extra_severity, EXTRA_ANSWER_MAX_PENALTY)
        else:
            penalty = min(extra_ratio * extra_severity * 0.5, EXTRA_ANSWER_MAX_PENALTY * 0.5)
        
        penalty = penalty * (2 - quality_factor)
        quantity_adjustment = 1.0 - penalty
    
    # Combine quality and quantity
    quality_component = quality_score * quality_weight
    quantity_component = quantity_adjustment * quantity_weight
    combined_factor = quality_component + quantity_component
    
    # Calculate final balanced score
    balanced_score = answer_score * combined_factor
    
    # Keep in valid range [0, 1]
    balanced_score = min(max(balanced_score, 0), 1)
    
    return balanced_score

def process_single_example_pure_llm_balanced(model, example):
    """Process with balanced evaluation matching main experiment methodology"""
    try:
        # 1. Generate response (closed-book)
        response = generate_pure_llm_response(model, example)
        if not response or "answers" not in response:
            return None

        output = {
            "id": example["id"],
            "question": example["question"],
            "pure_llm_answers": response["answers"],
            "evaluations": [],
            "raw_scores": [],
            "balanced_scores": []
        }

        generated_answers = response["answers"]
        props = example.get("properties", [])

        # Expected answer count
        expected_answer_count = len(props)
        actual_answer_count = len(generated_answers)
        answer_count_difference = actual_answer_count - expected_answer_count

        # 2. Greedy matching (same as main experiment)
        used_exp = set()
        matched_scores = []
        
        for i, llm_answer in enumerate(generated_answers):
            best_score = -float("inf")
            best_j = None
            
            for j, exp in enumerate(props):
                if j in used_exp:
                    continue
                
                # Answer scoring
                ans_eval = evaluate_answer_correctness(
                    input_text=example["question"],
                    actual_answer=llm_answer.get("answer", ""),
                    expected_answer=exp.get("groundtruth", "")
                )["score"]
                
                if ans_eval > best_score:
                    best_score = ans_eval
                    best_j = j
            
            # Record match
            if best_j is not None:
                used_exp.add(best_j)
                matched_scores.append(best_score)
                output["evaluations"].append({
                    "llm_answer_idx": i,
                    "llm_answer": llm_answer.get("answer", ""),
                    "matched_condition": props[best_j]["condition"],
                    "expected_answer": props[best_j]["groundtruth"],
                    "expected_idx": best_j,
                    "answer_score": best_score
                })
            else:
                # No match found
                matched_scores.append(0.0)
                output["evaluations"].append({
                    "llm_answer_idx": i,
                    "llm_answer": llm_answer.get("answer", ""),
                    "matched_condition": None,
                    "expected_answer": None,
                    "expected_idx": None,
                    "answer_score": 0.0
                })

        # 3. Calculate average score (from matched answers only)
        avg_raw_score = mean(matched_scores) if matched_scores else 0.0
        
        # 4. Apply balanced scoring with penalties
        balanced_score = calculate_balanced_scores(
            avg_raw_score,
            answer_count_difference,
            expected_answer_count
        )
        
        # Store results
        output["raw_average_score"] = avg_raw_score
        output["balanced_average_score"] = balanced_score
        output["answer_count"] = actual_answer_count
        output["expected_answer_count"] = expected_answer_count
        output["answer_count_difference"] = answer_count_difference

        return output

    except Exception as e:
        logging.error(f"Error processing pure LLM example: {e}")
        return None

def calculate_model_scores(results: dict) -> dict:
    """Calculate overall model scores using balanced evaluation"""
    raw_scores = []
    balanced_scores = []
    answer_counts = []
    answer_count_differences = []
    
    for example_output in results.values():
        if "raw_average_score" in example_output:
            raw_scores.append(example_output["raw_average_score"])
        if "balanced_average_score" in example_output:
            balanced_scores.append(example_output["balanced_average_score"])
        if "answer_count" in example_output:
            answer_counts.append(example_output["answer_count"])
        if "answer_count_difference" in example_output:
            answer_count_differences.append(example_output["answer_count_difference"])
    
    return {
        "average_raw_score": mean(raw_scores) if raw_scores else 0,
        "average_balanced_score": mean(balanced_scores) if balanced_scores else 0,
        "average_answer_count": mean(answer_counts) if answer_counts else 0,
        "average_answer_count_difference": mean(answer_count_differences) if answer_count_differences else 0,
        "average_citation_score": 0.0  # Pure LLM has no citations
    }

def main():
    # Load dataset
    with open("cleaned_mcaqa.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        if not dataset:
            logging.error("Failed to load dataset")
            return

    # Get model list - only process mistral and gemma2
    models = get_models()
    target_models = ["mistral", "gemma2"]
    models = [m for m in models if any(target in m.name.lower() for target in target_models)]
    
    # Store scores for all models
    all_model_scores = {}
    
    # Generate answers for each model using pure LLM
    for model in models:
        logging.info(f"\nProcessing pure LLM balanced experiment for model: {model.name}")
        
        # Load or create results file
        results = load_or_create_results(model.name)
        
        # Process each example (limit to 500 for testing)
        progress = tqdm(dataset[:500])
        for example in progress:
            example_id = example["id"]
            
            # Skip already processed examples
            if example_id in results:
                continue
            
            # Process current example with balanced evaluation
            output = process_single_example_pure_llm_balanced(model, example)
            if output:
                results[example_id] = output
                # Save results in real-time
                save_results(results, model.name)
            
            # Update progress bar
            progress.set_description(
                f"Pure LLM Balanced - Model: {model.name}, "
                f"Processed: {len(results)}"
            )
        
        # Calculate scores for this model
        model_scores = calculate_model_scores(results)
        all_model_scores[model.name] = model_scores
        
        logging.info(f"Pure LLM balanced model {model.name} processing completed")
        logging.info(f"Average raw score: {model_scores['average_raw_score']:.4f}")
        logging.info(f"Average balanced score: {model_scores['average_balanced_score']:.4f}")
        logging.info(f"Average answer count difference: {model_scores['average_answer_count_difference']:.2f}")
    
    # Save scores for all models
    with open("pure_llm_balanced_model_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_model_scores, f, ensure_ascii=False, indent=2)
    logging.info("\nAll pure LLM balanced model evaluations completed")

if __name__ == "__main__":
    main()