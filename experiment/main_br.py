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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment.log"),
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
) -> Tuple[Dict, bool]:
    """Process a single example by generating response"""
    try:
        question = example["question"]
        ctxs = example["ctxs"]
        example_id = example["id"]

        prompt = construct_prompt(question, ctxs)
        response = safe_generate_response(model, prompt)

        if not response:
            logging.warning(f"Model {model.name} gave no response for example {example_id}")
            return None, False

        result_entry = {
            "id": example_id,
            "question": question,
            "response": response,
            "fragment_mapping": fragment_mapping
        }

        return result_entry, True

    except Exception as e:
        logging.error(f"Error processing example: {str(e)}")
        return None, False

def run_main_experiment(dataset: List[Dict], models: List) -> None:
    """Run main experiment and save responses"""
    for model in models:
        intermediate_filepath = f"intermediate_results_{model.name}_experiment.json"
        
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
            )

            if success and result_entry:
                model_results.append(result_entry)
                results[model.name] = model_results
                save_dataset(results, intermediate_filepath)
                logging.info(f"Saved intermediate results for model {model.name}, example ID: {example['id']}")

def main():
    """Main function to run the experiment"""
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
    logging.info("Running main experiment...")
    run_main_experiment(dataset, models)
    logging.info("Experiment completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error running experiment: {str(e)}")
        raise