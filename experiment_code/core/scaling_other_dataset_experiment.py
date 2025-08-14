import json
import logging
import os
from models import get_models
from utils import  (
    construct_prompt,
    construct_prompt_no_condition,
)
from evaluator import (
    evaluate_condition_quality_no_ref,
    evaluate_answer_correctness,
    evaluate_citation_correctness,
)
from tqdm import tqdm
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

# Initialize OpenAI client
client = OpenAI()

def load_dataset(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_dataset(data: list, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def safe_generate_response(model, prompt):
    @retry(
        retry=retry_if_exception_type((APIConnectionError, APIError, RateLimitError)),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )
    def _call():
        return model.generate_response(prompt)
    return _call()

def run_direct_answer_experiment(dataset, models):
    for model in models:
        intermediate_filepath = f"intermediate_results_{model.name}_direct_only.json"
        results_filepath = f"results_{model.name}_direct_only.json"

        results = []
        if os.path.exists(intermediate_filepath):
            results = load_dataset(intermediate_filepath)
        processed_ids = set(res["sample_id"] for res in results if "sample_id" in res)

        for example in tqdm(dataset, desc=f"Running direct QA eval for {model.name}"):
            sample_id = example["sample_id"]
            if sample_id in processed_ids:
                continue

            question = example["question"]
            docs = example.get("docs", [])
            ctxs = [{"title": d.get("title", ""), "text": d.get("text", "")} for d in docs if d.get("text")][:20]
            prompt = construct_prompt_no_condition(question, ctxs)

            try:
                response = safe_generate_response(model, prompt)
                answers_list = response.get("answers", [])

                fragment_mapping = {f"Fragment {i+1}": ctx for i, ctx in enumerate(ctxs)}
                title_to_fragment = {ctx["title"]: frag_id for frag_id, ctx in fragment_mapping.items()}

                annotations = example.get("annotations", [])
                fallback_answer = example.get("answer", "")

                best_score = -1
                best_entry = None

                for cand in answers_list:
                    answer = cand.get("answer", "")
                    citations = cand.get("citations", [])

                    best_ans_score = -1
                    best_ans_eval = None
                    for annotation in annotations:
                        expected_answer = annotation.get("long_answer", fallback_answer)
                        ans_eval = evaluate_answer_correctness(
                            input_text=prompt,
                            actual_answer=answer,
                            expected_answer=expected_answer
                        )
                        if ans_eval.get("score", 0) > best_ans_score:
                            best_ans_score = ans_eval["score"]
                            best_ans_eval = ans_eval

                    final_ans_eval = best_ans_eval if best_ans_eval else evaluate_answer_correctness(
                        input_text=prompt,
                        actual_answer=answer,
                        expected_answer=fallback_answer
                    )

                    # citation evaluation
                    best_cit_score = -1
                    best_cit_eval = None
                    for annotation in annotations:
                        wikipages = [k.get("wikipage", "") for k in annotation.get("knowledge", [])]
                        expected_citations = [
                            frag_id for frag_id, ctx in fragment_mapping.items()
                            if ctx["title"] in wikipages
                        ]
                        cit_eval = evaluate_citation_correctness(
                            actual_output=citations,
                            expected_output=expected_citations,
                            fragment_mapping=fragment_mapping
                        )
                        if cit_eval["score"] > best_cit_score:
                            best_cit_score = cit_eval["score"]
                            best_cit_eval = cit_eval

                    final_cit_eval = best_cit_eval if best_cit_eval else {
                        "score": 0.0,
                        "reason": "No citation match found"
                    }

                    combo_score = final_ans_eval["score"] + final_cit_eval["score"]
                    if combo_score > best_score:
                        best_score = combo_score
                        best_entry = {
                            "sample_id": sample_id,
                            "question": question,
                            "answer": answer,
                            "evaluations": {
                                "answer_evaluation": final_ans_eval,
                                "citation_evaluation": final_cit_eval
                            }
                        }

                if best_entry:
                    results.append(best_entry)
                    save_dataset(results, intermediate_filepath)

            except Exception as e:
                logging.error(f"Error processing {sample_id}: {e}")

        save_dataset(results, results_filepath)
        logging.info(f"Saved final results for model {model.name} to {results_filepath}")




def run_condition_generation_experiment(dataset, models):
    for model in models:
        intermediate_filepath = f"intermediate_results_{model.name}_cond_only.json"
        results_filepath = f"results_{model.name}_cond_only.json"

        results = []
        if os.path.exists(intermediate_filepath):
            results = load_dataset(intermediate_filepath)
        processed_ids = set(res["sample_id"] for res in results if "sample_id" in res)

        for example in tqdm(dataset, desc=f"Running condition eval for {model.name}"):
            sample_id = example["sample_id"]
            if sample_id in processed_ids:
                continue

            question = example["question"]
            docs = example.get("docs", [])
            ctxs = [{"title": d.get("title", ""), "text": d.get("text", "")} for d in docs if d.get("text")][:20]
            prompt = construct_prompt(question, ctxs)

            try:
                response = safe_generate_response(model, prompt)
                conditions = response.get("conditions", [])
                fragment_mapping = {f"Fragment {i+1}": ctx for i, ctx in enumerate(ctxs)}

                annotations = example.get("annotations", [])
                fallback_answer = example.get("answer", "")

                # Build reverse mapping for citations
                title_to_fragment = {
                    ctx["title"]: frag_id for frag_id, ctx in fragment_mapping.items()
                }

                evaluations = []
                for cond in conditions:
                    condition_str = cond.get("condition", "")
                    answer_str = cond.get("answer", "")
                    citation_list = cond.get("citations", [])

                    cond_eval = evaluate_condition_quality_no_ref(
                        input_text=prompt,
                        actual_condition=condition_str
                    )

                    best_ans_score = -1
                    best_ans_eval = None
                    for annotation in annotations:
                        expected_answer = annotation.get("long_answer", fallback_answer)
                        ans_eval = evaluate_answer_correctness(
                            input_text=prompt,
                            actual_answer=answer_str,
                            expected_answer=expected_answer
                        )
                        if ans_eval.get("score", 0) > best_ans_score:
                            best_ans_score = ans_eval["score"]
                            best_ans_eval = ans_eval

                    final_ans_eval = best_ans_eval if best_ans_eval else evaluate_answer_correctness(
                        input_text=prompt,
                        actual_answer=answer_str,
                        expected_answer=fallback_answer
                    )

                    best_cit_score = -1
                    best_cit_eval = None
                    for annotation in annotations:
                        wikipages = [k.get("wikipage", "") for k in annotation.get("knowledge", [])]
                        expected_citations = [
                            frag_id for frag_id, ctx in fragment_mapping.items()
                            if ctx["title"] in wikipages
                        ]
                        cit_eval = evaluate_citation_correctness(
                            actual_output=citation_list,
                            expected_output=expected_citations,
                            fragment_mapping=fragment_mapping
                        )
                        if cit_eval["score"] > best_cit_score:
                            best_cit_score = cit_eval["score"]
                            best_cit_eval = cit_eval

                    final_cit_eval = best_cit_eval if best_cit_eval else {
                        "score": 0.0,
                        "reason": "No citation match found"
                    }

                    evaluations.append({
                        "condition_evaluation": cond_eval,
                        "answer_evaluation": final_ans_eval,
                        "citation_evaluation": final_cit_eval
                    })

                result_entry = {
                    "sample_id": sample_id,
                    "question": question,
                    "conditions": conditions,
                    "evaluations": evaluations
                }

                results.append(result_entry)
                save_dataset(results, intermediate_filepath)

            except Exception as e:
                logging.error(f"Error processing {sample_id}: {e}")

        save_dataset(results, results_filepath)
        logging.info(f"Saved final results for model {model.name} to {results_filepath}")


def main():
    dataset = load_dataset("asqa_eval_dpr_top100.json")
    models = get_models()
    # run_condition_generation_experiment(dataset, models)
    run_direct_answer_experiment(dataset, models)

if __name__ == "__main__":
    main()
