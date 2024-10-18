# utils.py

import re
import json

def clean_response(text: str) -> str:
    """
    Remove code block markers from the response.
    """
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```', '', text, flags=re.MULTILINE)
    return text.strip()

def construct_prompt(question: str, ctxs: list) -> str:
    """
    Construct prompt for R+C+AG approach (Retrieval + Conditions + Answer Generation).
    """
    ctx_text = "\n".join([f"Fragment {i+1}: {ctx['text']}" for i, ctx in enumerate(ctxs)])
    prompt = (
        f"Question: {question}\n"
        f"Retrieved fragments:\n{ctx_text}\n\n"
        f"Please complete the following tasks:\n"
        f"1. List all possible conditions.\n"
        f"2. Generate corresponding answers for each condition.\n"
        f"3. Cite the sources (fragment numbers) for all conditions and answers.\n"
        f"4. Output in JSON format as follows:\n"
        f"{{\n"
        f"  \"conditions\": [\n"
        f"    {{\n"
        f"      \"condition\": \"Condition 1\",\n"
        f"      \"answer\": \"Answer 1\",\n"
        f"      \"citations\": [\"Fragment 1\", \"Fragment 2\"]\n"
        f"    }},\n"
        f"    ...\n"
        f"  ]\n"
        f"}}\n\n"
        f"**Respond using JSON**"
    )
    return prompt

def construct_prompt_no_condition(question: str, ctxs: list) -> str:
    """
    Construct prompt for RAG without explicit condition generation.
    """
    ctx_text = "\n".join([f"Fragment {i+1}: {ctx['text']}" for i, ctx in enumerate(ctxs)])
    prompt = (
        f"Question: {question}\n"
        f"Retrieved fragments:\n{ctx_text}\n\n"
        f"Please answer the question based on the retrieved fragments. "
        f"Cite the sources (fragment numbers) for your answer.\n"
        f"Output in JSON format as follows:\n"
        f"{{\n"
        f"  \"answer\": \"Your answer here\",\n"
        f"  \"citations\": [\"Fragment 1\", \"Fragment 2\"]\n"
        f"}}\n\n"
        f"**Respond using JSON**"
    )
    return prompt

def save_results(results: dict, model_name: str, experiment_type: str):
    """
    Save results to a JSON file, named by model name and experiment type.
    """
    filename = f"{model_name}_{experiment_type}_results.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def save_intermediate_results(results: dict, filename: str):
    """
    Save intermediate results to a JSON file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)