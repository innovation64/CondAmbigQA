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
    Construct prompt for R+C+AG approach (Retrieval + Conditions + Answer Generation),
    allowing up to three conditions to avoid hallucination.
    """
    ctx_text = "\n".join([f"Fragment {i+1} - {ctx['title']}:\n{ctx['text']}" for i, ctx in enumerate(ctxs)])
    prompt = (
        f"Question: {question}\n"
        f"Retrieved fragments:\n{ctx_text}\n\n"
        f"Please complete the following tasks:\n"
        f"1. Identify up to THREE key conditions related to the question based solely on the provided fragments.\n"
        f"2. For each condition, provide a corresponding detailed answer.\n"
        f"3. Cite the sources (fragment numbers) that support each condition and answer.\n"
        f"4. Output the results in JSON format with the following structure:\n"
        f"{{\n"
        f"  \"conditions\": [\n"
        f"    {{\n"
        f"      \"condition\": \"[Condition 1]\",\n"
        f"      \"answer\": \"[Detailed answer for condition 1]\",\n"
        f"      \"citations\": [\"Fragment X\", \"Fragment Y\"]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n\n"
        f"**Important instructions:**\n"
        f"- Respond using JSON format only.\n"
        f"- Generate a MAXIMUM of THREE conditions. It's acceptable to provide fewer if the fragments don't support three distinct conditions.\n"
        f"- Do not include empty or placeholder conditions.\n"
        f"- Ensure each condition and answer is substantive, directly related to the question, and explicitly supported by the given fragments.\n"
        f"- Provide detailed answers for each condition.\n"
        f"- Use actual content for conditions and answers; do not use placeholder text.\n"
        f"- Do not include any information or conditions that are not directly supported by the given fragments.\n"
        f"- Prioritize quality and accuracy over quantity. It's better to have fewer, well-supported conditions than to include speculative or weakly supported ones.\n\n"
        f"**Example Output with Two Conditions:**\n"
        f"{{\n"
        f"  \"conditions\": [\n"
        f"    {{\n"
        f"      \"condition\": \"First condition\",\n"
        f"      \"answer\": \"Detailed answer for the first condition.\",\n"
        f"      \"citations\": [\"Fragment 1\", \"Fragment 2\"]\n"
        f"    }},\n"
        f"    {{\n"
        f"      \"condition\": \"Second condition\",\n"
        f"      \"answer\": \"Detailed answer for the second condition.\",\n"
        f"      \"citations\": [\"Fragment 3\"]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n\n"
        f"**Example Output with One Condition:**\n"
        f"{{\n"
        f"  \"conditions\": [\n"
        f"    {{\n"
        f"      \"condition\": \"Only condition\",\n"
        f"      \"answer\": \"Detailed answer for the only condition.\",\n"
        f"      \"citations\": [\"Fragment 4\", \"Fragment 5\"]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )
    return prompt

def construct_modified_prompt(question: str, ctxs: list, properties: list) -> str:
    """构建简化的JSON输出格式prompt"""
    # 构建上下文文本
    ctx_text = "\n".join([
        f"Fragment {i+1} - {ctx['title']}:\n{ctx['text']}" 
        for i, ctx in enumerate(ctxs)
    ])
    
    # 获取条件列表
    conditions_text = "\n".join([
        f"Condition {i+1}: {prop['condition']}"
        for i, prop in enumerate(properties)
    ])
    
    prompt = (
        f"Question: {question}\n\n"
        f"Context fragments:\n{ctx_text}\n\n"
        f"Conditions to address:\n{conditions_text}\n\n"
        f"IMPORTANT: Respond with ONLY the following JSON format, no other text:\n"
        f"{{\n"
        f"  \"conditions\": [\n"
        f"    {{\n"
        f"      \"condition\": [\"Condition X\"]\n"
        f"      \"answer\": \"[Detailed answer]\",\n"
        f"      \"citations\": [\"Fragment X\", \"Fragment Y\"]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n\n"
        f"Notes:\n"
        f"- Output MUST be valid JSON only\n"
        f"- Do NOT include any explanatory text outside the JSON\n"
        f"- Do NOT include asterisks or other formatting\n"
        f"- ONLY use Fragment X format for citations\n"
        f"- Each condition needs one answer and relevant citations\n"
    )
    return prompt

def construct_prompt_no_condition(question: str, ctxs: list) -> str:
    """
    Construct prompt for RAG without explicit condition generation,
    allowing up to three answers to ensure conciseness and support from fragments.
    """
    ctx_text = "\n".join([f"Fragment {i+1} - {ctx['title']}:\n{ctx['text']}" for i, ctx in enumerate(ctxs)])
    prompt = (
        f"Question: {question}\n"
        f"Retrieved fragments:\n{ctx_text}\n\n"
        f"Please complete the following tasks:\n"
        f"1. Answer the question based solely on the provided fragments.\n"
        f"2. Cite up to THREE sources (fragment numbers) that support your answer.\n"
        f"3. Ensure the answer is concise and directly addresses the question without unnecessary information.\n"
        f"4. Output the results in JSON format with the following structure:\n"
        f"{{\n"
        f"  \"answers\": [\n"
        f"    {{\n"
        f"      \"answer\": \"[Answer]\",\n"
        f"      \"citations\": [\"Fragment X\", \"Fragment Y\"]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n\n"
        f"**Important instructions:**\n"
        f"- Respond using JSON format only.\n"
        f"- Generate a MAXIMUM of THREE answers. It's acceptable to provide fewer if the fragments don't support three distinct answers.\n"
        f"- Do not include empty or placeholder answers.\n"
        f"- Ensure each answer is concise, directly related to the question, and explicitly supported by the given fragments.\n"
        f"- Use actual content for answers; do not use placeholder text.\n"
        f"- Do not include any information or answers that are not directly supported by the given fragments.\n"
        f"- Prioritize quality and accuracy over quantity. It's better to have fewer, well-supported answers than to include speculative or weakly supported ones.\n\n"
        f"**Example Output with Two Answers:**\n"
        f"{{\n"
        f"  \"answers\": [\n"
        f"    {{\n"
        f"      \"answer\": \"Your first answer here.\",\n"
        f"      \"citations\": [\"Fragment 1\", \"Fragment 2\"]\n"
        f"    }},\n"
        f"    {{\n"
        f"      \"answer\": \"Your second answer here.\",\n"
        f"      \"citations\": [\"Fragment 3\"]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n\n"
        f"**Example Output with One Answer:**\n"
        f"{{\n"
        f"  \"answers\": [\n"
        f"    {{\n"
        f"      \"answer\": \"Your only answer here.\",\n"
        f"      \"citations\": [\"Fragment 4\", \"Fragment 5\"]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
        f"**Respond using JSON**"
    )
    return prompt

def save_results(results: dict, model_name: str, experiment_type: str):
    """
    Save results to a JSON file, named by model name and experiment type.
    For main experiment, limit conditions to three. For ablation, save all answers.
    """
    if experiment_type == "main_experiment":
        # Enforce the limit of three conditions for main experiment
        for example in results[model_name]:
            if 'conditions' in example:
                example['conditions'] = example['conditions'][:3]
    elif experiment_type == "ablation_experiment":
        # Optionally, enforce a limit on the number of generated answers
        # Uncomment the following lines if you want to limit the answers
        # for example in results[model_name]:
        #     if 'generated_answers' in example:
        #         example['generated_answers'] = example['generated_answers'][:3]
        pass  # Currently, no limit

    filename = f"{model_name}_{experiment_type}_results.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results[model_name], f, ensure_ascii=False, indent=2)

def save_intermediate_results(results: dict, filename: str, experiment_type: str):
    """
    Save intermediate results to a JSON file.
    For main experiment, limit conditions to three. For ablation, save all answers.
    """

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)