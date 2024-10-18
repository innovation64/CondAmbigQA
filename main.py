# main.py

import json
import logging
from models import get_models
from utils import construct_prompt, construct_prompt_no_condition, save_results, save_intermediate_results
from evaluator import (
    evaluate_condition_correctness,
    evaluate_answer_correctness,
    evaluate_citation_correctness,
)
from tqdm import tqdm
import numpy as np
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)

def load_dataset(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_main_experiment(dataset, models):
    results = {model.name: [] for model in models}
    stats = {model.name: {
        "total_examples": 0,
        "total_expected_answers": 0,
        "total_generated_answers": 0,
        "answer_count_difference": 0
    } for model in models}
    
    for example in tqdm(dataset, desc="Processing examples (Main Experiment)"):
        question = example["question"]
        ctxs = example["ctxs"]
        expected_conditions = example.get("expected_conditions", [])
        
        prompt = construct_prompt(question, ctxs)
        
        for model in models:
            response = model.generate_response(prompt)
            
            if not response:
                logging.warning(f"No response from model {model.name} for example {example['id']}")
                continue
            
            actual_conditions = response.get("conditions", [])
            
            stats[model.name]["total_examples"] += 1
            stats[model.name]["total_expected_answers"] += len(expected_conditions)
            stats[model.name]["total_generated_answers"] += len(actual_conditions)
            stats[model.name]["answer_count_difference"] += len(actual_conditions) - len(expected_conditions)
            
            evaluations = []
            for idx, actual in enumerate(actual_conditions):
                expected = expected_conditions[idx] if idx < len(expected_conditions) else {}
                
                cond_eval = evaluate_condition_correctness(
                    input_text=prompt,
                    actual_condition=actual.get("condition", ""),
                    expected_condition=expected.get("condition", "")
                )
                
                ans_eval = evaluate_answer_correctness(
                    input_text=prompt,
                    actual_answer=actual.get("answer", ""),
                    expected_answer=expected.get("answer", "")
                )
                
                cit_eval = evaluate_citation_correctness(
                    input_text=prompt,
                    actual_output=actual.get("citations", []),
                    retrieval_context=[ctx['text'] for ctx in ctxs if 'text' in ctx]
                )
                
                evaluations.append({
                    "condition_evaluation": cond_eval,
                    "answer_evaluation": ans_eval,
                    "citation_evaluation": cit_eval
                })
            
            results[model.name].append({
                "id": example["id"],
                "question": question,
                "conditions": actual_conditions,
                "evaluations": evaluations,
                "answer_count_difference": len(actual_conditions) - len(expected_conditions)
            })
            
            save_intermediate_results(results, f"intermediate_results_{model.name}_main_experiment.json")
            logging.info(f"Saved intermediate results for model: {model.name}, Example ID: {example['id']}")
    
    return results, stats

def run_ablation_experiment(dataset, models):
    results = {model.name: [] for model in models}
    
    for example in tqdm(dataset, desc="Processing examples (Ablation Experiment)"):
        question = example["question"]
        ctxs = example["ctxs"]
        
        prompt = construct_prompt_no_condition(question, ctxs)
        
        for model in models:
            response = model.generate_response(prompt)
            
            if not response:
                logging.warning(f"No response from model {model.name} for example {example['id']}")
                continue
            
            answer = response.get("answer", "")
            citations = response.get("citations", [])
            
            # Evaluate answer and citations
            ans_eval = evaluate_answer_correctness(
                input_text=prompt,
                actual_answer=answer,
                expected_answer=""  # We don't have expected answers for this approach
            )
            
            cit_eval = evaluate_citation_correctness(
                input_text=prompt,
                actual_output=citations,
                retrieval_context=[ctx['text'] for ctx in ctxs if 'text' in ctx]
            )
            
            results[model.name].append({
                "id": example["id"],
                "question": question,
                "answer": answer,
                "citations": citations,
                "evaluations": {
                    "answer_evaluation": ans_eval,
                    "citation_evaluation": cit_eval
                }
            })
            
            save_intermediate_results(results, f"intermediate_results_{model.name}_ablation_experiment.json")
            logging.info(f"Saved intermediate results for model: {model.name}, Example ID: {example['id']}")
    
    return results

def compare_answer_scores(main_results, ablation_results):
    comparison = {}
    for model_name in main_results.keys():
        main_scores = []
        ablation_scores = []
        
        for example in main_results[model_name]:
            for evaluation in example['evaluations']:
                main_scores.append(evaluation['answer_evaluation']['score'])
        
        for example in ablation_results[model_name]:
            ablation_scores.append(example['evaluations']['answer_evaluation']['score'])
        
        comparison[model_name] = {
            "conditional_mean": np.mean(main_scores),
            "conditional_std": np.std(main_scores),
            "no_condition_mean": np.mean(ablation_scores),
            "no_condition_std": np.std(ablation_scores),
            "difference": np.mean(main_scores) - np.mean(ablation_scores)
        }
    
    return comparison

def main():
    dataset = load_dataset("mcaqa.json")
    models = get_models()
    
    print("Running Main Experiment (Conditional)")
    main_results, main_stats = run_main_experiment(dataset, models)
    
    for model_name, model_results in main_results.items():
        save_results(model_results, model_name, "main_experiment")
        print(f"Saved main experiment results for model: {model_name}")
    
    print("\n=== Main Experiment Statistics ===")
    for model_name, model_stats in main_stats.items():
        print(f"\nModel: {model_name}")
        print(f"Total Examples: {model_stats['total_examples']}")
        print(f"Total Expected Answers: {model_stats['total_expected_answers']}")
        print(f"Total Generated Answers: {model_stats['total_generated_answers']}")
        print(f"Answer Count Difference: {model_stats['answer_count_difference']}")
        avg_difference = model_stats['answer_count_difference'] / model_stats['total_examples']
        print(f"Average Answer Count Difference: {avg_difference:.2f}")
    
    print("\nRunning Ablation Experiment (No Condition)")
    ablation_results = run_ablation_experiment(dataset, models)
    
    for model_name, model_results in ablation_results.items():
        save_results(model_results, model_name, "ablation_experiment")
        print(f"Saved ablation experiment results for model: {model_name}")
    
    print("\nComparing Answer Scores")
    comparison = compare_answer_scores(main_results, ablation_results)
    
    print("\n=== Answer Score Comparison ===")
    for model_name, scores in comparison.items():
        print(f"\nModel: {model_name}")
        print(f"Conditional RAG - Mean: {scores['conditional_mean']:.4f}, Std: {scores['conditional_std']:.4f}")
        print(f"No Condition RAG - Mean: {scores['no_condition_mean']:.4f}, Std: {scores['no_condition_std']:.4f}")
        print(f"Difference (Conditional - No Condition): {scores['difference']:.4f}")
    
    # Save comparison results
    with open("answer_score_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print("\nSaved answer score comparison to answer_score_comparison.json")
    
    # Save overall statistics
    with open("experiment_statistics.json", "w", encoding="utf-8") as f:
        json.dump({
            "main_experiment": main_stats,
            "answer_score_comparison": comparison
        }, f, ensure_ascii=False, indent=2)
    print("\nSaved experiment statistics to experiment_statistics.json")
    
    # Run visualization script
    print("\nGenerating visualizations...")
    subprocess.run(["python", "visualize.py"])
    
    print("\nExperiments completed. Results saved to JSON files and visualizations generated.")

if __name__ == "__main__":
    main()