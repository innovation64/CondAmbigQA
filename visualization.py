# visualize.py

import matplotlib.pyplot as plt
import numpy as np
import json

def visualize_main_experiment(main_stats):
    models = list(main_stats.keys())
    metrics = ['total_examples', 'total_expected_conditions', 'total_generated_conditions', 
               'exact_match_count', 'over_generated_count', 'under_generated_count']
    
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Main Experiment Statistics', fontsize=16)
    
    for i, metric in enumerate(metrics):
        values = [main_stats[model][metric] for model in models]
        ax = axs[i // 3, i % 3]
        ax.bar(models, values)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xticklabels(models, rotation=45, ha='right')
        
    plt.tight_layout()
    plt.savefig('main_experiment_visualization.png')
    plt.close()

def visualize_answer_comparison(comparison):
    models = list(comparison.keys())
    conditional_means = [comparison[model]['conditional_mean'] for model in models]
    no_condition_means = [comparison[model]['no_condition_mean'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, conditional_means, width, label='Conditional')
    rects2 = ax.bar(x + width/2, no_condition_means, width, label='No Condition')
    
    ax.set_ylabel('Mean Answer Score')
    ax.set_title('Comparison of Answer Scores: Conditional vs No Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    fig.tight_layout()
    plt.savefig('answer_score_comparison_visualization.png')
    plt.close()

def generate_visualizations():
    # Load experiment statistics
    with open("experiment_statistics.json", "r") as f:
        stats = json.load(f)
    
    main_stats = stats["main_experiment"]
    comparison = stats["answer_score_comparison"]
    
    # Generate visualizations
    visualize_main_experiment(main_stats)
    print("Main experiment visualization saved as 'main_experiment_visualization.png'")
    
    visualize_answer_comparison(comparison)
    print("Answer score comparison visualization saved as 'answer_score_comparison_visualization.png'")

if __name__ == "__main__":
    generate_visualizations()