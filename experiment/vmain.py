import json
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from models import get_models

def setup_logging():
    """Configure logging settings"""
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "analysis.log")),
            logging.StreamHandler(),
        ],
    )

def load_json(filepath: str) -> Dict:
    """Load content from a JSON file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File {filepath} not found.")
        return {}
    except json.JSONDecodeError:
        logging.error(f"File {filepath} is not valid JSON format.")
        return {}

def analyze_model_results(models: List, results_dir: str = 'intermediate_results') -> pd.DataFrame:
    """Analyze results for each metric"""
    analysis_data = []

    for model in models:
        filepath = os.path.join(results_dir, f"intermediate_results_{model.name}_main_experiment.json")
        model_results = load_json(filepath)
        if not model_results or model.name not in model_results:
            logging.warning(f"Results for model {model.name} are missing or invalid.")
            continue

        for example in model_results[model.name]:
            evaluations = example.get("evaluations", [])
            if not evaluations:
                continue
            
            for eval in evaluations:
                analysis_data.append({
                    'Model': model.name,
                    'Example ID': example.get('id'),
                    'Condition Score': eval.get('condition_evaluation', {}).get('score', 0),
                    'Answer Score': eval.get('answer_evaluation', {}).get('score', 0),
                    'Citation Score': eval.get('citation_evaluation', {}).get('score', 0),
                    'Answer Count': len(example.get("conditions", []))
                })

    return pd.DataFrame(analysis_data)

def setup_plot_style():
    """Set up consistent plot styling with custom colors"""
    plt.style.use('default')
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    # 定义一组亮度相近但色调不同的颜色
    return {
        'model_colors': {
            'model1': '#6FC087',    # 绿色
            'model2': '#6FB5E2',    # 蓝色
            'model3': '#E27A6F',    # 红色
            'model4': '#DFA76F',    # 橙色
            'model5': '#916FE2',    # 紫色
        }
    }

def plot_combined_distributions(df: pd.DataFrame, metrics: List[str], output_dir: str):
    """Plot distribution curves for all metrics in a single figure"""
    colors = setup_plot_style()['model_colors']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Score Distributions by Metric', fontsize=14, y=1.02)
    
    # Only plot Condition, Answer, and Citation scores (exclude Answer Count)
    plot_metrics = [m for m in metrics if m != 'Answer Count']
    
    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        for i, model in enumerate(df['Model'].unique()):
            model_data = df[df['Model'] == model][metric]
            # 循环使用颜色
            color = list(colors.values())[i % len(colors)]
            sns.kdeplot(
                data=model_data,
                label=model,
                ax=ax,
                color=color
            )
        
        ax.set_title(metric)
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        
        # 将图例放在子图内部的右上角
        ax.legend(loc='upper right',      # 放在右上角
                 fontsize='small',        # 较小的字体
                 framealpha=0.8)          # 半透明背景
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'combined_distributions.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_bars(df: pd.DataFrame, metric: str, output_dir: str):
    """Plot bar chart for a metric with custom colors"""
    colors = setup_plot_style()['model_colors']
    
    # Calculate mean and standard error
    stats = df.groupby('Model')[metric].agg(['mean', 'std']).round(3)
    stats['se'] = stats['std'] / np.sqrt(df.groupby('Model').size())
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Create bar plot with different colors for each model
    bars = plt.bar(stats.index, stats['mean'], 
                  yerr=stats['se'], 
                  capsize=5,
                  color=[list(colors.values())[i % len(colors)] for i in range(len(stats.index))])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title(f'Average {metric} by Model')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric.lower().replace(" ", "_")}_bars.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics
    stats.to_csv(os.path.join(output_dir, f'{metric.lower().replace(" ", "_")}_stats.csv'))
def main():
    """Main function to run analysis and visualization"""
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    setup_logging()
    logging.info("Starting metric analysis...")
    
    models = get_models()
    if not models:
        logging.error("No models loaded. Please check get_models function.")
        return

    # Get results data
    results_df = analyze_model_results(models)
    if results_df.empty:
        logging.error("No results data available for analysis.")
        return

    # Define metrics
    metrics = ['Condition Score', 'Answer Score', 'Citation Score', 'Answer Count']
    
    # Plot combined distribution curves
    plot_combined_distributions(results_df, metrics, output_dir)
    
    # Plot individual bar charts
    for metric in metrics:
        logging.info(f"Analyzing {metric}...")
        plot_score_bars(results_df, metric, output_dir)
    
    logging.info("Analysis completed. Results saved in 'analysis_results' directory.")

if __name__ == "__main__":
    main()
