import json
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from models import get_models

class MetricsAnalyzer:
    def __init__(self, output_dir: str = "analysis_results"):
        """Initialize analyzer with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_df = None
        self.models = get_models()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "analysis.log"),
                logging.StreamHandler(),
            ],
        )

    def load_data(self, results_dir: str = 'intermediate_results') -> pd.DataFrame:
        """Load and process results data"""
        analysis_data = []
        results_dir = Path(results_dir)

        for model in self.models:
            filepath = results_dir / f"intermediate_results_{model.name}_main_experiment.json"
            model_results = self._load_json(filepath)
            
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

        self.results_df = pd.DataFrame(analysis_data)
        return self.results_df

    def _load_json(self, filepath: Path) -> Dict:
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

    def plot_combined_distributions(self, metrics: List[str]):
        """Plot distribution curves for all metrics with enlarged fonts"""
        if self.results_df is None:
            logging.error("No data loaded. Please load data first.")
            return
            
        # Color scheme for different models
        model_colors = {
            'gemma2': '#2ecc71',    # Green
            'gpt-4o': '#3498db',    # Blue 
            'glm4': '#e74c3c',      # Red
            'llama3.1': '#f1c40f',  # Yellow
            'mistral': '#9b59b6',   # Purple
            'qwen2.5': '#1abc9c',   # Teal
            'glm-4-plus': '#e67e22'    # Orange
        }
        
        # Increase figure size for better readability
        fig, axes = plt.subplots(1, 3, figsize=(22.5, 6))  # Increased from (15, 5)
        
        # Increase main title size and adjust position
        fig.suptitle('Score Distributions by Metric', fontsize=20, y=1.05)
        
        plot_metrics = [m for m in metrics if m != 'Answer Count']
        
        for idx, metric in enumerate(plot_metrics):
            ax = axes[idx]
            for model in self.results_df['Model'].unique():
                model_data = self.results_df[self.results_df['Model'] == model][metric]
                sns.kdeplot(
                    data=model_data,
                    label=model,
                    ax=ax,
                    color=model_colors[model],
                    linewidth=2  # Increase line width for better visibility
                )
            
            # Increase font sizes for title and labels
            ax.set_title(metric, fontsize=16, pad=15)
            ax.set_xlabel('Score', fontsize=14, labelpad=10)
            ax.set_ylabel('Density', fontsize=14, labelpad=10)
            
            # Increase tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=12, pad=8)
            
            # Adjust legend
            ax.legend(
                loc='upper right', 
                fontsize=12,  # Increased legend font size
                framealpha=0.8,
                bbox_to_anchor=(1.0, 1.0)  # Slightly adjust legend position
            )
            
            # Add grid with adjusted style
            ax.grid(True, linestyle='--', alpha=0.2)
            
            # Optional: Adjust axis limits for better visualization
            ax.set_xlim(0, 1)
            # ax.set_xlim(left=ax.get_xlim()[0], right=ax.get_xlim()[1])
            ax.set_ylim(bottom=0)  # Force density plot to start from 0
        
        # Adjust layout with more space
        plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.5)
        
        # Save plot with high resolution
        save_path = self.output_dir / 'combined_distributions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_score_bars(self, metric: str):
        """
        Plot bar chart for a metric showing mean values with error bars
        
        Args:
            metric (str): Name of the metric column to plot
        """
        if self.results_df is None:
            logging.error("No data loaded. Please load data first.")
            return
            
        plt.figure(figsize=(7.5, 6))
        
        # Define color scheme for different models
        model_colors = {
            'gemma2': '#2ecc71',    # Green
            'gpt-4o': '#3498db',    # Blue 
            'glm4': '#e74c3c',      # Red
            'llama3.1': '#f1c40f',  # Yellow
            'mistral': '#9b59b6',   # Purple
            'qwen2.5': '#1abc9c',   # Teal
            'glm-4-plus': '#e67e22'    # Orange
        }
        
        # Create barplot using seaborn with fixed deprecation warning
        ax = sns.barplot(
            data=self.results_df,
            x='Model',
            y=metric,
            hue='Model',            # Assign hue to avoid deprecation
            legend=False,           # Hide redundant legend
            estimator=np.mean,
            errorbar=('ci', 68),    # Equivalent to standard error
            palette=model_colors,
            alpha=0.9,
            width=0.6 
        )
        
        # Set title and labels
        ax.set_title(f'Average {metric} by Model', fontsize=16, pad=15)
        ax.set_xlabel('Model', fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        
        # Configure tick labels
        ax.tick_params(axis='x', rotation=45,labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.2, color='gray')
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        for i in ax.containers:
            ax.bar_label(i, fmt='%.2f', fontsize=14)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f'{metric.lower().replace(" ", "_")}_bars.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics
        stats = self.results_df.groupby('Model')[metric].agg(['mean', 'std']).round(3)
        stats.to_csv(self.output_dir / f'{metric.lower().replace(" ", "_")}_stats.csv')
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        logging.info("Starting metric analysis...")
        
        if not self.models:
            logging.error("No models loaded. Please check get_models function.")
            return

        # Load and process data
        self.load_data()
        if self.results_df is None or self.results_df.empty:
            logging.error("No results data available for analysis.")
            return

        # Define metrics to analyze
        metrics = ['Condition Score', 'Answer Score', 'Citation Score', 'Answer Count']
        
        # Generate plots
        self.plot_combined_distributions(metrics)
        for metric in metrics:
            logging.info(f"Analyzing {metric}...")
            self.plot_score_bars(metric)
        
        logging.info(f"Analysis completed. Results saved in {self.output_dir}")

def main():
    """Main execution function"""
    analyzer = MetricsAnalyzer(output_dir="analysis_results")
    analyzer.run_analysis()

if __name__ == "__main__":
    main()