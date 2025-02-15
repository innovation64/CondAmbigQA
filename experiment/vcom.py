import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

class AblationAnalyzer:
    def __init__(self, output_dir: str = "output"):
        self.condition_results = {}
        self.no_condition_results = {}
        self.main_results = {}
        self.condition_scores = {}
        self.no_condition_scores = {}
        self.main_scores = {}
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, data_dir: str = ".", main_results_dir: str = "intermediate_results"):
        """
        Load results from all experiments
        Args:
            data_dir: Directory containing ablation experiment results
            main_results_dir: Directory containing main experiment results
        """
        data_path = Path(data_dir)
        main_path = Path(main_results_dir)
        
        # Load ablation experiment scores
        with open(data_path / "model_scores.json", "r") as f:
            self.condition_scores = json.load(f)
        with open(data_path / "model_scores_ablation.json", "r") as f:
            self.no_condition_scores = json.load(f)
            
        # Process main experiment results
        for model_name in self.condition_scores.keys():
            # Load ablation results
            try:
                with open(data_path / f"results_{model_name}.json", "r") as f:
                    self.condition_results[model_name] = json.load(f)
                with open(data_path / f"results_{model_name}_ablation.json", "r") as f:
                    self.no_condition_results[model_name] = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Could not find ablation results for {model_name}")

            # Load main experiment results
            try:
                main_file = main_path / f"intermediate_results_{model_name}_main_experiment.json"
                with open(main_file, "r") as f:
                    main_data = json.load(f)
                    self.main_results[model_name] = main_data[model_name]
                    
                    # Calculate average scores for main experiment
                    answer_scores = []
                    citation_scores = []
                    
                    for example in main_data[model_name]:
                        for eval in example.get("evaluations", []):
                            answer_scores.append(eval.get("answer_evaluation", {}).get("score", 0))
                            citation_scores.append(eval.get("citation_evaluation", {}).get("score", 0))
                    
                    self.main_scores[model_name] = {
                        "average_answer_score": np.mean(answer_scores),
                        "average_citation_score": np.mean(citation_scores)
                    }
            except FileNotFoundError:
                print(f"Warning: Could not find main results for {model_name}")
                
    def prepare_comparison_data(self) -> pd.DataFrame:
        """Prepare data for comparison visualization"""
        data = []
        
        for model in self.condition_scores.keys():
            # Only add data if model exists in all experiments
            if model in self.main_scores:
                # With conditions
                data.append({
                    'model': model,
                    'setting': 'Answering with Ground Truth Conditions',
                    'answer_score': self.condition_scores[model]['average_answer_score'],
                    'citation_score': self.condition_scores[model]['average_citation_score']
                })
                
                # Without conditions
                data.append({
                    'model': model,
                    'setting': 'Answering without Conditions',
                    'answer_score': self.no_condition_scores[model]['average_answer_score'],
                    'citation_score': self.no_condition_scores[model]['average_citation_score']
                })
                
                # Main experiment
                data.append({
                    'model': model,
                    'setting': 'Answering with Model-generated Conditions',
                    'answer_score': self.main_scores[model]['average_answer_score'],
                    'citation_score': self.main_scores[model]['average_citation_score']
                })
            
        return pd.DataFrame(data)

    def plot_comparison(self):
        """Create comparison visualizations with classic style"""
        df = self.prepare_comparison_data()
        
        # Set figure style
        plt.style.use('classic')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))
        
        # Custom color palette (list of colors for different models)
        color_palette = sns.color_palette("Set2", n_colors=len(df['model'].unique()))  # Example Set2 palette
        
        # Reorder the settings to put Main Experiment in the middle
        df['setting'] = pd.Categorical(df['setting'], 
                                    categories=['Answering with Ground Truth Conditions', 'Answering with Model-generated Conditions', 'Answering without Conditions'],
                                    ordered=True)
        
        # Plot answer scores (first plot)
        g1 = sns.barplot(
            data=df,
            x='model',
            y='answer_score',
            hue='setting',
            ax=ax1,
            width=0.8,
            palette=color_palette,
            saturation=0.8,
            edgecolor='none'
        )
        
        # Customize answer score plot
        ax1.set_ylim(bottom=0, top=0.45)  
        g1.set_title('Answer Score Comparison', fontsize=24,pad=-20)
        g1.set_xlabel('Model', fontsize=12)
        g1.set_ylabel('Average Answer Score', fontsize=12)
        
        # Add value labels
        for container in g1.containers:
            g1.bar_label(container, fmt='%.2f', padding=3)
        
        # Rotate x labels
        plt.setp(g1.get_xticklabels(), ha='center',x = 1.05,fontsize=22)
        
        # Hide legend for the first plot
        g1.legend_.set_visible(False)
        
        # Plot citation scores (second plot)
        g2 = sns.barplot(
            data=df,
            x='model',
            y='citation_score',
            hue='setting',
            ax=ax2,
            width=0.8,
            palette=color_palette,
            saturation=0.8,
            edgecolor='none'
        )
        
        # Customize citation score plot
        ax2.set_ylim(bottom=0, top=0.3) 
        g2.set_title('Citation Score Comparison', fontsize=24,pad=-20)
        g2.set_xlabel('Model', fontsize=12)
        g2.set_ylabel('Average Citation Score', fontsize=12)
        g2.legend_.set_visible(False)
        # Add value labels
        for container in g2.containers:
            g2.bar_label(container, fmt='%.2f', padding=3)
        
        # Rotate x labels
        plt.setp(g2.get_xticklabels(), ha='center',x=1.05,fontsize=22)
        
        # Adjust legend for the second plot
        # g2.legend(title=None, bbox_to_anchor=(-1.011, 1.17), loc='upper left', ncol=3,fontsize=22)
        # Add a shared legend in the center
        handles, labels = g1.get_legend_handles_labels()  # Get handles and labels from the first plot
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 1.05),ncol=3, fontsize=22)
        
        # Adjust layout
        plt.tight_layout()
        
        # Set spines visible
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            # ax.grid()
            ax.grid(False)
        
        # Save plot as PDF
        save_path = self.output_dir / "experiment_comparison.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        return save_path

def main():
    # Create analyzer instance
    analyzer = AblationAnalyzer(output_dir="analysis_outputa")
    
    # Load data from both experiments
    analyzer.load_data(
        data_dir=".", 
        main_results_dir="intermediate_results"
    )
    
    # Generate and save visualization
    comparison_plot_path = analyzer.plot_comparison()
    
    print(f"Analysis completed. Visualization saved to: {comparison_plot_path}")

if __name__ == "__main__":
    main()