import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

class AblationAnalyzer:
    def __init__(self, output_dir: str = "output"):
        self.baseline_results = {}
        self.compare_results = {}
        self.intermediate_results = {}
        self.baseline_scores = {}
        self.compare_scores = {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, data_dir: str = ".", intermediate_dir: str = "intermediate_results"):
        """Load results from baseline and comparison experiments"""
        data_path = Path(data_dir)
        intermediate_path = Path(intermediate_dir)
        
        # Load scores
        with open(data_path / "model_scores.json", "r") as f:
            self.baseline_scores = json.load(f)
        with open(data_path / "model_scores_compare.json", "r") as f:
            self.compare_scores = json.load(f)
            
        # Model names from the baseline scores
        model_names = ['llama3.1', 'mistral', 'qwen2.5', 'gemma2', 'glm4']
            
        # Load detailed results for each model
        for model_name in model_names:
            try:
                # Load baseline and compare results
                baseline_file = f"results_{model_name}.json"
                compare_file = f"results_{model_name}_compare.json"
                intermediate_file = f"intermediate_results_{model_name}_main_experiment.json"
                
                with open(data_path / baseline_file, "r") as f:
                    self.baseline_results[model_name] = json.load(f)
                with open(data_path / compare_file, "r") as f:
                    self.compare_results[model_name] = json.load(f)
                
                # Try to load intermediate results if they exist
                try:
                    with open(intermediate_path / intermediate_file, "r") as f:
                        self.intermediate_results[model_name] = json.load(f)
                except FileNotFoundError:
                    print(f"Warning: Could not find intermediate results for {model_name}")
                    
            except FileNotFoundError as e:
                print(f"Warning: Could not find results for {model_name}: {e}")

    def prepare_comparison_data(self) -> pd.DataFrame:
        """Prepare data for comparison visualization"""
        data = []
        
        for model in self.baseline_scores.keys():
            # Baseline (Main Experiment)
            data.append({
                'model': model,
                'setting': 'Main Experiment',
                'answer_score': self.baseline_scores[model]['average_answer_score'],
                'citation_score': self.baseline_scores[model]['average_citation_score']
            })
            
            # Without Conditions
            data.append({
                'model': model,
                'setting': 'Without Conditions',
                'answer_score': self.compare_scores[model]['average_answer_score'],
                'citation_score': self.compare_scores[model]['average_citation_score']
            })
            
            # With Conditions (from intermediate results if available)
            if model in self.intermediate_results:
                intermediate_scores = self._calculate_intermediate_scores(model)
                data.append({
                    'model': model,
                    'setting': 'With Conditions',
                    'answer_score': intermediate_scores['answer_score'],
                    'citation_score': intermediate_scores['citation_score']
                })
            
        return pd.DataFrame(data)

    def _calculate_intermediate_scores(self, model_name: str) -> Dict:
        """Calculate scores from intermediate results"""
        results = self.intermediate_results.get(model_name, {})
        if not results:
            return {'answer_score': 0, 'citation_score': 0}
            
        answer_scores = []
        citation_scores = []
        
        # Extract scores based on the structure of intermediate results
        # You may need to adjust this based on the actual structure
        for example in results.values():
            if isinstance(example, dict):
                if 'answer_score' in example:
                    answer_scores.append(example['answer_score'])
                if 'citation_score' in example:
                    citation_scores.append(example['citation_score'])
                
        return {
            'answer_score': np.mean(answer_scores) if answer_scores else 0,
            'citation_score': np.mean(citation_scores) if citation_scores else 0
        }

    def plot_comparison(self):
        """Create comparison visualizations matching the paper figure style"""
        df = self.prepare_comparison_data()
        
        # Set figure size and style
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Custom colors matching the paper
        colors = ['#2ecc71', '#e74c3c', '#3498db']  # green, red, blue
        
        # Plot answer scores
        sns.barplot(data=df, x='model', y='answer_score', hue='setting', 
                   ax=ax1, palette=colors, width=0.8)
        ax1.set_title('Answer Score Comparison', fontsize=12)
        ax1.set_xlabel('Model', fontsize=10)
        ax1.set_ylabel('Average Answer Score', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Add value labels on the bars
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.2f', padding=3)
        
        # Plot citation scores
        sns.barplot(data=df, x='model', y='citation_score', hue='setting',
                   ax=ax2, palette=colors, width=0.8)
        ax2.set_title('Citation Score Comparison', fontsize=12)
        ax2.set_xlabel('Model', fontsize=10)
        ax2.set_ylabel('Average Citation Score', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Add value labels on the bars
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.2f', padding=3)
        
        # Adjust layout
        plt.suptitle('CondAmbigQA: Conditional Ambiguous Question Answering', 
                    fontsize=14, y=1.05)
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / "model_performance_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

def main():
    analyzer = AblationAnalyzer(output_dir="condambigqa_analysis")
    
    # Load data from both directories
    analyzer.load_data(
        data_dir=".", 
        intermediate_dir="intermediate_results"
    )
    
    # Generate visualization
    plot_path = analyzer.plot_comparison()
    print(f"Analysis completed. Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()