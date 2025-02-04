# ablation
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
        self.condition_scores = {}
        self.no_condition_scores = {}
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, data_dir: str = "."):
        """Load results from both experiments"""
        data_path = Path(data_dir)
        
        # Load scores
        with open(data_path / "model_scores.json", "r") as f:
            self.condition_scores = json.load(f)
        with open(data_path / "model_scores_ablation.json", "r") as f:
            self.no_condition_scores = json.load(f)
            
        # Load detailed results for each model
        for model_name in self.condition_scores.keys():
            try:
                with open(data_path / f"results_{model_name}.json", "r") as f:
                    self.condition_results[model_name] = json.load(f)
                with open(data_path / f"results_{model_name}_ablation.json", "r") as f:
                    self.no_condition_results[model_name] = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Could not find detailed results for {model_name}")

    def prepare_comparison_data(self) -> pd.DataFrame:
        """Prepare data for comparison visualization"""
        data = []
        
        for model in self.condition_scores.keys():
            # With conditions
            data.append({
                'model': model,
                'setting': 'With Conditions',
                'answer_score': self.condition_scores[model]['average_answer_score'],
                'citation_score': self.condition_scores[model]['average_citation_score']
            })
            
            # Without conditions
            data.append({
                'model': model,
                'setting': 'Without Conditions',
                'answer_score': self.no_condition_scores[model]['average_answer_score'],
                'citation_score': self.no_condition_scores[model]['average_citation_score']
            })
            
        return pd.DataFrame(data)

    def calculate_detailed_metrics(self) -> Dict:
        """Calculate detailed metrics for each model and setting"""
        metrics = {}
        
        for model_name in self.condition_scores.keys():
            metrics[model_name] = {
                'with_condition': self._calculate_metrics(self.condition_results.get(model_name, {})),
                'without_condition': self._calculate_metrics(self.no_condition_results.get(model_name, {}))
            }
            
        return metrics
    
    def _calculate_metrics(self, results: Dict) -> Dict:
        """Calculate detailed metrics for a single result set"""
        if not results:
            return {}
            
        answer_scores = []
        citation_scores = []
        
        for example in results.values():
            for eval_data in example.get('evaluations', []):
                answer_scores.append(eval_data.get('answer_score', 0))
                citation_scores.append(eval_data.get('citation_score', 0))
                
        return {
            'mean_answer_score': np.mean(answer_scores),
            'std_answer_score': np.std(answer_scores),
            'mean_citation_score': np.mean(citation_scores),
            'std_citation_score': np.std(citation_scores),
            'total_examples': len(results)
        }

    def plot_comparison(self):
        """Create comparison visualizations"""
        df = self.prepare_comparison_data()
        
        # Create figure with custom style
        plt.figure(figsize=(15, 6))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Set color palette
        colors = ['#2ecc71', '#3498db']
        
        # Plot answer scores
        sns.barplot(data=df, x='model', y='answer_score', hue='setting', ax=ax1, palette=colors)
        ax1.set_title('Answer Score Comparison', fontsize=12, pad=15)
        ax1.set_xlabel('Model', fontsize=10)
        ax1.set_ylabel('Average Answer Score', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot citation scores
        sns.barplot(data=df, x='model', y='citation_score', hue='setting', ax=ax2, palette=colors)
        ax2.set_title('Citation Score Comparison', fontsize=12, pad=15)
        ax2.set_xlabel('Model', fontsize=10)
        ax2.set_ylabel('Average Citation Score', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        save_path = self.output_dir / "ablation_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_detailed_metrics(self):
        """Create detailed metrics visualization with error bars"""
        metrics = self.calculate_detailed_metrics()
        
        # Prepare data for plotting
        plot_data = []
        for model, settings in metrics.items():
            for setting, metric_values in settings.items():
                if metric_values:  # Skip empty metrics
                    plot_data.append({
                        'model': model,
                        'setting': setting,
                        'answer_score': metric_values['mean_answer_score'],
                        'answer_std': metric_values['std_answer_score'],
                        'citation_score': metric_values['mean_citation_score'],
                        'citation_std': metric_values['std_citation_score']
                    })
        
        df = pd.DataFrame(plot_data)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Colors for different settings
        colors = ['#2ecc71', '#3498db']
        
        # Plot answer scores with error bars
        plt.subplot(2, 1, 1)
        for i, setting in enumerate(['with_condition', 'without_condition']):
            data = df[df['setting'] == setting]
            plt.errorbar(data['model'], data['answer_score'],
                        yerr=data['answer_std'],
                        fmt='o-', label=setting.replace('_', ' ').title(),
                        capsize=5, color=colors[i], markersize=8)
        
        plt.title('Answer Scores with Standard Deviation', fontsize=12, pad=15)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot citation scores with error bars
        plt.subplot(2, 1, 2)
        for i, setting in enumerate(['with_condition', 'without_condition']):
            data = df[df['setting'] == setting]
            plt.errorbar(data['model'], data['citation_score'],
                        yerr=data['citation_std'],
                        fmt='o-', label=setting.replace('_', ' ').title(),
                        capsize=5, color=colors[i], markersize=8)
        
        plt.title('Citation Scores with Standard Deviation', fontsize=12, pad=15)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        save_path = self.output_dir / "detailed_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def generate_report(self) -> Tuple[str, Path]:
        """Generate a detailed analysis report"""
        metrics = self.calculate_detailed_metrics()
        
        report = "Ablation Study Analysis Report\n"
        report += "=" * 30 + "\n\n"
        
        for model, settings in metrics.items():
            report += f"Model: {model}\n"
            report += "-" * 20 + "\n"
            
            for setting, values in settings.items():
                if values:
                    report += f"\n{setting.replace('_', ' ').title()}:\n"
                    report += f"Answer Score: {values['mean_answer_score']:.4f} (±{values['std_answer_score']:.4f})\n"
                    report += f"Citation Score: {values['mean_citation_score']:.4f} (±{values['std_citation_score']:.4f})\n"
                    report += f"Total Examples: {values['total_examples']}\n"
            
            report += "\n"
        
        save_path = self.output_dir / "ablation_analysis_report.txt"
        with open(save_path, "w") as f:
            f.write(report)
        
        return report, save_path

def main():
    # Create analyzer instance with output directory
    output_dir = "analysis_outputb"
    analyzer = AblationAnalyzer(output_dir=output_dir)
    
    # Load data
    data_dir = "."  # Specify your data directory here
    analyzer.load_data(data_dir)
    
    # Generate visualizations and get file paths
    comparison_plot_path = analyzer.plot_comparison()
    detailed_plot_path = analyzer.plot_detailed_metrics()
    
    # Generate and save report
    _, report_path = analyzer.generate_report()
    
    print(f"Analysis completed. All files saved to: {output_dir}")
    print(f"Generated files:")
    print(f"- {comparison_plot_path}")
    print(f"- {detailed_plot_path}")
    print(f"- {report_path}")

if __name__ == "__main__":
    main()