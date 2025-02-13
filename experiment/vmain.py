import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from models import get_models
import seaborn as sns
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
        """Plot distribution curves for all metrics with enlarged fonts and adjust to vertical layout"""
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
        
        # Increase figure size for better readability and adjust layout to vertical orientation
        fig, axes = plt.subplots(3, 1, figsize=(12, 18.5))  # Adjusted to vertical layout (3 rows, 1 column)
        
        # Increase main title size and adjust position
        fig.suptitle('Score Distributions by Metric', fontsize=20, y=0.95)
        
        plot_metrics = [m for m in metrics if m != 'Answer Count']
        
        for idx, metric in enumerate(plot_metrics):
            ax = axes[idx]
            
            for model in self.results_df['Model'].unique():
                model_data = self.results_df[self.results_df['Model'] == model][metric]
                
                # Set line style to dashed for 'glm-4-plus' and 'gpt-4o'
                linestyle = 'dashed' if model in ['glm-4-plus', 'gpt-4o'] else 'solid'
                
                sns.kdeplot(
                    data=model_data,
                    label=model,
                    ax=ax,
                    color=model_colors[model],
                    linewidth=2,
                    linestyle=linestyle  # Apply dashed line style for the specified models
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
            ax.set_ylim(bottom=0)  # Force density plot to start from 0
        
        # Adjust layout with more space
        plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.5)
        
        # Save plot with high resolution
        save_path = self.output_dir / 'combined_distributions_vertical.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_combined_bars(self, metrics: List[str]):
        """
        Plot a combined bar chart for all metrics showing mean values with error bars.

        Args:
            metrics (List[str]): List of metric names to plot.
        """
        if self.results_df is None:
            logging.error("No data loaded. Please load data first.")
            return

        # 将指标分为左轴指标与右轴指标
        left_metrics = [m for m in metrics if m != "Answer Count"]
        right_metric = "Answer Count"

        # 获取所有模型（保持顺序一致）
        models = sorted(self.results_df["Model"].unique())
        x = np.arange(len(models))  # 位置
        bar_width = 0.18  # 设置柱状图宽度
        # 对于四个柱状图，采用对称偏移：
        offsets = {
            "Condition Score": -1.5 * bar_width,
            "Answer Score": -0.5 * bar_width,
            "Citation Score": 0.5 * bar_width,
            "Answer Count": 1.5 * bar_width
        }

        # 定义颜色（左侧指标各自颜色，右侧指标单独一种颜色）
        colors_left = {
            "Condition Score": "#2ecc71", 
            "Answer Score": "#3498db", 
            "Citation Score": "#e74c3c"
        }
        color_right = {"Answer Count": "#f39c12"}

        # 创建图表和双坐标轴
        fig, ax1 = plt.subplots(figsize=(16, 12))  # Increased the figure size further
        ax2 = ax1.twinx()

        # 绘制左轴指标的柱状图
        for metric in left_metrics:
            # 按模型分组计算均值和标准差
            grouped = self.results_df.groupby("Model")[metric].agg(["mean", "std"]).reindex(models)
            positions = x + offsets[metric]  # 根据偏移量调整每个柱的位置
            bars = ax1.bar(positions, grouped["mean"], bar_width,
                        color=colors_left[metric], alpha=0.8, label=metric, capsize=5)

            # 为柱状图添加值标签
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, height,  # Increased vertical spacing
                        f'{height:.2f}', ha='center', va='bottom', fontsize=16)

        # 绘制右轴指标“Answer Count”
        grouped_rc = self.results_df.groupby("Model")[right_metric].agg(["mean", "std"]).reindex(models)
        positions_rc = x + offsets[right_metric]
        bars_rc = ax2.bar(positions_rc, grouped_rc["mean"], bar_width,
                        color=color_right[right_metric], alpha=0.8, label=right_metric, capsize=5)

        # 为“Answer Count”柱状图添加值标签
        for bar in bars_rc:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height,  # Increased vertical spacing
                    f'{height:.2f}', ha='center', va='bottom', fontsize=16)

        # 设置x轴刻度
        ax1.set_ylim(0, 0.4)
        ax2.set_ylim(0, 4)  
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=16, rotation=45, ha="right")  # Adjust x-axis labels
        ax1.set_xlabel("Model", fontsize=18)
        ax1.set_ylabel("Score", fontsize=18)
        ax2.set_ylabel("Answer Count", fontsize=18)
        ax1.set_title("Average Scores by Model and Metric", fontsize=20)

        # 合并图例（由于柱状图分别在左右轴上，需要手动合并图例）
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles1 + handles2
        all_labels = labels1 + labels2
        ax1.legend(all_handles, all_labels, loc='upper left', fontsize=16)

        # 调整图表布局，避免重叠
        plt.tight_layout(pad=5.0, rect=[0, 0, 1, 0.96])  # Increased padding and adjusted layout

        # 保存图表
        save_path = self.output_dir / 'combined_metrics_bars_larger_with_values_adjusted.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

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

        # 定义指标顺序
        metrics = ['Condition Score', 'Answer Score', 'Citation Score', 'Answer Count']

        # 生成图表
        self.plot_combined_distributions(metrics)
        self.plot_combined_bars(metrics)

        logging.info(f"Analysis completed. Results saved in {self.output_dir}")

def main():
    """Main execution function"""
    analyzer = MetricsAnalyzer(output_dir="analysis_resultsa")
    analyzer.run_analysis()

if __name__ == "__main__":
    main()