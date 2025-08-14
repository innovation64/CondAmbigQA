import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, List, Set, Tuple
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # Ensures text is editable in PDF
matplotlib.rcParams['ps.fonttype'] = 42   # Ensures text is editable in PDF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("visualizer.log"),
        logging.StreamHandler(),
    ],
)

class EnhancedVisualizer:
    """
    Creates visualizations that properly highlight the performance of top models
    with a scoring mechanism matching main experiment results
    """
    
    def __init__(self, 
                 data_dir: str = ".", 
                 output_dir: str = "visualizations"):
        """
        Initialize visualizer
        
        Args:
            data_dir: Directory containing data files
            output_dir: Directory for saving visualizations
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.models = []
        self.model_data = {}
        self.example_stats = {}  # For storing example-level statistics
        
        # For storing results
        self.model_stats = None
        self.df = None
        
        # Set high-quality plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'axes.labelpad': 10,
            'figure.dpi': 300,
        })
        
        # Model name standardization
        self.model_mapping = {
            'gpt-4o': 'gpt-4o',
            'glm-4-plus': 'glm4-plus',
            'glm4': 'glm4',
            'llama3.1': 'llama3.1',
            'mistral': 'mistral', 
            'gemma2': 'gemma2',
            'qwen2.5': 'qwen2.5',
            'deepseekr1': 'deepseek-r1'
        }
        
        # Set color palette for models - improved colors for better distinction
        self.model_colors = {
                'gemma2': '#2ecc71',    # Green
                'gpt-4o': '#3498db',    # Blue 
                'glm4': '#e74c3c',      # Red
                'llama3.1': '#f1c40f',  # Yellow
                'mistral': '#9b59b6',   # Purple
                'qwen2.5': '#1abc9c',   # Teal
                'glm4-plus': '#e67e22',
                'deepseek-r1': '#95a5a6'     # Orange
        }
        
        # Set which models to highlight (primary focus models)
        self.highlight_models = ['gpt-4o', 'glm4-plus']
        
        # Set API models for styling
        self.api_models = ['gpt-4o', 'glm4-plus']
        self.local_models = ['llama3.1', 'mistral', 'qwen2.5', 'gemma2', 'glm4','deepseek-r1']
        
        # Set line styles
        self.line_styles = {model: 'dashed' if model in self.api_models else 'solid' 
                          for model in self.model_colors.keys()}
        
        # Set line widths - thicker for highlighted models
        self.line_widths = {model: 3 if model in self.highlight_models else 2
                           for model in self.model_colors.keys()}
        
        # Reference expected answer count if available
        self.expected_answer_counts = {}  # Will be populated from data
        
        # Display settings
        self.show_raw_scores = True  # Show both raw and adjusted scores for comparison
        
        # Expected balanced scores for validation
        self.expected_balanced_scores = {}

    def load_data(self):
        """Load all data files"""
        # Find all model data files
        model_files = []
        
        for pattern in ["*_main.json", "*.json"]:
            files = list(self.data_dir.glob(pattern))
            if files:
                model_files.extend(files)
                break
        
        if not model_files:
            logging.error(f"No model data files found in {self.data_dir}")
            return False
        
        logging.info(f"Found {len(model_files)} model data files")
        
        # Process each file
        for file_path in model_files:
            try:
                file_stem = file_path.stem
                model_name = None
                
                # Try to match from the filename
                for orig_name, clean_name in self.model_mapping.items():
                    if orig_name in file_stem:
                        model_name = clean_name
                        break
                
                if not model_name:
                    # Skip files that don't match known models
                    continue
                
                # Load the data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different data formats
                examples = []
                if isinstance(data, dict):
                    # Try to find the model key
                    for key in data.keys():
                        if key in self.model_mapping.values() or key in self.model_mapping.keys():
                            if isinstance(data[key], list):
                                examples = data[key]
                                break
                    
                    # If no model key found, check if it's a direct list
                    if not examples and any(isinstance(data.get(k), list) for k in data.keys()):
                        for k, v in data.items():
                            if isinstance(v, list) and len(v) > 0:
                                examples = v
                                break
                                
                elif isinstance(data, list):
                    examples = data
                
                if not examples:
                    logging.warning(f"Could not extract examples from {file_path}")
                    continue
                
                self.model_data[model_name] = examples
                if model_name not in self.models:
                    self.models.append(model_name)
                
                logging.info(f"Loaded {len(examples)} examples for model {model_name}")
                
                # Extract expected answer counts
                for example in examples:
                    example_id = example.get('id')
                    if example_id:
                        answer_count_diff = example.get('answer_count_difference', 0)
                        self.expected_answer_counts[example_id] = {
                            'answer_count_difference': answer_count_diff
                        }
                
            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}")
        
        # Process the data into a flat format
        if self.models:
            self.process_data()
            return True
        else:
            logging.error("No valid model data loaded")
            return False

    def process_data(self):
        """
        Process the data into example-level statistics with score calculation
        that matches main experiment results
        """
        # Initialize storage for example-level stats
        self.example_stats = {}
        
        # First, collect stats for each example and model
        for model, examples in self.model_data.items():
            for example in examples:
                example_id = example.get('id')
                if not example_id:
                    continue
                
                conditions = example.get('conditions', [])
                evaluations = example.get('evaluations', [])
                
                if not evaluations or not conditions:
                    continue
                
                # This is how many conditions/answers this model generated for this example
                actual_answer_count = len(conditions)
                
                # Get the answer count difference
                answer_count_difference = example.get('answer_count_difference', 0)
                
                # We can calculate the expected answer count
                expected_answer_count = actual_answer_count - answer_count_difference
                
                # Calculate raw average scores
                condition_scores = []
                answer_scores = []
                citation_scores = []
                
                for eval_data in evaluations:
                    # Ensure proper extraction of scores
                    if 'condition_evaluation' in eval_data and 'score' in eval_data['condition_evaluation']:
                        condition_scores.append(eval_data['condition_evaluation']['score'])
                    
                    if 'answer_evaluation' in eval_data and 'score' in eval_data['answer_evaluation']:
                        answer_scores.append(eval_data['answer_evaluation']['score'])
                    
                    if 'citation_evaluation' in eval_data and 'score' in eval_data['citation_evaluation']:
                        citation_scores.append(eval_data['citation_evaluation']['score'])
                
                avg_condition_score = np.mean(condition_scores) if condition_scores else 0
                avg_answer_score = np.mean(answer_scores) if answer_scores else 0
                avg_citation_score = np.mean(citation_scores) if citation_scores else 0
                
                # Get average citation count
                citation_counts = [len(condition.get('citations', [])) for condition in conditions]
                avg_citation_count = np.mean(citation_counts) if citation_counts else 0
                
                # Calculate adjusted scores with method that matches main experiment results
                balanced_condition_score, balanced_answer_score, balanced_citation_score = self.calculate_balanced_scores(
                    avg_condition_score, avg_answer_score, avg_citation_score, 
                    answer_count_difference, expected_answer_count,
                    avg_citation_count, model
                )
                
                # Store example stats
                if example_id not in self.example_stats:
                    self.example_stats[example_id] = {}
                
                self.example_stats[example_id][model] = {
                    'condition_score': avg_condition_score,
                    'answer_score': avg_answer_score,
                    'citation_score': avg_citation_score,
                    'balanced_condition_score': balanced_condition_score,
                    'balanced_answer_score': balanced_answer_score,
                    'balanced_citation_score': balanced_citation_score,
                    'citation_count': avg_citation_count,
                    'answer_count': actual_answer_count,
                    'expected_answer_count': expected_answer_count,
                    'answer_count_difference': answer_count_difference
                }
        
        # Calculate model-level statistics
        self.calculate_model_stats()
        
        logging.info(f"Processed {len(self.example_stats)} examples across {len(self.models)} models")

    def calculate_balanced_scores(self, 
                           condition_score: float, 
                           answer_score: float, 
                           citation_score: float,
                           answer_count_difference: int,
                           expected_answer_count: int,
                           citation_count: float,
                           model: str) -> Tuple[float, float, float]:
        """
        Calculate balanced scores considering answer quantity, answer quality, and citation quality.
        
        Args:
            condition_score: Original condition score
            answer_score: Original answer score
            citation_score: Original citation score
            answer_count_difference: Difference between actual and expected answers
            expected_answer_count: Number of expected answers
            citation_count: Average number of citations per answer
            model: Model name
            
        Returns:
            Tuple of balanced scores (condition, answer, citation)
        """
        # Constants definition, improving readability and making adjustments easier
        HIGH_QUALITY_THRESHOLD = 0.5
        MAX_CITATION_BOOST_COUNT = 3
        CITATION_BOOST_FACTOR = 0.3
        SINGLE_ANSWER_PENALTY = 0.9
        MISSING_ANSWER_MAX_PENALTY = 0.2
        EXTRA_ANSWER_MAX_PENALTY = 0.15
        
        # Model-specific parameter configuration
        model_config = {
            "default": {
                "quality_weight": 0.6,      # Weight of quality factor
                "quantity_weight": 0.4,     # Weight of quantity factor
                "citation_weight": 0.5,     # Weight of citation factor
                "missing_severity": 0.6,    # Severity of missing answers
                "extra_severity": 0.3       # Severity of extra answers
            }
            # Can add specific configurations for different models
            # "gpt-4o": { ... }
        }
        
        # Get configuration for current model, use default if no specific configuration exists
        config = model_config.get(model, model_config["default"])
        
        # Prevent division by zero error
        safe_expected_count = max(expected_answer_count, 1)
        
        # Calculate actual answer count
        actual_answer_count = safe_expected_count + answer_count_difference
        
        # 1. Quality factor calculation
        # Base quality score: weighted average of condition score and answer score
        base_quality = (condition_score + answer_score) / 2
        
        # Citation quality: boost citation score based on citation count
        citation_quality = citation_score * (1 + CITATION_BOOST_FACTOR * min(citation_count, MAX_CITATION_BOOST_COUNT))
        
        # Comprehensive quality score: combine base quality and citation quality
        quality_score = (base_quality * (1 - config["citation_weight"]) + 
                        citation_quality * config["citation_weight"])
        
        # 2. Quantity adjustment calculation
        quantity_adjustment = 1.0  # Default no adjustment
        
        if actual_answer_count == 1 and safe_expected_count > 1:
            # Special case for single answer
            quantity_adjustment = 1.0 - SINGLE_ANSWER_PENALTY
        elif answer_count_difference < 0:
            # Case for missing answers
            missing_ratio = abs(answer_count_difference) / safe_expected_count
            
            # Adjust missing answer penalty based on answer quality: higher quality answers receive lighter penalties
            quality_factor = max(0.5, min(1.0, answer_score / HIGH_QUALITY_THRESHOLD))
            penalty = min(missing_ratio * config["missing_severity"] * (2 - quality_factor), 
                        MISSING_ANSWER_MAX_PENALTY)
            
            quantity_adjustment = 1.0 - penalty
        elif answer_count_difference > 0:
            # Case for extra answers
            extra_ratio = answer_count_difference / safe_expected_count
            
            # Adjust extra answer penalty based on answer quality: higher quality extra answers receive lighter penalties
            quality_factor = max(0.5, min(1.0, answer_score / HIGH_QUALITY_THRESHOLD))
            
            if extra_ratio > 1.0:  # Significantly exceeds expectations
                penalty = min(extra_ratio * config["extra_severity"], EXTRA_ANSWER_MAX_PENALTY)
            else:
                penalty = min(extra_ratio * config["extra_severity"] * 0.5, EXTRA_ANSWER_MAX_PENALTY * 0.5)
                
            # High-quality answers reduce the penalty for extra answers
            penalty = penalty * (2 - quality_factor)
            
            quantity_adjustment = 1.0 - penalty
        
        # 3. Calculate final balanced score
        # Weighted combination of quality and quantity factors
        quality_component = quality_score * config["quality_weight"]
        quantity_component = quantity_adjustment * config["quantity_weight"]
        
        # Combined scoring factor
        combined_factor = quality_component + quantity_component
        
        # Calculate final balanced scores
        balanced_condition_score = condition_score * combined_factor
        balanced_answer_score = answer_score * combined_factor
        balanced_citation_score = citation_quality * combined_factor
        
        # Ensure scores are within valid range [0, 1]
        balanced_condition_score = min(max(balanced_condition_score, 0), 1)
        balanced_answer_score = min(max(balanced_answer_score, 0), 1)
        balanced_citation_score = min(max(balanced_citation_score, 0), 1)
        
        # Log debug information
        logging.debug(
            f"Model {model}, Original: C={condition_score:.3f}, A={answer_score:.3f}, Cit={citation_score:.3f} | "
            f"Quality={quality_score:.3f}, Quantity={quantity_adjustment:.3f}, Combined={combined_factor:.3f} | "
            f"Balanced: C={balanced_condition_score:.3f}, A={balanced_answer_score:.3f}, Cit={balanced_citation_score:.3f} | "
            f"Diff={answer_count_difference}, CitCnt={citation_count:.2f}"
        )
        
        return balanced_condition_score, balanced_answer_score, balanced_citation_score
    def calculate_model_stats(self):
        """Calculate statistics for each model with balanced scores"""
        if not self.example_stats:
            logging.error("No example stats available, cannot calculate model statistics")
            return
        
        # Prepare data for DataFrame
        rows = []
        
        for example_id, model_stats in self.example_stats.items():
            for model, stats in model_stats.items():
                rows.append({
                    'model': model,
                    'example_id': example_id,
                    'condition_score': stats['condition_score'],
                    'answer_score': stats['answer_score'],
                    'citation_score': stats['citation_score'],
                    'balanced_condition_score': stats['balanced_condition_score'],
                    'balanced_answer_score': stats['balanced_answer_score'],
                    'balanced_citation_score': stats['balanced_citation_score'],
                    'citation_count': stats['citation_count'],
                    'answer_count': stats['answer_count'],
                    'expected_answer_count': stats['expected_answer_count'],
                    'answer_count_difference': stats['answer_count_difference']
                })
        
        # Create DataFrame
        self.df = pd.DataFrame(rows)
        
        # Calculate combined scores
        self.df['raw_combined_score'] = (self.df['condition_score'] + 
                                        self.df['answer_score'] + 
                                        self.df['citation_score']) / 3
                                        
        self.df['balanced_combined_score'] = (self.df['balanced_condition_score'] + 
                                            self.df['balanced_answer_score'] + 
                                            self.df['balanced_citation_score']) / 3
        
        # Add model_type column (API or Local)
        self.df['model_type'] = self.df['model'].apply(
            lambda x: 'API' if x in self.api_models else 'Local'
        )
        
        # Group by model and calculate statistics
        model_stats = self.df.groupby('model').agg({
            'condition_score': ['mean', 'std'],
            'answer_score': ['mean', 'std'],
            'citation_score': ['mean', 'std'],
            'balanced_condition_score': ['mean', 'std'],
            'balanced_answer_score': ['mean', 'std'],
            'balanced_citation_score': ['mean', 'std'],
            'raw_combined_score': ['mean', 'std'],
            'balanced_combined_score': ['mean', 'std'],
            'citation_count': ['mean', 'std'],
            'answer_count': ['mean', 'std'],
            'expected_answer_count': ['mean', 'std'],
            'answer_count_difference': ['mean', 'std'],
            'model_type': 'first'
        })
        
        # Flatten MultiIndex columns
        model_stats.columns = ['_'.join(col).strip('_') for col in model_stats.columns]
        
        # Sort by balanced combined score (high to low)
        model_stats = model_stats.sort_values('balanced_combined_score_mean', ascending=False)
        self.model_stats = model_stats.reset_index()
        
        # Verify against expected values and log any discrepancies
        self.verify_balanced_scores()
        
        # Save model stats to CSV
        self.model_stats.to_csv(self.output_dir / 'model_stats.csv', index=False)
        
        # Sort models by performance for consistent ordering in visualizations
        self.models = self.model_stats['model'].tolist()
        
        logging.info(f"Calculated model statistics and saved to {self.output_dir / 'model_stats.csv'}")

    def verify_balanced_scores(self):
        """Verify balanced scores match expected values"""
        model_stats_dict = self.model_stats.set_index('model').to_dict()
        
        for model, expected in self.expected_balanced_scores.items():
            if model not in self.model_stats['model'].values:
                logging.warning(f"Model {model} not found in results, cannot verify scores")
                continue
                
            for metric, expected_value in expected.items():
                actual_value = model_stats_dict.get(f'{metric}_mean', {}).get(model, 0)
                diff = abs(actual_value - expected_value)
                
                if diff > 0.02:  # Allow 0.02 difference
                    logging.warning(f"Score mismatch for {model} {metric}: Expected {expected_value:.2f}, Got {actual_value:.2f}")
                else:
                    logging.info(f"Score match for {model} {metric}: {actual_value:.2f}")

    def create_score_distributions_pdf(self):
        """Create a PDF for score distributions using balanced scores"""
        if self.df is None:
            logging.error("No data available, cannot create score distributions PDF")
            return
        
        pdf_path = self.output_dir / 'score_distributions.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Create figure with subplots
            sns.set_style('white')
            fig, axes = plt.subplots(3, 1, figsize=(12, 18.5), height_ratios=[1, 1, 1.2])
            
            metrics = ['balanced_condition_score', 'balanced_answer_score', 'balanced_citation_score']
            metric_names = ['Condition Score', 'Answer Score', 'Citation Score']
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                ax = axes[i]
                ax.grid(False)
                ax.margins(y=0.1)
                # Plot each model
                for model in self.models:
                    model_data = self.df[self.df['model'] == model][metric]
                    
                    # Skip if no data for this model
                    if model_data.empty:
                        logging.warning(f"No data for model {model} on metric {metric}")
                        continue
                    
                    # Plot density
                    sns.kdeplot(
                        data=model_data,
                        ax=ax,
                        label=model,
                        color=self.model_colors.get(model, 'gray'),
                        linestyle=self.line_styles.get(model, 'solid'),
                        linewidth=self.line_widths.get(model, 2),
                        alpha=0.9 if model in self.highlight_models else 0.7
                    )
                
                # ax.set_title(name)
                ax.text(
                    0.5,0.95, name,
                    transform=ax.transAxes,
                    fontsize=24,
                    fontweight='bold',
                    va='top', ha='center'
                )
                ax.set_xlabel('Score', fontsize=18, labelpad=10)
                ax.set_ylabel('Density', fontsize=18, labelpad=10)
                ax.tick_params(axis='both', which='major', labelsize=16, pad=8)
                ax.set_xlim(0, 1)
                
                if i == 2:
                    handles, labels = ax.get_legend_handles_labels()

                    ax.legend(
                        handles, labels,
                        loc='upper right',
                        ncol=1,
                        framealpha=0.8,
                        fontsize=22,
                        bbox_to_anchor=(1.0, 1.0)
                    )
                else:
                    if ax.get_legend():
                        ax.get_legend().remove()
            
            plt.suptitle('Score Distributions by Metric (Balanced Scores)',  fontsize=26, y=0.95)
            fig.subplots_adjust(top=0.92, hspace=0.3)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save to PDF
            pdf.savefig(fig)
            plt.close()
        
        logging.info(f"Created score distributions PDF at {pdf_path}")
        
        # Also save as PNG for convenience
        plt.savefig(self.output_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
    def create_comprehensive_bar_chart(self):
        """Create a comprehensive bar chart with balanced scores and API vs Local comparison"""
        if self.model_stats is None:
            logging.error("No model stats available, cannot create model performance PDF")
            return

        pdf_path = self.output_dir / 'comprehensive_performance.pdf'
        with PdfPages(pdf_path) as pdf:
            # 0. Turn off background grid
            sns.set_style('white')
            plt.rcParams['axes.grid'] = False
            for ax in plt.gcf().axes:
                ax.grid(False)

            # 1. Create canvas and subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1.5, 1])
            for ax in (ax1, ax2):
                ax.grid(False)

            # 2. Data preparation
            models   = list(self.model_stats['model'])
            x        = np.arange(len(models))
            api_mask = [m in self.api_models for m in models]

            # 3. Color scheme & hatch patterns: API vs Local + highlighting
            base = {
                'cond': ['#1f77b4', '#2ca02c'],
                'ans':  ['#5798c9', '#5dba5d'],
                'cite': ['#9bbade', '#8ecc8e'],
            }
            cond_cols  = [base['cond'][0] if is_api else base['cond'][1] for is_api in api_mask]
            ans_cols   = [base['ans'][0]  if is_api else base['ans'][1]  for is_api in api_mask]
            cite_cols  = [base['cite'][0] if is_api else base['cite'][1] for is_api in api_mask]
            cond_hatch = ['//'] * len(models)
            ans_hatch  = ['\\\\'] * len(models)
            cite_hatch = ['..'] * len(models)

            # Highlighting
            for i, m in enumerate(models):
                if m in self.highlight_models:
                    if api_mask[i]:
                        # cond_cols[i], ans_cols[i], cite_cols[i] = '#0a4a8f', '#1a64a3', '#3b7eb8'
                        cond_cols[i], ans_cols[i], cite_cols[i] = '#4f7bb1', '#5f94c1', '#7da2cd'
                    else:
                        cond_cols[i], ans_cols[i], cite_cols[i] = '#166616', '#237b23', '#3a9c3a'

            # 4. Top subplot: three bars + combined line
            w = 0.2
            b1 = ax1.bar(x - w, self.model_stats['balanced_condition_score_mean'],
                        w, label='Condition Score',
                        color=cond_cols, edgecolor='black', hatch=cond_hatch)
            b2 = ax1.bar(x,     self.model_stats['balanced_answer_score_mean'],
                        w, label='Answer Score',
                        color=ans_cols,  edgecolor='black', hatch=ans_hatch)
            b3 = ax1.bar(x + w, self.model_stats['balanced_citation_score_mean'],
                        w, label='Citation Score',
                        color=cite_cols, edgecolor='black', hatch=cite_hatch)

            ax1_twin = ax1.twinx()
            line, = ax1_twin.plot(
                x, self.model_stats['balanced_combined_score_mean'],
                'o-', label='Combined Score',
                color='darkred', linewidth=2, markersize=8
            )

            # 5. Add value labels
            def add_bar_labels(bars, axis):
                for bar in bars:
                    h = bar.get_height()
                    axis.text(bar.get_x() + bar.get_width() / 2, h,
                            f'{h:.2f}', ha='center', va='bottom', fontsize=9)
            add_bar_labels(b1, ax1)
            add_bar_labels(b2, ax1)
            add_bar_labels(b3, ax1)

            # Combined score labels & leave more header space
            ax1_twin.margins(y=0.12)
            ymin, ymax = ax1_twin.get_ylim()
            offset = (ymax - ymin) * 0.02
            for i, v in enumerate(self.model_stats['balanced_combined_score_mean']):
                ax1_twin.text(
                    x[i], v + offset,
                    f'{v:.2f}', ha='center', va='bottom',
                    fontsize=9, color='darkred', weight='bold'
                )

            # 6. Increase font size for axes
            ax1.set_ylabel('Individual Scores', fontsize=18)
            ax1_twin.set_ylabel('Combined Score', fontsize=18, color='darkred')
            ax1.tick_params(axis='both', labelsize=16)
            ax1_twin.tick_params(axis='y', labelsize=16, colors='darkred')
            ax1.set_ylim(0, 1)
            ax1_twin.set_ylim(0, 0.8)

            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=0, ha='center', fontsize=16)
            ax1.set_title('Balanced Performance Scores on CondAmbigQA', fontsize=20)

            # 7. Merge legends and hide color blocks/lines
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax1_twin.get_legend_handles_labels()
            leg = ax1.legend(
                h1 + h2, l1 + l2,
                loc='upper right', frameon=False,
                fontsize=10, ncol=1
            )
            for lh in leg.legend_handles:
                if hasattr(lh, 'set_facecolor'): lh.set_facecolor('none')
                if hasattr(lh, 'set_edgecolor'): lh.set_edgecolor('black')
                # if hasattr(lh, 'set_linestyle'):  lh.set_linestyle('')

            # 8. Background distinction
            for i, is_api in enumerate(api_mask):
                bg = '#e6f2ff' if is_api else '#e6ffe6'
                ax1.axvspan(i - 0.4, i + 0.4, color=bg, alpha=0.3, zorder=-1)

            # 9. Bottom subplot: Answer Count & Difference with hatch
            w2 = 0.3
            x2 = np.arange(len(models))
            cnt_cols, diff_cols = [], []
            for i, is_api in enumerate(api_mask):
                cnt_cols.append('#aed6f1' if is_api else '#a3e4a3')
                dv = self.model_stats.iloc[i]['answer_count_difference_mean']
                diff_cols.append(
                    '#5dade2' if dv >= 0 and is_api else
                    '#58d258' if dv >= 0 else
                    '#2874a6' if is_api else
                    '#2e8b57'
                )
            for i, m in enumerate(models):
                if m in self.highlight_models:
                    cnt_cols[i]  = '#6baed6' if api_mask[i] else '#74c476'

            # hatch patterns for bottom bars
            cnt_hatch = ['//'] * len(models)
            diff_hatch = ['//'] * len(models)
            ax2.bar(
                x2 - w2/2,
                self.model_stats['answer_count_mean'],
                w2,
                color=cnt_cols,
                edgecolor='black',        # Black border
                linewidth=1.2,
                alpha=0.8,
                label='Answer Count'
            )

            # Overlay for hatch with orange
            bc = ax2.bar(
                x2 - w2/2,
                self.model_stats['answer_count_mean'],
                w2,
                color='none',
                edgecolor='orange',       # Hatch color
                hatch=cnt_hatch,
                linewidth=0.0,
                alpha=0.8
            )

            # Draw count difference bars with black border
            ax2.bar(
                x2 + w2/2,
                self.model_stats['answer_count_difference_mean'],
                w2,
                color=diff_cols,
                edgecolor='black',        # Black border
                linewidth=1.2,
                alpha=0.8,
                label='Count Difference'
            )

            # Overlay for hatch with red
            bd = ax2.bar(
                x2 + w2/2,
                self.model_stats['answer_count_difference_mean'],
                w2,
                color='none',
                edgecolor='red',          # Hatch color
                hatch=diff_hatch,
                linewidth=0.0,
                alpha=0.8
            )
            
     
            ax2.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

            # 10. Bottom value labels
            for bar in bc:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=9)
            for bar in bd:
                h = bar.get_height()
                va = 'bottom' if h >= 0 else 'top'
                off = 0.05 if h >= 0 else -0.15
                ax2.text(bar.get_x() + bar.get_width()/2, h + off,
                        f'{h:.2f}', ha='center', va=va, fontsize=9)

            ax2.set_ylabel('Count / Difference', fontsize=18)
            ax2.set_xticks(x2)
            ax2.set_xticklabels(models, rotation=0, ha='center', fontsize=12)
            ax2.tick_params(axis='both', labelsize=16)
            ax2.set_ylim(-1.5, 4)
            ax2.set_title('Answer Count Statistics by Model', fontsize=20)

            # 11. Bottom legend: hatch distinction
            legend_patches = [
                Patch(facecolor='white', edgecolor='orange', hatch='//', label='Answer Count'),
                Patch(facecolor='white', edgecolor='red', hatch='//', label='Count Difference'),
            ]
            ax2.legend(
                handles=legend_patches,
                loc='upper right',
                frameon=False,
                fontsize=10,
                ncol=1
            )

            # 12. Bottom background
            for i, is_api in enumerate(api_mask):
                bg = '#e6f2ff' if is_api else '#e6ffe6'
                ax2.axvspan(i - 0.4, i + 0.4, color=bg, alpha=0.3, zorder=-1)


            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            pdf.savefig(fig)
            plt.close(fig)

        logging.info(f"Created comprehensive performance PDF at {pdf_path}")
        plt.savefig(self.output_dir / 'comprehensive_performance.png',
                    dpi=300, bbox_inches='tight')

    def create_horizontal_ranking_chart(self):
        """Create a horizontal bar chart showing model ranking with clear visual differentiation"""
        if self.model_stats is None:
            logging.error("No model stats available, cannot create ranking chart")
            return
        
        pdf_path = self.output_dir / 'model_ranking.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort models by combined score for ranking display
            sorted_stats = self.model_stats.sort_values('balanced_combined_score_mean', ascending=True)
            
            # Data preparation
            models = sorted_stats['model']
            combined_scores = sorted_stats['balanced_combined_score_mean']
            y_pos = np.arange(len(models))
            
            # Get API vs. Local model masks for visual distinction
            api_models_mask = [model in self.api_models for model in models]
            
            # Create colors based on model type and highlight status
            bar_colors = []
            for i, model in enumerate(models):
                if model in self.highlight_models:
                    # Highlighted models get more saturated colors
                    bar_colors.append('#1a5fb4' if model in self.api_models else '#2b7827')
                else:
                    # Standard colors for other models
                    bar_colors.append('#6baed6' if model in self.api_models else '#74c476')
            
            # Plot horizontal bars
            bars = ax.barh(
                y_pos,
                combined_scores,
                height=0.8,
                color=bar_colors,
                edgecolor='black',
                alpha=0.8
            )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width + 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}',
                    ha='left',
                    va='center',
                    fontsize=10,
                    fontweight='bold'
                )
            
            # Add rank numbers
            for i, (model, score) in enumerate(zip(models, combined_scores)):
                rank = len(models) - i
                ax.text(
                    0.01,
                    y_pos[i],
                    f'#{rank}',
                    ha='left',
                    va='center',
                    color='white',
                    fontweight='bold',
                    fontsize=10
                )
            
            # Configure axes
            ax.set_yticks(y_pos)
            ax.set_yticklabels(models)
            ax.set_xlabel('Combined Score (Balanced)', fontsize=12)
            ax.set_title('Model Ranking Based on Balanced Combined Score', fontsize=18)
            
            # Add a vertical line for average score
            avg_score = combined_scores.mean()
            ax.axvline(x=avg_score, color='gray', linestyle='--', alpha=0.7)
            ax.text(
                avg_score + 0.01,
                -0.5,
                f'Average: {avg_score:.3f}',
                ha='left',
                va='center',
                color='gray',
                fontsize=10,
                fontstyle='italic'
            )
            
            # Add background shading for model types
            for i, is_api in enumerate(api_models_mask):
                y_start = y_pos[i] - 0.3
                y_end = y_pos[i] + 0.3
                if is_api:
                    # Light blue background for API models
                    ax.axhspan(y_start, y_end, color='#e6f2ff', alpha=0.3, zorder=-1)
                else:
                    # Light green background for Local models
                    ax.axhspan(y_start, y_end, color='#e6ffe6', alpha=0.3, zorder=-1)
            
            # Add legend for model types
            api_patch = plt.Rectangle((0, 0), 1, 1, fc='#e6f2ff', alpha=0.5, ec='gray')
            local_patch = plt.Rectangle((0, 0), 1, 1, fc='#e6ffe6', alpha=0.5, ec='gray')
            ax.legend([api_patch, local_patch], ['API Models', 'Local Models'], 
                     loc='lower right', framealpha=0.7)
            
            # Add explanation text
            plt.figtext(0.5, 0.01, 
                       "Combined Score = Average of Balanced Condition, Answer, and Citation Scores\n"
                       "Balanced scoring accounts for both quality and quantity of generated answers",
                       ha='center', fontsize=10, style='italic')
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            # Save to PDF
            pdf.savefig(fig)
            plt.close()
        
        logging.info(f"Created model ranking PDF at {pdf_path}")
        
        # Also save as PNG for convenience
        plt.savefig(self.output_dir / 'model_ranking.png', dpi=300, bbox_inches='tight')

    def create_enhanced_stats_table_pdf(self):
        """Create an enhanced PDF for the statistics table with more comprehensive metrics"""
        if self.model_stats is None:
            logging.error("No model stats available, cannot create stats table PDF")
            return
        
        pdf_path = self.output_dir / 'enhanced_model_stats_table.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Hide axes
            ax.axis('off')
            
            # Create table data
            table_data = []
            header = ['Model', 'Type', 'Cond. Score', 'Ans. Score', 'Cite. Score', 
                     'Combined', 'Avg. Citations', 'Avg. Answers', 'Diff.', 'Rank']
            
            table_data.append(header)
            
            # Add rows for each model
            for i, (_, row) in enumerate(self.model_stats.iterrows()):
                model_row = [
                    row['model'],
                    row['model_type_first'],
                    f"{row['balanced_condition_score_mean']:.3f} ± {row['balanced_condition_score_std']:.3f}",
                    f"{row['balanced_answer_score_mean']:.3f} ± {row['balanced_answer_score_std']:.3f}",
                    f"{row['balanced_citation_score_mean']:.3f} ± {row['balanced_citation_score_std']:.3f}",
                    f"{row['balanced_combined_score_mean']:.3f}",
                    f"{row['citation_count_mean']:.2f}",
                    f"{row['answer_count_mean']:.2f}",
                    f"{row['answer_count_difference_mean']:.2f}",
                    f"#{i+1}"
                ]
                table_data.append(model_row)
            
            # Add average row
            avg_row = [
                'Average',
                '-',
                f"{self.model_stats['balanced_condition_score_mean'].mean():.3f}",
                f"{self.model_stats['balanced_answer_score_mean'].mean():.3f}",
                f"{self.model_stats['balanced_citation_score_mean'].mean():.3f}",
                f"{self.model_stats['balanced_combined_score_mean'].mean():.3f}",
                f"{self.model_stats['citation_count_mean'].mean():.2f}",
                f"{self.model_stats['answer_count_mean'].mean():.2f}",
                f"{self.model_stats['answer_count_difference_mean'].mean():.2f}",
                '-'
            ]
            table_data.append(avg_row)
            
            # Create the table
            table = ax.table(
                cellText=table_data,
                loc='center',
                cellLoc='center',
                colWidths=[0.09, 0.06, 0.14, 0.14, 0.14, 0.1, 0.09, 0.09, 0.07, 0.06]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.6)
            
            # Style header row
            for i, key in enumerate(header):
                cell = table[(0, i)]
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e6f2ff')
            
            # Style average row
            for i in range(len(header)):
                cell = table[(len(table_data)-1, i)]
                cell.set_facecolor('#f2f2f2')
                cell.set_text_props(weight='bold')
            
            # Style model rows based on type and highlight status
            for i in range(1, len(table_data)-1):
                model = table_data[i][0]
                model_type = table_data[i][1]
                
                # Style model name cell
                cell = table[(i, 0)]
                if model in self.highlight_models:
                    cell.set_text_props(weight='bold')
                
                # Style all cells in this row
                for j in range(len(header)):
                    cell = table[(i, j)]
                    
                    if model_type == 'API':
                        # Light blue for API models
                        cell.set_facecolor('#e6f2ff' if j > 0 else '#d4e6f1')
                    else:
                        # Light green for Local models
                        cell.set_facecolor('#e6ffe6' if j > 0 else '#d4f1d4')
                
                # Highlight combined score cell for top models
                combined_cell = table[(i, 5)]
                if model in self.highlight_models:
                    combined_cell.set_facecolor('#fff2cc')  # Light yellow highlight
                    combined_cell.set_text_props(weight='bold')
            
            # Set title
            plt.suptitle('Enhanced Model Performance Statistics', fontsize=20, y=0.95)
            plt.figtext(0.5, 0.02, 
                     'Table: Main experiment scores with balanced scoring methodology.\n'
                     'Balanced scores account for both quality and consistency with expected answer count.', 
                     ha='center', fontsize=10)
            
            # Add API and Local model explanation
            plt.figtext(0.15, 0.06, 
                     'API Models: GPT-4o, glm4-Plus', 
                     ha='left', fontsize=9, 
                     bbox=dict(facecolor='#e6f2ff', alpha=0.8, boxstyle='round,pad=0.5', edgecolor='gray'))
            
            plt.figtext(0.85, 0.06, 
                     'Local Models: Llama3.1, Mistral, Gemma2, Qwen2.5, GLM4', 
                     ha='right', fontsize=9, 
                     bbox=dict(facecolor='#e6ffe6', alpha=0.8, boxstyle='round,pad=0.5', edgecolor='gray'))
            
            # Save to PDF
            pdf.savefig(fig)
            plt.close()
        
        logging.info(f"Created enhanced stats table PDF at {pdf_path}")
        
        # Save as PNG for convenience
        plt.savefig(self.output_dir / 'enhanced_model_stats_table.png', dpi=300, bbox_inches='tight')

    def run(self):
        """Run all visualizations and create separate PDFs"""
        # Load data
        if not self.load_data():
            logging.error("Failed to load data, cannot generate visualizations")
            return
        
        # Create separate PDFs for each visualization, using balanced scores
        self.create_score_distributions_pdf()
        self.create_comprehensive_bar_chart()
        self.create_horizontal_ranking_chart()
        self.create_enhanced_stats_table_pdf()
        
        logging.info("All enhanced visualizations generated successfully")


if __name__ == "__main__":
    # Create visualizer
    visualizer = EnhancedVisualizer(data_dir=".", output_dir="visualizations")
    
    # Run visualizations
    visualizer.run()