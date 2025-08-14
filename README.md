# ![CondAmbigQA](./img/label.svg)

🏠 [Homepage](https://github.com/innovation64/CondAmbigQA) | 📝 [Paper](https://arxiv.org/abs/2502.01523) | 🤗 [Dataset](https://huggingface.co/datasets/Apocalypse-AGI-DAO/CondAmbigQA) | 🆕 [CondAmbigQA-2K](https://huggingface.co/datasets/Apocalypse-AGI-DAO/CondAmbigQA-2K) | 📊 [Results](#results) | 🌏 [中文](./README_zh.md)

🔄 Supported by RAG, LLMs, Conditions-based QA

💫 A novel benchmark for resolving ambiguous questions through conditions identification.

## 📌 CondAmbigQA: A Conditional Ambiguous Question Answering Benchmark

- 📚 News
  - 🆕 **CondAmbigQA-2K Dataset**: [Hugging Face](https://huggingface.co/datasets/Apocalypse-AGI-DAO/CondAmbigQA-2K) - Extended 2,000 sample dataset
  - 🎉 Original Dataset: [Hugging Face](https://huggingface.co/datasets/Apocalypse-AGI-DAO/CondAmbigQA)
  - 📄 Paper Release: [arXiv](https://arxiv.org/abs/2502.01523)

- 🚀 Getting Started
  - [Installation](#installation)
  - [Dataset Usage](#dataset-usage)
  - [Model Evaluation](#evaluation)
  - [Annotation System](#annotation-system)
  - [Experiment Code](#experiment-code)

## 💡 Quick Start

### Basic Dataset Usage

```python
from datasets import load_dataset
dataset = load_dataset("Apocalypse-AGI-DAO/CondAmbigQA")
```

### Run Experiments

```bash
# Run main experiment with retrieval
cd experiment_code/core
python main_experiment.py

# Run annotation system
cd annotation-system
python main.py --input data/questions.json --output data/annotated.json
```

## 🏗️ Project Components

### 📁 Directory Structure

```
CondAmbigQA/
├── annotation-system/     # AI-AI annotation system
├── experiment_code/       # Experimental evaluation code
├── retrieval/            # Document retrieval components
├── label/               # Labeling utilities
└── results/             # Experimental results and visualizations
```

### 📝 Annotation Methodology

**Primary Approach**: **Human-AI Collaborative Annotation** (as described in the paper)
- Expert human annotators working with AI assistance
- Manual quality control and validation
- Iterative refinement through human judgment

**Supplementary Tool**: **AI-AI Pre-Annotation System**
- **⚠️ Auxiliary tool only** - requires human verification
- Automated preliminary annotation generation
- 8-dimensional quality assessment for flagging issues
- Enhanced multi-agent collaboration with GPT-4o
- **Workflow**: AI-AI Pre-annotation → **Human Expert Review** → Final Annotation

### 🧪 Experiment Code
- **Multiple Model Support**: OpenAI API and local Ollama models
- **Comprehensive Evaluation**: Multi-dimensional scoring system
- **Rich Visualization**: Statistical analysis and comparison charts
- **Flexible Configuration**: Easy model and parameter adjustment

## 📋 Installation

### Prerequisites
- Python 3.7+
- OpenAI API key (for GPT models)
- Ollama (for local models)

### Setup
```bash
# Clone the repository
git clone https://github.com/innovation64/CondAmbigQA.git
cd CondAmbigQA

# Install main dependencies
pip install -r requirements.txt

# For annotation system
cd annotation-system
pip install -r requirements.txt

# For experiment code
cd experiment_code
pip install -r requirements.txt
```

### Environment Configuration
```bash
# For OpenAI models
export OPENAI_API_KEY="your_api_key"

# Ensure Ollama is running (for local models)
# Default: localhost:11434
```

## 📖 Dataset Usage

### Loading the Dataset

#### Original CondAmbigQA Dataset
```python
from datasets import load_dataset
dataset = load_dataset("Apocalypse-AGI-DAO/CondAmbigQA")
```

#### Extended CondAmbigQA-2K Dataset (2,000 samples)
```python
from datasets import load_dataset
dataset_2k = load_dataset("Apocalypse-AGI-DAO/CondAmbigQA-2K")
```

### Data Structure
```json
{
  "id": "unique_id",
  "question": "Ambiguous question",
  "properties": [
    {
      "condition": "Condition description",
      "groundtruth": "Answer for condition",
      "condition_citations": [...],
      "answer_citations": [...]
    }
  ],
  "ctxs": [{"title": "...", "text": "..."}]
}
```

## 🔬 Model Evaluation

### Supported Models
- **OpenAI**: GPT-4o, GLM-4-plus
- **Local Ollama**: Llama3.1:8b, Qwen2.5, Mistral, Gemma2, GLM4, DeepSeek-R1

### Running Experiments
```bash
# Main experiment (with retrieval)
cd experiment_code/core
python main_experiment.py

# Pure LLM experiment (no retrieval)
python pure_llm_experiment.py

# Ground truth annotation
python ground_truth_experiment.py
```

### Visualization
```bash
cd experiment_code/visualization
python main_visualizer.py          # Comprehensive analysis
python comparison_visualizer.py    # Experiment comparison
python statistics_visualizer.py    # Statistical analysis
```

## 📊 Results & Performance Analysis

### 🏆 Main Experiment Results (With Retrieval Augmentation)

| Model | Type | Condition Score | Answer Score | Citation Score | Combined Score | Rank |
|-------|------|----------------|-------------|----------------|----------------|------|
| **GPT-4o** | API | **0.552 ± 0.190** | **0.558 ± 0.157** | **0.875 ± 0.207** | **0.662** | 🥇 #1 |
| **GLM4-plus** | API | 0.302 ± 0.069 | 0.420 ± 0.097 | 0.441 ± 0.261 | 0.388 | 🥈 #2 |
| **Qwen2.5** | Local | 0.235 ± 0.120 | 0.287 ± 0.161 | 0.558 ± 0.359 | 0.360 | 🥉 #3 |
| **DeepSeek-R1** | Local | 0.245 ± 0.112 | 0.293 ± 0.142 | 0.501 ± 0.342 | 0.346 | #4 |
| GLM4 | Local | 0.231 ± 0.071 | 0.290 ± 0.090 | 0.320 ± 0.215 | 0.280 | #5 |
| LLaMA3.1 | Local | 0.232 ± 0.076 | 0.252 ± 0.093 | 0.306 ± 0.246 | 0.264 | #6 |
| Mistral | Local | 0.196 ± 0.060 | 0.231 ± 0.079 | 0.263 ± 0.214 | 0.230 | #7 |
| Gemma2 | Local | 0.170 ± 0.091 | 0.203 ± 0.118 | 0.217 ± 0.277 | 0.197 | #8 |
| **Average** | - | **0.270** | **0.317** | **0.435** | **0.341** | - |

### 📈 Comprehensive Performance Analysis

| Model Type | Pure LLM (Avg. Answer Score) | w/ Model-generated Conditions | w/ GT Conditions | Improvement |
|------------|------------------------------|-------------------------------|------------------|-------------|
| **Proprietary Models** | | | | |
| **GPT-4o** | **0.25** | **0.56** | **0.57** | **+128%** |
| GLM4-plus | 0.24 | 0.42 | 0.53 | +121% |
| **Local Models** | | | | |
| Qwen2.5 | 0.15 | 0.29 | 0.40 | +167% |
| Mistral | 0.17 | 0.23 | 0.29 | +61% |
| Gemma2 | 0.15 | 0.20 | 0.29 | +93% |
| LLaMA3.1 | 0.14 | 0.25 | 0.29 | +107% |
| GLM4 | 0.14 | 0.29 | 0.38 | +171% |
| **DeepSeek-R1** | 0.07 | 0.29 | 0.34 | **+400%** |
| **Average** | **0.164** | **0.316** | **0.386** | **+135%** |

### 🎯 Key Performance Insights

#### 🚀 Main Experiment (RAG) Performance
- **Clear Winner**: **GPT-4o** dominates all metrics with 0.662 combined score
  - **Citation Excellence**: 0.875 citation score (far ahead of others)
  - **Balanced Performance**: Strong across conditions (0.552), answers (0.558), and citations
  
- **API vs Local Model Gap**: 
  - **API Models**: GPT-4o (0.662), GLM4-plus (0.388)
  - **Best Local**: Qwen2.5 (0.360) - competitive with API models
  - **Local Model Range**: 0.197 (Gemma2) to 0.360 (Qwen2.5)

- **Citation Performance Hierarchy**:
  - **Tier 1**: GPT-4o (0.875) - exceptional citation quality
  - **Tier 2**: Qwen2.5 (0.558), DeepSeek-R1 (0.501) - good citation skills
  - **Tier 3**: Others (0.217-0.441) - moderate citation ability

#### 📊 Cross-Experiment Analysis
- **Dramatic Improvement with Conditions**: All models show substantial performance gains when provided with conditions
  - **DeepSeek-R1**: Most dramatic improvement (+400%, from 0.07 to 0.34 in pure LLM vs conditional)
  - **Average Improvement**: +135% across all models
  - **Local Models**: Show higher improvement rates than proprietary models

- **Performance Consistency**:
  - **Most Stable**: GLM4-plus (±0.069 condition score variance)
  - **Most Variable**: GPT-4o (±0.190) but still top performer
  - **Standard Deviations**: Show model reliability across different questions

### 📊 Visualizations & Charts

The project includes comprehensive performance analysis charts organized by experiment type:

#### 🎯 Main Experiment Visualizations
- **Comprehensive Performance**: `experiment_code/sample_charts/main_visualizations/comprehensive_performance.pdf`
- **Model Ranking**: `experiment_code/sample_charts/main_visualizations/model_ranking.pdf`
- **Score Distributions**: `experiment_code/sample_charts/main_visualizations/score_distributions.pdf`
- **Enhanced Model Stats Table**: `experiment_code/sample_charts/main_visualizations/enhanced_model_stats_table.pdf`

#### 🔄 Comparison Analysis Charts  
- **Experiment Comparison**: `experiment_code/sample_charts/comparison_visualizations/experiment_comparison.pdf`
- **Cross-Experiment Performance**: `experiment_code/sample_charts/comparison_visualizations/comprehensive_performance.pdf`
- **Model Ranking Comparison**: `experiment_code/sample_charts/comparison_visualizations/model_ranking.pdf`

#### 📈 Statistical Analysis
- **Condition vs Answer Scatter Plot**: `experiment_code/sample_charts/statistical_visualizations/condition_vs_answer_scatter copy.pdf`

#### 📋 Summary Charts (Root Level)
- **Overall Performance Summary**: `experiment_code/sample_charts/comprehensive_performance.pdf`
- **Model Ranking Overview**: `experiment_code/sample_charts/model_ranking.pdf`
- **Score Distribution Summary**: `experiment_code/sample_charts/score_distributions.pdf`
- **Enhanced Stats Table**: `experiment_code/sample_charts/enhanced_model_stats_table.pdf`

### 🔬 Experiment Types

1. **Main Experiments**: Full RAG pipeline with retrieval augmentation
2. **Pure LLM**: Models without external knowledge retrieval
3. **Ground Truth**: Human-annotated reference performance
4. **Comparison**: Cross-experiment performance analysis

## 📖 Citation

```bibtex
@article{li2024condambigqa,
  title={CondAmbigQA: A Benchmark and Dataset for Conditional Ambiguous Question Answering},
  author={Li, Zongxi and Li, Yang and Xie, Haoran and Qin, S. Joe},
  journal={arXiv preprint arXiv:2502.01523},
  year={2025}
}
```

## 📬 Contact 

- Email: zongxili@ln.edu.hk
