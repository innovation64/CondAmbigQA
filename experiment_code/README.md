# CondAmbigQA - Clean Experimental Code

## 🎯 Project Overview

CondAmbigQA (Conditional Ambiguous Question Answering) is a research project evaluating large language models' performance on ambiguous questions that require additional conditional information for accurate answering.

## 📁 Directory Structure

```
clean_experiment_code/
├── core/                      # Core experiment scripts
│   ├── config.py             # Model configuration
│   ├── models.py             # Model interface classes  
│   ├── utils.py              # Utility functions
│   ├── evaluator.py          # Evaluation framework
│   ├── main_experiment.py    # Main experiment runner
│   ├── main_and_comapre_experiment.py # Combined main & comparison experiments
│   ├── pure_llm_experiment.py # Pure LLM experiment (no retrieval)
│   ├── ground_truth_experiment.py # Ground truth annotation
│   └── scaling_other_dataset_experiment.py # Scaling to other datasets
├── visualization/            # Visualization scripts  
│   ├── main_visualizer.py    # Comprehensive result visualization
│   ├── comparison_visualizer.py # Experiment comparison charts
│   ├── statistics_visualizer.py # Statistical analysis plots
│   ├── main_experiment_stats_viz.py # Main experiment statistics visualization
│   └── comp_experiment_stats_viz.py # Comparison experiment statistics visualization
├── data/                     # Datasets
│   ├── cleaned_mcaqa.json   # Main dataset (2,200 samples) - USE THIS
│   ├── condambigqa_2000.json # Alternative dataset (2,000 samples)
│   └── mcaqa.json          # Legacy dataset (200 samples) - IGNORE
├── sample_results/          # Complete result files (45+ files)
│   ├── main_experiments/        # Main results (7 models)
│   ├── comparison_experiments/  # Comparison results (7 models)  
│   ├── ground_truth_experiments/ # Ground truth results (7 models)
│   ├── pure_llm_experiments/    # Pure LLM results (8 models)
│   ├── intermediate_results/    # Processing data samples (4 files)
│   └── summary_scores/         # Aggregated score files (6 files)
├── retrieval/               # Retrieval augmentation
│   ├── indexing.py          # Document indexing
│   └── retrieval_data.py    # Retrieval processing
├── docs/                    # Documentation
│   ├── DATA_REQUIREMENTS.md  # Detailed data requirements guide
│   └── COMPLETE_FILE_INVENTORY.md # Complete file listing (97 files)
├── sample_charts/             # Complete visualization outputs (26 charts)
│   ├── main_visualizations/       # Main experiment charts
│   ├── comparison_visualizations/ # Comparison charts  
│   ├── performance_analysis/      # Performance analysis
│   └── statistical_visualizations/ # Statistical charts
└── requirements.txt         # Python dependencies
```

## 📊 Data Requirements

### Essential Dataset
- **Main Dataset**: `data/cleaned_mcaqa.json` (2,200 samples)
- **⚠️ IMPORTANT**: Do NOT use `mcaqa.json` (only 200 samples - outdated)

### Experiment → Visualization Data Flow
1. **Run Experiments** → Generates result files in current directory
2. **Move Results** → Copy to `sample_results/` for visualization
3. **Run Visualization** → Reads from `sample_results/`

**For Complete Data Requirements**: See `docs/DATA_REQUIREMENTS.md`

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt

# For OpenAI models
export OPENAI_API_KEY="your_api_key"

# For Ollama local models - ensure Ollama is running at localhost:11434
```

### 2. Run Experiments

#### Main Experiment (with retrieval augmentation)
```bash
cd core
python main_experiment.py
```

#### Pure LLM Experiment (no retrieval)
```bash
cd core  
python pure_llm_experiment.py
```

#### Ground Truth Annotation
```bash
cd core
python ground_truth_experiment.py
```

#### Combined Main & Comparison Experiment
```bash
cd core  
python main_and_comapre_experiment.py
```

#### Scaling to Other Datasets
```bash
cd core
python scaling_other_dataset_experiment.py
```

### 3. Visualization

**⚠️ Prerequisites**: Run experiments first to generate result files

#### Comprehensive Results Analysis
```bash
cd visualization
python main_visualizer.py
# Needs: results_*_main.json files in sample_results/
```

#### Compare Different Experiments  
```bash
cd visualization
python comparison_visualizer.py
# Needs: *_ground_truth.json, *_main.json, *_compare.json files
```

#### Statistical Analysis
```bash
cd visualization
python statistics_visualizer.py
# Works with: any results_*.json files
```

#### Main Experiment Statistics Visualization
```bash
cd visualization
python main_experiment_stats_viz.py
# Generates detailed statistical visualizations for main experiments
```

#### Comparison Experiment Statistics Visualization
```bash
cd visualization
python comp_experiment_stats_viz.py
# Generates statistical comparisons between different experiment types
```

## ⚙️ Configuration

### Supported Models (core/config.py)

**OpenAI Models:**
- `gpt-4o`
- `glm-4-plus`

**Local Ollama Models:**  
- `llama3.1:8b`
- `qwen2.5:latest`
- `mistral:latest`
- `gemma2:latest`
- `glm4:latest`
- `deepseek-r1:7b`

## 📊 Evaluation Metrics

- **Condition Score**: Accuracy of generated conditions
- **Answer Score**: Correctness of conditional answers  
- **Citation Score**: Quality of source citations
- **Overall Score**: Weighted combination

## 📈 Output Files

**Main Results:**
- `results_{model_name}.json` - Complete experiment results
- `intermediate_results_{model_name}_main_experiment.json` - Intermediate data

**Pure LLM Results:**  
- `pure_llm_results_{model_name}.json` - No retrieval results

**Ground Truth:**
- `results_{model_name}_ground_truth.json` - Annotated ground truth

## 🛠️ Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Check API keys and network connectivity
   - Verify Ollama service status for local models

2. **Memory Issues** 
   - Use smaller models or reduce batch size in config

3. **Missing Dependencies**
   - Run `pip install -r requirements.txt --force-reinstall`

## 📝 Key Features

- **Clean Architecture**: Organized, maintainable code structure
- **Multiple Model Support**: OpenAI API and local Ollama models
- **Comprehensive Evaluation**: Multi-dimensional scoring system
- **Rich Visualization**: Statistical analysis and comparison charts  
- **Flexible Configuration**: Easy model and parameter adjustment

## 📄 License

Academic research use only.

---

## 🔄 Workflow

1. **Setup** → Install dependencies and configure models
2. **Run** → Execute experiments using core scripts
3. **Analyze** → Generate visualizations and statistics  
4. **Compare** → Evaluate different models and approaches

For detailed Chinese documentation, see `README_CN.md`.