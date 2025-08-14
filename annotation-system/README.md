# 🤖 AI-AI Annotation System (Enhanced Supplementary Tool)

This directory contains the **enhanced AI-AI annotation system** for the CondAmbigQA project - a **supplementary tool** that assists in preliminary annotation generation for conditional ambiguous questions. 

⚠️ **Important Note**: This system is designed as an **auxiliary tool** for annotation assistance. **Human verification and refinement are essential** for final annotation quality. The primary annotation methodology in the paper relies on **human-AI collaborative annotation**.

## 📁 About This Supplementary Component

This is a **supplementary component** of the [CondAmbigQA](https://github.com/innovation64/CondAmbigQA) project, providing automated pre-annotation capabilities to assist human annotators in generating preliminary condition-answer pairs from ambiguous questions. **The final annotations require human expert review and validation**.

## 🌟 Enhanced Features & Upgrades

### 🚀 **New in This Version**
- **Enhanced GPT-4o Integration**: Primary model with GPT-4o-mini fallback support
- **Advanced Quality Metrics**: 8 comprehensive evaluation dimensions (up from 7)
- **Robust Error Handling**: Enhanced retry mechanisms and graceful shutdown
- **Parallel Processing**: Improved concurrent processing with safety controls
- **Rich Conversation Logs**: Detailed conversation history tracking
- **Extended Token Support**: Up to 6,000 tokens for comprehensive annotations

### 🎯 **Core Features**
- **Pre-Annotation Generation**: Three specialized AI agents (Annotator, Reviewer, Facilitator) for preliminary annotation drafts
- **Quality-Based Iteration**: 3-5 rounds of automated improvements to assist human reviewers
- **Citation Assistance**: Automated separation of condition-supporting and answer-supporting citations for human review
- **Batch Pre-Processing**: Efficient preliminary annotation generation for large datasets
- **Quality Assessment Support**: 8-dimensional evaluation to flag potential issues for human attention

⚠️ **Workflow Integration**: AI-AI → **Human Review** → Final Annotation

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- OpenAI API key with GPT-4 access
- Install the main CondAmbigQA requirements

### Installation

From the main CondAmbigQA directory:

```bash
# Navigate to annotation system
cd annotation-system

# Install specific requirements
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

### Basic Usage

```bash
# Run annotation on a dataset
python main.py --input ../data/questions.json --output ../data/annotated.json

# With batch processing
python batch_processor.py --input ../data/large_dataset.json --output-dir ../results/
```

## 📊 Input/Output Integration

### Input Format
The system expects JSON files with the structure used in CondAmbigQA:
```json
[
  {
    "id": "unique_id",
    "question": "Ambiguous question",
    "ctxs": [{"title": "...", "text": "..."}]
  }
]
```

### Output Format
Produces annotations compatible with the CondAmbigQA evaluation pipeline:
```json
[
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
    ]
  }
]
```

## 🏗️ System Architecture

```
annotation-system/
├── Core Modules
│   ├── config.py           # Configuration settings
│   ├── models.py           # AI agent implementations
│   ├── conversation.py     # Dialogue management
│   ├── data_processor.py   # Data I/O utilities
│   ├── main.py            # Main entry point
│   ├── utils.py           # Helper functions
│   └── batch_processor.py # Large-scale processing
│
├── Configuration
│   ├── requirements.txt   # Python dependencies
│   └── .env.example       # Environment template
│
└── Documentation
    ├── README.md          # This file
    └── example_input.json # Input examples
```

## ⚙️ Configuration

Key parameters in `config.py`:

- `MAX_CONDITIONS`: Maximum conditions per question (default: 5)
- `MIN_ROUNDS`: Minimum dialogue rounds (default: 3)
- `MAX_ROUNDS`: Maximum dialogue rounds (default: 5)
- `BATCH_SIZE`: Items per batch (default: 10)
- `MAX_PARALLEL_PROCESSES`: Concurrent processes (default: 4)

## 🎯 Enhanced Quality Metrics

The enhanced system evaluates annotations across **8 comprehensive dimensions**:

1. **Condition Comprehensiveness** (0.8 threshold) - *New: Evaluates condition completeness and richness*
2. **Condition Utility** (0.8 threshold) - *Enhanced evaluation criteria*
3. **Answer Completeness** (0.8 threshold) - *Improved detail level assessment*
4. **Condition Citation Relevance** (0.7 threshold) - *Separated from answer citations*
5. **Answer Citation Relevance** (0.7 threshold) - *New: Dedicated answer citation evaluation*
6. **Distinctness** (0.8 threshold) - *Enhanced uniqueness measurement*
7. **Logical Flow** (0.8 threshold) - *Improved coherence evaluation between conditions and answers*
8. **Overall Quality** (0.85 threshold) - *New: Comprehensive quality assessment*

## 🔗 Integration with CondAmbigQA

### Using with Main Pipeline

```python
# In your CondAmbigQA script
from annotation_system import main as annotation_main

# Annotate dataset
annotation_main.process_dataset(
    input_file="data/questions.json",
    output_file="data/annotated.json"
)
```

### Processing Pipeline

1. **Data Preparation**: Use CondAmbigQA's retrieval module to get contexts
2. **Annotation**: Run this system to generate condition-answer pairs
3. **Evaluation**: Use CondAmbigQA's evaluation metrics on annotated data
4. **Analysis**: Analyze results with CondAmbigQA's analysis tools

## 📈 Pre-Annotation Performance

- **Processing time**: ~30-60 seconds per preliminary annotation
- **API cost**: ~$0.08-0.15 per pre-annotation (before human review)
- **Memory usage**: ~1GB per 1000 items (improved efficiency)
- **Pre-annotation success rate**: >97% for generating draft annotations
- **Token capacity**: Up to 6,000 tokens for comprehensive preliminary annotations
- **Concurrent processing**: Up to 4 parallel processes with safety controls

⚠️ **Important**: These metrics are for **preliminary AI-generated annotations only**. Final annotation quality depends on human expert review and validation.

## 🛠️ Troubleshooting

Common issues and solutions:

1. **API Rate Limits**: Reduce `MAX_PARALLEL_PROCESSES`
2. **Memory Issues**: Decrease `BATCH_SIZE`
3. **Timeout Errors**: Increase `API_TIMEOUT` in config.py

## 📝 Logs

- Main log: `annotation.log`
- Batch logs: `batch_processor_*.log`
- Conversations: `conversation_logs/`

## 🤝 Contributing

Contributions should align with the main CondAmbigQA project. Please:

1. Follow the existing code style
2. Test with sample data
3. Update documentation
4. Submit PR to the main repository

## 📄 License

This component follows the same license as the main CondAmbigQA project.

## 📧 Support

For issues specific to this annotation system, please open an issue in the main [CondAmbigQA repository](https://github.com/innovation64/CondAmbigQA/issues) with the tag `[annotation-system]`.

---

## ⚠️ Important Disclaimers

- **This is a supplementary tool**: The AI-AI annotation system is designed to assist human annotators, not replace them
- **Human verification required**: All AI-generated annotations must be reviewed and validated by human experts
- **Paper methodology**: The primary annotation approach described in the CondAmbigQA paper relies on human-AI collaboration
- **Quality assurance**: Final annotation quality depends on human expertise and judgment
- **API costs**: Monitor usage and costs when processing large datasets

---

Part of the [CondAmbigQA](https://github.com/innovation64/CondAmbigQA) project for Conditional Ambiguous Question Answering research.