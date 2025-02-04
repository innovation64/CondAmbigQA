# ![CondAmbigQA](https://your-banner-image-url.png)

🏠 [Homepage](https://github.com/innovation64/CondAmbigQA) | 📝 [Paper](https://arxiv.org/abs/2502.01523) | 🤗 [Dataset](https://huggingface.co/datasets/Apocalypse-AGI-DAO/CondAmbigQA) | 📊 [Results](#results) | 🌏 [中文](./README_zh.md)

🔄 Supported by RAG, LLMs, Conditions-based QA

💫 A novel benchmark for resolving ambiguous questions through conditions identification.

## 📌 CondAmbigQA: A Conditional Ambiguous Question Answering Benchmark

- 📚 News
  - 🎉 Dataset Release: [Hugging Face](https://huggingface.co/datasets/Apocalypse-AGI-DAO/CondAmbigQA)
  - 📄 Paper Release: [arXiv](https://arxiv.org/abs/2502.01523)

- 🚀 Getting Started
  - [Installation](#installation)
  - [Dataset Usage](#dataset-usage)
  - [Model Evaluation](#evaluation)

## 💡 Quick Start

```python
from datasets import load_dataset
dataset = load_dataset("Apocalypse-AGI-DAO/CondAmbigQA")
```

## 📊 Results 

| Model | Condition Score | Answer Score | Citation Score |
|-------|----------------|--------------|----------------|
| LLaMA3.1 | 0.305 | 0.276 | 0.058 |
| Mistral | 0.316 | 0.272 | 0.036 |
| Qwen2.5 | 0.317 | 0.297 | 0.050 |
| GLM4 | 0.313 | 0.295 | 0.059 |
| Gemma2 | 0.309 | 0.306 | 0.077 |

## 📖 Citation

```bibtex
@article{li2024condambigqa,
  title={CondAmbigQA: A Benchmark and Dataset for Conditional Ambiguous Question Answering},
  author={Li, Zongxi and Li, Yang and Xie, Haoran and Qin, S. Joe},
  journal={arXiv preprint arXiv:2502.01523},
  year={2024}
}
```

## 📬 Contact 

- Email: zongxili@ln.edu.hk
- Join our community: [Discord](#) | [Slack](#) | [WeChat](#)
