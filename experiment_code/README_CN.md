# CondAmbigQA - 实验代码（整理版）

## 🎯 项目概述

CondAmbigQA（条件性模糊问答）是一个研究大型语言模型在处理需要额外条件信息才能准确回答的模糊问题时表现的项目。

## 📁 目录结构

```
clean_experiment_code/
├── core/                      # 核心实验脚本
│   ├── config.py             # 模型配置文件
│   ├── models.py             # 模型接口类
│   ├── utils.py              # 工具函数
│   ├── evaluator.py          # 评估框架
│   ├── main_experiment.py    # 主实验运行器
│   ├── pure_llm_experiment.py # 纯LLM实验（无检索）
│   └── ground_truth_experiment.py # 真值标注实验
├── visualization/            # 可视化脚本
│   ├── main_visualizer.py    # 综合结果可视化
│   ├── comparison_visualizer.py # 实验对比图表
│   └── statistics_visualizer.py # 统计分析图表
├── data/                     # 数据集
│   ├── cleaned_mcaqa.json   # 主数据集 (2,200样本) - 使用这个
│   └── mcaqa.json          # 旧数据集 (200样本) - 忽略
├── sample_results/          # 示例输出文件
│   ├── results_gpt-4o.json  # GPT-4o结果示例
│   ├── results_gemma2_latest.json # Gemma2结果示例
│   └── pure_llm_results_gpt-4o.json # 纯LLM结果示例
├── retrieval/               # 检索增强
│   ├── indexing.py          # 文档索引
│   └── retrieval_data.py    # 检索处理
├── docs/                    # 文档
└── requirements.txt         # Python依赖
```

## 📊 数据需求

### 核心数据集
- **主数据集**: `data/cleaned_mcaqa.json` (2,200个样本)
- **⚠️ 重要**: 不要使用 `mcaqa.json` (只有200个样本 - 过时版本)

### 实验 → 可视化数据流程
1. **运行实验** → 在当前目录生成结果文件
2. **移动结果** → 复制到 `sample_results/` 供可视化使用
3. **运行可视化** → 从 `sample_results/` 读取数据

**完整数据需求**: 参见 `docs/DATA_REQUIREMENTS.md`

## 🚀 快速开始

### 1. 环境配置
```bash
pip install -r requirements.txt

# OpenAI模型配置
export OPENAI_API_KEY="your_api_key"

# Ollama本地模型 - 确保Ollama运行在localhost:11434
```

### 2. 运行实验

#### 主实验（带检索增强）
```bash
cd core
python main_experiment.py
```

#### 纯LLM实验（无检索）
```bash
cd core  
python pure_llm_experiment.py
```

#### 真值标注实验
```bash
cd core
python ground_truth_experiment.py
```

### 3. 结果可视化

#### 综合结果分析
```bash
cd visualization
python main_visualizer.py
```

#### 实验对比分析
```bash
cd visualization
python comparison_visualizer.py
```

#### 统计分析
```bash
cd visualization
python statistics_visualizer.py
```

## ⚙️ 配置说明

### 支持的模型 (core/config.py)

**OpenAI模型:**
- `gpt-4o`
- `glm-4-plus`

**本地Ollama模型:**  
- `llama3.1:8b`
- `qwen2.5:latest`
- `mistral:latest`
- `gemma2:latest`
- `glm4:latest`
- `deepseek-r1:7b`

## 📊 评估指标

- **条件分数**: 生成条件的准确性
- **答案分数**: 条件性答案的正确性
- **引用分数**: 引用来源的质量
- **综合分数**: 加权组合评分

## 📈 输出文件

**主要结果:**
- `results_{model_name}.json` - 完整实验结果
- `intermediate_results_{model_name}_main_experiment.json` - 中间数据

**纯LLM结果:**  
- `pure_llm_results_{model_name}.json` - 无检索结果

**真值标注:**
- `results_{model_name}_ground_truth.json` - 标注真值

## 🛠️ 故障排除

### 常见问题

1. **API连接错误**
   - 检查API密钥和网络连接
   - 验证本地模型的Ollama服务状态

2. **内存问题**
   - 使用更小的模型或在配置中减少批处理大小

3. **依赖包缺失**
   - 运行 `pip install -r requirements.txt --force-reinstall`

## 📝 核心特性

- **清晰架构**: 组织良好、易于维护的代码结构
- **多模型支持**: OpenAI API和本地Ollama模型
- **综合评估**: 多维度评分系统
- **丰富可视化**: 统计分析和对比图表
- **灵活配置**: 易于调整模型和参数

## 📄 许可证

仅供学术研究使用。

---

## 🔄 工作流程

1. **配置** → 安装依赖并配置模型
2. **运行** → 使用核心脚本执行实验
3. **分析** → 生成可视化和统计
4. **对比** → 评估不同模型和方法

详细英文文档请参见 `README.md`。