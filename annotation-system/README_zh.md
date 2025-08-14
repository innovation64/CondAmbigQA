# 标注系统

本目录包含 CondAmbigQA 项目的 AI-AI 标注系统，通过多智能体协作自动化标注条件性歧义问题。

## 📁 关于本组件

这是 [CondAmbigQA](https://github.com/innovation64/CondAmbigQA) 项目的重要组件，为歧义问题提供自动化标注功能，生成高质量的条件-答案对。

## 🌟 核心特性

- **多智能体系统**: 三个专门的AI智能体（标注员、审查员、协调员）通过对话协作
- **迭代优化**: 基于质量指标进行3-5轮改进
- **引用分离**: 区分支持条件的引用和支持答案的引用
- **批处理**: 高效处理大规模数据集
- **质量保证**: 在7个维度上进行全面评估

## 🚀 快速开始

### 环境要求

- Python 3.7+
- OpenAI API密钥（需支持GPT-4）
- 安装主项目 CondAmbigQA 的依赖

### 安装配置

从 CondAmbigQA 主目录执行：

```bash
# 进入标注系统目录
cd annotation-system

# 安装特定依赖
pip install -r requirements.txt

# 复制并配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 OPENAI_API_KEY
```

### 基本使用

```bash
# 对数据集进行标注
python main.py --input ../data/questions.json --output ../data/annotated.json

# 使用批处理
python batch_processor.py --input ../data/large_dataset.json --output-dir ../results/
```

## 📊 数据格式集成

### 输入格式
系统期望的JSON文件格式与CondAmbigQA保持一致：
```json
[
  {
    "id": "unique_id",
    "question": "歧义问题",
    "ctxs": [{"title": "...", "text": "..."}]
  }
]
```

### 输出格式
生成与CondAmbigQA评估流水线兼容的标注：
```json
[
  {
    "id": "unique_id",
    "question": "歧义问题",
    "properties": [
      {
        "condition": "条件描述",
        "groundtruth": "该条件下的答案",
        "condition_citations": [...],
        "answer_citations": [...]
      }
    ]
  }
]
```

## 🏗️ 系统架构

```
annotation-system/
├── 核心模块
│   ├── config.py           # 配置设置
│   ├── models.py           # AI智能体实现
│   ├── conversation.py     # 对话管理
│   ├── data_processor.py   # 数据输入输出工具
│   ├── main.py            # 主入口程序
│   ├── utils.py           # 辅助函数
│   └── batch_processor.py # 大规模处理
│
├── 配置文件
│   ├── requirements.txt   # Python依赖
│   └── .env.example       # 环境变量模板
│
└── 文档
    ├── README.md          # 英文文档
    ├── README_zh.md       # 本文件（中文文档）
    └── example_input.json # 输入示例
```

## ⚙️ 配置选项

`config.py` 中的关键参数：

- `MAX_CONDITIONS`: 每个问题的最大条件数（默认：5）
- `MIN_ROUNDS`: 最少对话轮数（默认：3）
- `MAX_ROUNDS`: 最多对话轮数（默认：5）
- `BATCH_SIZE`: 每批处理项目数（默认：10）
- `MAX_PARALLEL_PROCESSES`: 并发进程数（默认：4）

## 🎯 质量评估指标

系统在以下维度评估标注质量：

1. **条件精确性**（阈值：0.8）
2. **条件实用性**（阈值：0.8）
3. **答案完整性**（阈值：0.8）
4. **引用相关性**（阈值：0.7）
5. **区分度**（阈值：0.8）
6. **逻辑流畅性**（阈值：0.8）

## 🔗 与 CondAmbigQA 集成

### 在主流水线中使用

```python
# 在你的 CondAmbigQA 脚本中
from annotation_system import main as annotation_main

# 标注数据集
annotation_main.process_dataset(
    input_file="data/questions.json",
    output_file="data/annotated.json"
)
```

### 处理流水线

1. **数据准备**: 使用 CondAmbigQA 的检索模块获取上下文
2. **标注**: 运行本系统生成条件-答案对
3. **评估**: 使用 CondAmbigQA 的评估指标对标注数据进行评估
4. **分析**: 使用 CondAmbigQA 的分析工具分析结果

## 🎯 工作原理

### 三个AI智能体角色

1. **标注员（Annotator）**
   - 分析歧义问题
   - 识别不同的条件情况
   - 生成相应的答案
   - 提供支持引用

2. **审查员（Reviewer）**
   - 评估标注质量
   - 在7个维度上打分
   - 提供具体改进建议
   - 检查引用相关性

3. **协调员（Facilitator）**
   - 管理对话流程
   - 决定继续或结束对话
   - 控制质量标准
   - 提供最终评估

### 迭代改进流程

```
输入问题 → 初始标注 → 质量审查 → 改进决策
    ↑                                    ↓
    ←──────── 继续优化（2-5轮）←─────────
                                        ↓
                                    最终输出
```

## 📈 性能表现

- **处理时间**: 每个项目约30-60秒
- **API成本**: 每个项目约$0.10-0.20（GPT-4）
- **内存使用**: 1000个项目约需1GB
- **成功率**: 正确配置下>95%

## 🛠️ 常见问题排查

常见问题及解决方案：

1. **API速率限制**: 降低 `MAX_PARALLEL_PROCESSES`
2. **内存问题**: 减少 `BATCH_SIZE`
3. **超时错误**: 增加 config.py 中的 `API_TIMEOUT`

### 详细排查步骤

- **检查API密钥**: 确保.env文件中的API密钥有效
- **查看日志**: 检查 `annotation.log` 获取错误详情
- **监控API配额**: 确保有足够的API调用额度
- **调整并发**: 根据API等级调整并发数

## 📝 日志系统

- **主日志**: `annotation.log`
- **批处理日志**: `batch_processor_*.log`
- **对话记录**: `conversation_logs/`

## 🔧 高级用法

### 处理大型数据集

```bash
# 使用批处理器处理大数据集
python batch_processor.py \
  --input large_dataset.json \
  --output-dir ./batch_results \
  --batch-size 100 \
  --max-parallel 4

# 恢复中断的处理
python main.py \
  --input questions.json \
  --output final.json \
  --interim interim_annotations.json
```

### 自定义配置

```python
# 修改 config.py 中的参数
QUALITY_THRESHOLDS = {
    "condition_precision": 0.85,  # 提高质量要求
    "answer_completeness": 0.85,
    # ... 其他阈值
}
```

## 🤝 贡献指南

贡献内容应与主 CondAmbigQA 项目保持一致。请：

1. 遵循现有代码风格
2. 使用样本数据测试
3. 更新相关文档
4. 向主仓库提交PR

## 📄 许可证

本组件遵循与主 CondAmbigQA 项目相同的许可证。

## 📧 技术支持

如有标注系统相关问题，请在主 [CondAmbigQA 仓库](https://github.com/innovation64/CondAmbigQA/issues) 中提交issue，并添加 `[annotation-system]` 标签。

## 📚 相关资源

- [CondAmbigQA 主项目](https://github.com/innovation64/CondAmbigQA)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [GPT-4 使用指南](https://platform.openai.com/docs/models/gpt-4)

---

**注意**: 本系统需要大量API调用。处理大型数据集时请监控使用量和成本。

---

CondAmbigQA 条件性歧义问答研究项目的组成部分。