import faiss
import numpy as np
import os
import psutil
from datasets import load_dataset
from tqdm import tqdm

# 创建 CPU 索引，使用内积作为距离度量
index = faiss.IndexFlatIP(768)  # 使用内积
print("CPU 索引初始化完成。")

# 加载数据集
hf_token = "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"
dataset = load_dataset("WhereIsAI/bge_wikipedia-data-en", split="train", token=hf_token)
print("数据集加载完成。")

# 初始化批次大小
initial_batch_size = 5000
batch_size = initial_batch_size

# 处理批次数据
total_batches = (len(dataset) + batch_size - 1) // batch_size  # 确保包括所有数据

for i in tqdm(range(0, len(dataset), batch_size), total=total_batches, desc="Processing batches"):
    batch = dataset[i:i + batch_size]
    embeddings = np.array(batch["emb"], dtype=np.float32)  # 确保"emb"是正确的字段名
    
    try:
        # 监控内存使用，调整批次大小
        memory = psutil.virtual_memory()
        if memory.available < 500 * 1024 * 1024:  # 如果可用内存小于500MB
            batch_size = max(10, batch_size // 2)  # 减少批次大小
        index.add(embeddings)  # 将嵌入添加到索引
        print(f"处理批次 {i//batch_size + 1}/{total_batches}，批次大小：{len(embeddings)}，当前批次大小：{batch_size}。")
    except Exception as e:
        print(f"处理过程中发生异常：{str(e)}，跳过当前批次。")
        continue

# 保存最终的索引
faiss.write_index(index, "./wikipedia_float32_bge_restored.index")
print("索引保存完毕。")