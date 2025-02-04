import argparse
import json
import pandas as pd
from datasets import load_dataset
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # 引入tqdm

# 解析命令行参数
parser = argparse.ArgumentParser(description="Search Wikipedia data and append results.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file")
parser.add_argument("--top_k", type=int, default=100, help="Number of documents to retrieve")
args = parser.parse_args()

# 加载模型和tokenizer
hf_token = "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"
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
model.eval()

# 加载数据集
dataset = load_dataset("WhereIsAI/bge_wikipedia-data-en", split="train", token=hf_token)

# 加载索引
index_path = "./quantized-retrieval/wikipedia_float32_bge.index"
float_index = faiss.read_index(index_path)

# 函数：生成嵌入向量
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

# 函数：搜索并创建DataFrame
def search_and_create_dataframe(query, top_k):
    query_embedding = generate_embeddings(query)
    D, I = float_index.search(query_embedding, top_k)
    top_k_titles = [dataset[int(idx)]['title'] for idx in I[0]]
    top_k_texts = [dataset[int(idx)]['text'] for idx in I[0]]
    return pd.DataFrame({
        "Score": D[0],
        "Title": top_k_titles,
        "Text": top_k_texts
    })

# 读取输入文件，处理每行，写入输出文件
def process_json(input_file, output_file, top_k):
    """
    Processes a JSON file containing queries and creates a new JSON file with query and context lists.

    Args:
    - input_file (str): The path to the input JSON file containing queries.
    - output_file (str): The path to the output JSON file with query and context lists.
    - top_k (int): The number of top contexts to include for each query.

    Returns:
    - None
    """
    # Read the input JSON file
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    # Process each entry in the data
    for entry in tqdm(data, desc="Processing queries"):  # Add tqdm progress bar
        query = entry["question"]
        result_df = search_and_create_dataframe(query, top_k)

        # Construct output context lists
        ctxs = [{"title": row["Title"], "text": row["Text"], "score": row["Score"]} for index, row in result_df.iterrows()]
        entry["ctxs"] = ctxs
        print(f"Processed entry: {entry['question']}")

    # Write the modified data to the output JSON file
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

process_json(args.input_file,args.output_file,args.top_k)
# with open(args.input_file, 'r') as infile, open(args.output_file, 'w') as outfile:
#     for line in tqdm(infile, desc="Processing queries"):  # 添加tqdm进度条
#         data = json.loads(line)
#         query = data["question"]
#         result_df = search_and_create_dataframe(query, args.top_k)

#         # 构造输出上下文列表
#         ctxs = [{"title": row["Title"], "text": row["Text"], "score": row["Score"]} for index, row in result_df.iterrows()]
#         data["ctxs"] = ctxs

#         # 将修改后的字典写回新的JSONL文件
#         json.dump(data, outfile)
#         outfile.write("\n")
