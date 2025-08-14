import faiss
import numpy as np
import os
import psutil
from datasets import load_dataset
from tqdm import tqdm

# Create a CPU index, using inner product as the distance metric
index = faiss.IndexFlatIP(768)  # Using inner product
print("CPU index initialization complete.")

# Load the dataset
hf_token = "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"
dataset = load_dataset("WhereIsAI/bge_wikipedia-data-en", split="train", token=hf_token)
print("Dataset loaded.")

# Initialize batch size
initial_batch_size = 5000
batch_size = initial_batch_size

# Process the batches
total_batches = (len(dataset) + batch_size - 1) // batch_size  # Ensure all data is included

for i in tqdm(range(0, len(dataset), batch_size), total=total_batches, desc="Processing batches"):
    batch = dataset[i:i + batch_size]
    embeddings = np.array(batch["emb"], dtype=np.float32)  # Ensure "emb" is the correct field name
    
    try:
        # Monitor memory usage and adjust batch size
        memory = psutil.virtual_memory()
        if memory.available < 500 * 1024 * 1024:  # If available memory is less than 500MB
            batch_size = max(10, batch_size // 2)  # Reduce batch size
        index.add(embeddings)  # Add embeddings to the index
        print(f"Processed batch {i//batch_size + 1}/{total_batches}, batch size: {len(embeddings)}, current batch size: {batch_size}.")
    except Exception as e:
        print(f"Exception occurred during processing: {str(e)}, skipping current batch.")
        continue

# Save the final index
faiss.write_index(index, "./wikipedia_float32_bge_restored.index")
print("Index saved.")