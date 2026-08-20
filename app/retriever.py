import os
os.environ["OMP_NUM_THREADS"] = "4"

import json
import time
import functools
from pathlib import Path
import faiss
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "data" / "faiss_index_hnsw.bin"
META_PATH = BASE_DIR / "data" / "chunks_meta_hnsw.json"
MODEL_PATH = BASE_DIR / "onnx-multilingual-minilm"

# Load Tokenizer & ONNX Session
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, fix_mistral_regex=True)

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 4  
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(str(MODEL_PATH / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])

# Load FAISS Index
index = faiss.read_index(str(INDEX_PATH))
index.hnsw.efSearch = 16  

with open(META_PATH, "r", encoding="utf-8") as f:
    chunks_meta = json.load(f)

@functools.lru_cache(maxsize=1024)
def embed_query_onnx(text: str):
    """Encodes query text into a vector embedding using ONNX, with memory caching."""
    # Strict 24-token limit for sub-200ms performance
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=24,
        return_tensors="np"
    )
    
    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64)
    }
    
    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
    
    outputs = session.run(None, inputs)
    embeddings = outputs[0]
    
    mask = inputs["attention_mask"][:, :, None]
    sum_embeddings = np.sum(embeddings * mask, axis=1)
    sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = sum_embeddings / sum_mask
    
    pooled = np.ascontiguousarray(pooled, dtype=np.float32)
    faiss.normalize_L2(pooled)
    
    return tuple(pooled[0].tolist()), text

def search_index(query: str, top_k: int = 5):
    t0 = time.perf_counter()
    vector_tuple, _ = embed_query_onnx(query)
    t1 = time.perf_counter()
    
    query_vector = np.array(vector_tuple, dtype=np.float32).reshape(1, -1)
    embed_ms = (t1 - t0) * 1000.0 
    
    t2 = time.perf_counter()
    distances, indices = index.search(query_vector, top_k)
    t3 = time.perf_counter()
    search_ms = (t3 - t2) * 1000.0
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(chunks_meta):
            item = chunks_meta[idx].copy()
            item["score"] = float(dist)
            results.append(item)
            
    return results, embed_ms, search_ms

search = search_index