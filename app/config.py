import os

LATENCY_BUDGET_MS = 50
DATA_PATH = "./data/msmarco_tri_sample.jsonl"
INDEX_PATH = "./data/faiss_index_fast.bin"
META_PATH = "./data/chunks_meta_fast.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"