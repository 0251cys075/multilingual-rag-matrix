import time
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# 1. Load the compiled dataset
DATA_PATH = "./data/msmarco_tri_sample.jsonl"
print(f"Loading dataset from {DATA_PATH}...")
df = pd.read_json(DATA_PATH, lines=True)

# 2. Multi-Strategy Chunking Implementation
chunks = []
for _, row in df.iterrows():
    target_lang = row.get("target_lang", "unknown")
    query_id = str(row.get("query_id", "0"))
    
    # Grab translated passages list
    passages_dict = row.get("passages", {})
    trans_passages = passages_dict.get("Translated_passages", [])
    if isinstance(trans_passages, str):
        trans_passages = [trans_passages]
        
    for p_idx, p_text in enumerate(trans_passages):
        if not p_text or not p_text.strip():
            continue
        
        # Strategy A: Semantic / Sentence Boundary Chunking (Indic & Urdu terminals)
        sentences = [
            s.strip() 
            for s in p_text.replace("।", ".").replace("۔", ".").replace("?", ".").split(".") 
            if len(s.strip().split()) >= 3
        ]
        
        for c_idx, sentence in enumerate(sentences):
            chunks.append({
                "text": sentence,
                "metadata": {
                    "lang": target_lang,
                    "query_id": query_id,
                    "passage_idx": p_idx,
                    "chunk_idx": c_idx,
                    "strategy": "semantic_boundary"
                }
            })

print(f"Total chunks extracted: {len(chunks)}")

# 3. Vector Embedding with Multilingual Model
print("Loading multilingual embedding model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Encoding chunks into vector space...")
t_emb_start = time.perf_counter()
chunk_embeddings = model.encode(
    [c["text"] for c in chunks],
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)
print(f"Embedding completed in {(time.perf_counter() - t_emb_start):.2f}s")

# 4. Build In-Memory FAISS Flat Index (Cosine Similarity via Inner Product)
dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(chunk_embeddings, dtype=np.float32))
print(f"FAISS Index ready with {index.ntotal} vectors in RAM.\n")

# 5. Fast Retrieval Function with Precision Timing
def search_index(query: str, top_k: int = 3, target_lang: str = None):
    t0 = time.perf_counter()
    
    # Step A: Query Embedding
    t_q0 = time.perf_counter()
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    q_emb_ms = (time.perf_counter() - t_q0) * 1000.0
    
    # Step B: Vector Search
    t_s0 = time.perf_counter()
    scores, indices = index.search(np.array(q_emb, dtype=np.float32), top_k * 4 if target_lang else top_k)
    search_ms = (time.perf_counter() - t_s0) * 1000.0
    
    # Step C: Metadata Filtering (if language specified)
    matched_results = []
    matched_scores = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        item = chunks[idx]
        if target_lang and item["metadata"]["lang"] != target_lang:
            continue
        matched_results.append(item)
        matched_scores.append(float(score))
        if len(matched_results) == top_k:
            break
            
    total_ms = (time.perf_counter() - t0) * 1000.0
    return matched_results, matched_scores, {"query_embed_ms": q_emb_ms, "faiss_search_ms": search_ms, "total_ms": total_ms}

# 6. Test Multi-lingual Query Retrivals
test_cases = [
    ("कॉरपोरेशन का मतलब क्या है?", "hin_Deva"),
    ("دمہ کی علامات کیا ہیں؟", "urd_Arab"),
]

print("="*60)
print("  SAMPLE RETRIEVAL LATENCY AUDIT")
print("="*60)

for q_text, lang in test_cases:
    results, scores, timings = search_index(q_text, top_k=2, target_lang=lang)
    print(f"\nQuery: '{q_text}' [{lang}]")
    print(f"Timings -> Embed: {timings['query_embed_ms']:.2f}ms | Search: {timings['faiss_search_ms']:.2f}ms | Total: {timings['total_ms']:.2f}ms")
    for r, s in zip(results, scores):
        print(f" -> [Sim: {s:.3f}] {r['text'][:80]}...")