import json
import time
from pathlib import Path
import numpy as np
import faiss
import pandas as pd

from app.retriever import embed_query_onnx 

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "indic_datasets"
OUTPUT_INDEX = BASE_DIR / "data" / "faiss_index_hnsw.bin"
OUTPUT_META = BASE_DIR / "data" / "chunks_meta_hnsw.json"

def build_index():
    print(f"Searching for Hindi, Urdu, and English files inside: {BASE_DIR.resolve()}")
    
    target_patterns = ["*hin*", "*urd*", "*msmarco*", "*eng*"]
    all_files = []
    for pattern in target_patterns:
        all_files.extend(list(BASE_DIR.rglob(pattern)))
        if DATA_DIR.exists():
            all_files.extend(list(DATA_DIR.rglob(pattern)))
            
    unique_files = []
    seen = set()
    for f in all_files:
        if f.is_file() and f.suffix.lower() in [".parquet", ".jsonl"] and f.resolve() not in seen:
            seen.add(f.resolve())
            unique_files.append(f)

    print(f"Selected files: {[f.name for f in unique_files]}")
    
    if not unique_files:
        print("Error: No matching files found!")
        return

    print("Initializing FAISS HNSW Index...")
    d = 384
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 64 
    
    chunks_meta = []
    t_start = time.perf_counter()
    
    for filename in unique_files:
        print(f"\n--- Processing {filename.name} ---")
        try:
            texts = []
            if filename.suffix.lower() == ".parquet":
                df = pd.read_parquet(filename)
                
                # FIX: Robust Parquet Row Extraction
                if "translation" in df.columns:
                    # For standard nested Hugging Face translation dicts
                    for val in df["translation"].dropna():
                        if isinstance(val, dict):
                            texts.append(" | ".join(str(v) for v in val.values()))
                        else:
                            texts.append(str(val))
                else:
                    # For datasets with language-specific columns (like 'eng_Latn', 'hin_Deva')
                    # We extract all text columns and join the actual row data
                    valid_cols = [col for col in df.columns if df[col].dtype in ['object', 'string']]
                    if not valid_cols:
                        valid_cols = df.columns
                        
                    for _, row in df[valid_cols].iterrows():
                        row_text = " | ".join(str(val).strip() for val in row.values if pd.notna(val) and str(val).strip())
                        if row_text:
                            texts.append(row_text)
                            
            else:
                with open(filename, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        t = data.get("text", data.get("passage", ""))
                        if t.strip():
                            texts.append(t.strip())
            
            name_lower = filename.stem.lower()
            if "hin" in name_lower or "hi" in name_lower:
                lang_code = "hi"
            elif "urd" in name_lower or "ur" in name_lower:
                lang_code = "ur"
            else:
                lang_code = "en"
                
            split_type = "val" if "val" in name_lower else "train"
            total_items = len(texts)
            
            for idx, text_chunk in enumerate(texts):
                text_chunk = str(text_chunk).strip()
                if not text_chunk:
                    continue
                    
                vector_tuple, _ = embed_query_onnx(text_chunk)
                vector = np.ascontiguousarray(np.array(vector_tuple, dtype=np.float32).reshape(1, -1), dtype=np.float32)
                index.add(vector)
                
                chunks_meta.append({
                    "text": text_chunk,
                    "lang": lang_code,
                    "split": split_type
                })
                
                # Print live progress every 50 items
                if (idx + 1) % 50 == 0 or (idx + 1) == total_items:
                    percent = ((idx + 1) / total_items) * 100
                    print(f"[{filename.name}] Progress: {idx + 1}/{total_items} ({percent:.1f}%)", end="\r")
            print() # New line after file completes
            
        except Exception as e:
            print(f"Error processing {filename.name}: {e}")

    elapsed = (time.perf_counter() - t_start) / 60
    print(f"\nFinished! Embedded {len(chunks_meta)} chunks in {elapsed:.2f} minutes.")
    
    print("Saving Index and Metadata...")
    faiss.write_index(index, str(OUTPUT_INDEX))
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False, indent=2)
        
    print("HNSW index successfully built!")

if __name__ == "__main__":
    build_index()