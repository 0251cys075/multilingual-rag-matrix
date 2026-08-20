import os
import pandas as pd
from huggingface_hub import hf_hub_download

OUTPUT_DIR = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "msmarco_tri_sample.jsonl")

# Direct relative file paths in ai4bharat/MSMARCO-XI repo
TARGET_FILES = [
    ("validation/hinval.parquet", "hin_Deva", 200),
    ("validation/urdval.parquet", "urd_Arab", 200),
    ("validation/engval.parquet", "eng_Latn", 100),
]

all_records = []

for file_path, lang_code, max_rows in TARGET_FILES:
    try:
        print(f"Downloading {file_path}...")
        local_file = hf_hub_download(
            repo_id="ai4bharat/MSMARCO-XI",
            filename=file_path,
            repo_type="dataset",
            local_dir=OUTPUT_DIR,
            resume_download=True
        )
        print(f"Reading {file_path} into memory...")
        df_sub = pd.read_parquet(local_file)
        df_sub = df_sub.head(max_rows)
        all_records.append(df_sub)
        print(f"Loaded {len(df_sub)} rows for {lang_code}.")
    except Exception as e:
        print(f"Could not load {file_path}: {e}")

if all_records:
    final_df = pd.concat(all_records, ignore_index=True)
    final_df.to_json(OUTPUT_FILE, orient="records", lines=True, force_ascii=False)
    print(f"\nSuccessfully compiled dataset to {OUTPUT_FILE} ({len(final_df)} total records).")
    print(final_df.head(2))
else:
    print("\nDownload failed. Check your internet connection or DNS.")