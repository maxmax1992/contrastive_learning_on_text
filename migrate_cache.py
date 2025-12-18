import pandas as pd
import shutil

cache_file = "test_split_cache.csv"
backup_file = "test_split_cache.csv.bak"

# Backup
shutil.copy(cache_file, backup_file)

df = pd.read_csv(cache_file)
print(f"Columns before: {df.columns.tolist()}")

rename_map = {
    "emb_Baseline": "emb_BERT_Baseline",
    "emb_Trained": "emb_Contrastive_BERT"
}

df.rename(columns=rename_map, inplace=True)
print(f"Columns after: {df.columns.tolist()}")

df.to_csv(cache_file, index=False)
print("Cache updated.")
