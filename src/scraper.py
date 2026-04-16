# src/scraper.py
# Download True/Fake news dataset from HuggingFace (GonzaloA/fake_news)

import pandas as pd
import os
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)

print("Downloading fake news dataset...")

dataset = load_dataset("GonzaloA/fake_news")

# Combine train, test, validation splits
train = dataset["train"].to_pandas()
test  = dataset["test"].to_pandas()
val   = dataset["validation"].to_pandas()

df = pd.concat([train, test, val], ignore_index=True)

print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")

df.to_csv("data/raw/fakenews_raw.csv", index=False)
print("\n✅ Saved to data/raw/fakenews_raw.csv")