# src/preprocess.py
# Clean and prepare the dataset for EDA and model training

import pandas as pd
import re
import os

os.makedirs("data/processed", exist_ok=True)

print("Loading raw data...")
df = pd.read_csv("data/raw/fakenews_raw.csv")

# Drop unnamed column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Drop rows with missing text or title
df = df.dropna(subset=["text", "title"])

# Clean text function
def clean_text(text):
    text = str(text).lower()                          # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)           # remove special chars
    text = re.sub(r"\s+", " ", text).strip()          # remove extra spaces
    return text

print("Cleaning text...")
df["clean_title"] = df["title"].apply(clean_text)
df["clean_text"]  = df["text"].apply(clean_text)

# Add text length columns (useful for EDA)
df["title_length"] = df["clean_title"].apply(lambda x: len(x.split()))
df["text_length"]  = df["clean_text"].apply(lambda x: len(x.split()))

# Rename label for clarity
df["label_name"] = df["label"].map({0: "real", 1: "fake"})

print(df.head())
print(f"\nShape after cleaning: {df.shape}")
print(f"\nLabel distribution:\n{df['label_name'].value_counts()}")

df.to_csv("data/processed/fakenews_clean.csv", index=False)
print("\n✅ Saved to data/processed/fakenews_clean.csv")