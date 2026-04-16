# src/eda.py
# Exploratory Data Analysis on the fake news dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs/graphs", exist_ok=True)

print("Loading cleaned data...")
df = pd.read_csv("data/processed/fakenews_clean.csv")

print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nLabel distribution:\n{df['label_name'].value_counts()}")

# ── Plot 1: Label Distribution ──────────────────────────────
plt.figure(figsize=(6,4))
sns.countplot(x="label_name", data=df, palette={"fake":"#e74c3c","real":"#2ecc71"})
plt.title("Fake vs Real News Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/graphs/label_distribution.png")
plt.show()
print("✅ Saved label_distribution.png")

# ── Plot 2: Text Length Distribution ────────────────────────
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(data=df, x="text_length", hue="label_name", bins=50,
             palette={"fake":"#e74c3c","real":"#2ecc71"})
plt.title("Article Text Length")
plt.xlim(0, 2000)

plt.subplot(1,2,2)
sns.histplot(data=df, x="title_length", hue="label_name", bins=30,
             palette={"fake":"#e74c3c","real":"#2ecc71"})
plt.title("Title Length")

plt.tight_layout()
plt.savefig("outputs/graphs/text_length_distribution.png")
plt.show()
print("✅ Saved text_length_distribution.png")

# ── Plot 3: Average text length by label ────────────────────
plt.figure(figsize=(6,4))
df.groupby("label_name")["text_length"].mean().plot(
    kind="bar",
    color=["#2ecc71","#e74c3c"],
    edgecolor="black"
)
plt.title("Average Article Length: Fake vs Real")
plt.ylabel("Average Word Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/graphs/avg_text_length.png")
plt.show()
print("✅ Saved avg_text_length.png")

# ── Summary Stats ────────────────────────────────────────────
print("\n=== Fake News Stats ===")
print(df[df["label_name"]=="fake"][["text_length","title_length"]].describe())

print("\n=== Real News Stats ===")
print(df[df["label_name"]=="real"][["text_length","title_length"]].describe())

print("\n✅ EDA complete!")