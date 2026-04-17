# src/classifier.py
# Fine-tune DistilBERT for fake news classification

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import json

os.makedirs("outputs/models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load Data ────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv("data/processed/fakenews_clean.csv")
df = df.dropna(subset=["clean_title", "clean_text", "label"])

# Use title + text combined for better accuracy
df["input_text"] = df["clean_title"] + " " + df["clean_text"]

# Use 10,000 samples for faster training (still very good)
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

X = df["input_text"].tolist()
y = df["label"].astype(int).tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size : {len(X_train)}")
print(f"Test size  : {len(X_test)}")

# ── Tokenizer ────────────────────────────────────────────────
print("\nLoading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# ── Dataset Class ────────────────────────────────────────────
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids"      : self.encodings["input_ids"][idx],
            "attention_mask" : self.encodings["attention_mask"][idx],
            "labels"         : self.labels[idx]
        }

print("Tokenizing data (this takes a minute)...")
train_dataset = FakeNewsDataset(X_train, y_train, tokenizer)
test_dataset  = FakeNewsDataset(X_test,  y_test,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ── Model ────────────────────────────────────────────────────
print("\nLoading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
model.to(device)

# ── Training ─────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_steps  = num_epochs * len(train_loader)
scheduler  = get_scheduler("linear", optimizer=optimizer,
                            num_warmup_steps=0,
                            num_training_steps=num_steps)

print(f"\nStarting training for {num_epochs} epochs...")
print(f"Total steps: {num_steps}\n")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 25 == 0:
            print(f"  Epoch {epoch+1} | Step {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"\n✅ Epoch {epoch+1} complete — Avg Loss: {avg_loss:.4f}\n")

# ── Evaluation ───────────────────────────────────────────────
print("Evaluating on test set...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
report   = classification_report(all_labels, all_preds,
                                  target_names=["Real", "Fake"])

print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%")
print(f"\n📊 Classification Report:\n{report}")

# Save results
results = {"accuracy": round(accuracy, 4), "report": report}
with open("outputs/models/results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save model
model.save_pretrained("outputs/models/distilbert-fakenews")
tokenizer.save_pretrained("outputs/models/distilbert-fakenews")
print("\n✅ Model saved to outputs/models/distilbert-fakenews")
print("✅ Training complete!")