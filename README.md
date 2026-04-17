# 🔍 Twitter/X Misinformation Spread Tracker

A end-to-end data science project that detects and tracks how misinformation spreads using NLP and network analysis. Built as part of my learning journey in B.Tech CSE (Data Science), Semester 4.

---

## 📌 Project Overview

Misinformation spreads faster than truth on social media. This project aims to:
- **Classify** news articles as Fake or Real using a fine-tuned DistilBERT model
- **Analyse** how fake vs real news spreads through networks using graph theory
- **Visualise** the spread patterns using NetworkX and Gephi
- **Deploy** a REST API so anyone can classify a news article in real time

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **98.35%** |
| Fake Precision | 0.99 |
| Fake Recall | 0.98 |
| Real Precision | 0.98 |
| Real Recall | 0.99 |
| Macro F1-Score | 0.98 |
| Model | DistilBERT (fine-tuned) |
| Dataset Size | 40,587 labelled articles |
| Training Device | NVIDIA RTX 4060 Laptop GPU |

---

## 🗂️ Project Structure

misinformation-tracker/
│
├── data/
│   ├── raw/                  ← Downloaded dataset (not pushed to GitHub)
│   └── processed/            ← Cleaned and preprocessed CSV
│
├── src/
│   ├── scraper.py            ← Downloads the FakeNews dataset
│   ├── preprocess.py         ← Cleans and prepares data for training
│   ├── eda.py                ← Exploratory Data Analysis + graphs
│   ├── graph_builder.py      ← Network graph analysis using NetworkX
│   └── classifier.py        ← Fine-tunes DistilBERT for classification
│
├── api/
│   └── main.py               ← FastAPI REST endpoint for predictions
│
├── outputs/
│   ├── graphs/               ← Saved visualisation plots + .gexf file
│   └── models/               ← Saved DistilBERT model weights
│
├── requirements.txt
└── README.md
---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| NLP Model | DistilBERT (HuggingFace Transformers) |
| Deep Learning | PyTorch (CUDA) |
| Network Analysis | NetworkX |
| Visualisation | Matplotlib, Seaborn, Gephi |
| API | FastAPI, Uvicorn |
| Data | Pandas, NumPy |
| ML Utilities | Scikit-learn |

---

## 📊 Dataset

- **Source:** GonzaloA/fake_news via HuggingFace Datasets
- **Size:** 40,587 articles
- **Labels:** 0 = Real, 1 = Fake
- **Columns:** title, text, label
- **Split:** 80% train / 20% test

> ⚠️ Data files are not pushed to GitHub due to size limits.
> Run `python src/scraper.py` to download the dataset locally.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/p3iyanshu/misinformation-tracker.git
cd misinformation-tracker
```

### 2. Create virtual environment (Python 3.11 required)
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA (for GPU training)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Run the Project

### Step 1 — Download Data
```bash
python src/scraper.py
```

### Step 2 — Preprocess
```bash
python src/preprocess.py
```

### Step 3 — EDA
```bash
python src/eda.py
```

### Step 4 — Network Analysis
```bash
python src/graph_builder.py
```

### Step 5 — Train Classifier
```bash
python src/classifier.py
```
> ⏱️ Training takes ~15-20 mins on RTX 4060. Uses GPU automatically if available.

### Step 6 — Run API
```bash
uvicorn api.main:app --reload
```
Then open: http://127.0.0.1:8000/docs

---

## 🌐 API Usage

### Endpoint: POST /predict

**Request:**
```json
{
  "title": "Your news article title here",
  "text": "Full body of the news article here"
}
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 97.25,
  "message": "This article is FAKE with 97.25% confidence"
}
```

---

## 📉 EDA Findings

- Dataset contains **21,924 fake** and **18,663 real** articles — well balanced
- Fake news articles are on average **shorter** (~384 words) than real news (~424 words)
- Fake news titles tend to be more **sensational and shorter** in length
- In the spread network, fake news generates **~6x more nodes** than real news

---

## 🕸️ Network Analysis Findings

- Built a directed graph with **500 source nodes** and **6,429 edges**
- Fake news source nodes have **3-10 spreaders** on average
- Real news source nodes have **1-5 spreaders** on average
- Graph exported as `.gexf` for advanced visualisation in Gephi

---

## 🔮 Future Improvements

- [ ] Collect real Twitter/X retweet data using Apify scraper
- [ ] Add temporal spread analysis (how fast fake news spreads over time)
- [ ] Build a Streamlit frontend for non-technical users
- [ ] Deploy the API on Render or Railway for public access
- [ ] Experiment with RoBERTa for potentially higher accuracy

---

## 👨‍💻 About

Made by **Priyanshu** — B.Tech CSE (Data Science), Semester 4

This is one of my learning projects where I tried to apply real-world
data science concepts including NLP, graph theory, and API development
from scratch.

📧 Feel free to reach out or raise an issue if you find any bugs!

---
