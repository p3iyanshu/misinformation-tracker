# api/main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re

app = FastAPI(
    title="Misinformation Tracker API",
    description="Classify news articles as Fake or Real using DistilBERT",
    version="1.0.0",
    docs_url=None  # disable default swagger
)

print("Loading model...")
MODEL_PATH = "outputs/models/distilbert-fakenews"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print(f"Model loaded on {device} ✅")

class NewsInput(BaseModel):
    title: str
    text: str

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(open("api/index.html").read())

@app.post("/predict")
def predict(news: NewsInput):
    input_text = clean(news.title) + " " + clean(news.text)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    label = "FAKE" if pred == 1 else "REAL"
    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "message": f"This article is {label} with {round(confidence * 100, 2)}% confidence"
    }