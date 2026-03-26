"""
Lightweight FastAPI app for Render deployment.
Loads moel from HUggingface Hub instead of local Mlflow.
"""
import os
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Config
HF_MODEL_REPO = os.getenv ("HF_MODEL_REPO", "sickstart/sentiment_classifier_01")
MAX_LENGTH = 128

# schemas
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_repo: str

# Global state
model = None
tokenizer = None
device = None

# App
app = FastAPI(
    title="Sentiment Classifier API",
    description="Production ML API - BERT-based sentiment analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_Event("startup")
async def load_model():
    global model, tokenizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {HF_MODEL_REPO}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO, subfolder="tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_REPO, subfolder="model"
        )
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load from HF subfolder, trying direct load: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
            model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO)
            model.eval()
            print("Model loaded successfully (direct)")
        except Exception as e2:
            print(f"Model loading failed : {e2}")

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded = model is not None,
        model_repo = HF_MODEL_REPO,
    )

@app.get("/")
async def root():
    return {
        "message": "sentiment classifier API",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/predict", response_model = PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    inputs = tokenizer(
        request.text, return_tensors="pt", truncation=True,
        padding="max_length", max_length = MAX_LENGTH,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()
    
    label = "positive" if pred_class == 1 else "negative"
    latency = round((time.time() - start) * 1000, 2)
    print(f"prediction:'{request.text[:50]}' -> {label} ({confidence:.4f}) [{latency}ms]")

    return PredictionResponse(
        text = request.text,
        label = label,
        confidence=round(confidence, 4),
    )