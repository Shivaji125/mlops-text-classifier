"""
FastAPI application for sentiment predictin
Loads model from Mlflow registry or local path.
"""
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.serving.schemas import(
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse, 
    HealthResponse,
)
from src.serving.health import get_uptime, check_model_health
from src.logging_config import get_logger

logger = get_logger(__name__, log_file="logs/serving.log")

MODEL = None
TOKENIZER = None
DEVICE = None
MODEL_INFO = {"name": None, "version": None, "run_id": None}

def load_model():
    """Load model from Mlflow registry or fallback to latest run."""
    global MODEL, TOKENIZER, DEVICE, MODEL_INFO

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else  "cpu")
    logger.info(f"Using device: {DEVICE}")

    mlflow.set_tracking_uri("mlruns")

    # Try loading from MLflow registry 
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("sentiment-classifier", stages=["Production"])

        if versions:
            version = versions[0]
            run_id = version.run_id
            logger.info(f"Loading production model v{version.version} (run: {run_id})")
        else:
            # Fallback: get the best run from experiments
            logger.info("NO production model found, loading best experiment run...")
            experiment = mlflow.get_experiment_by_name("sentiment-classifier")
            if experiment is None:
                raise RuntimeError("NO experiment found. Train a model first.")
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="metrics.test_f1 > 0",
                order_by=["metrics.test_f1 DESC"],
                max_results=1,
            )
            if runs.empty:
                raise RuntimeError("NO successful training runs found.")
            run_id = runs.iloc[0]["run_id"]
            version = None
            logger.info(f"Loading best run: {run_id}")
        
        # Load model
        model_uri = f"runs:/{run_id}/model"
        MODEL = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
        MODEL.eval()

        # Load tokenizer
        tokenizer_path = mlflow.artifacts.download_artifacts(
            run_id = run_id, artifact_path="tokenizer"
        )

        # Detect model type for correct tokenizer
        from transformers import BertTokenizer, DistilBertTokenizer
        try:
            TOKENIZER = BertTokenizer.from_pretrained(tokenizer_path)
        except Exception:
            TOKENIZER = DistilBertTokenizer.from_pretrained(tokenizer_path)

        MODEL_INFO = {
            "name": "sentiment-classifier",
            "version": str(version.version) if version else "latest",
            "run_id": run_id,
        }
        logger.info(f"Model loaded successfully: {MODEL_INFO}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("Starting up - loading model...")
    try:
        load_model()
        logger.info("Model loaded, server ready!")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
    yield
    logger.info("Shutting down...")

# FastAPI App
app = FastAPI(
    title= "Sentiment Classifier API",
    description="Production ML API for text sentiment analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_sentiment(text: str):
    """Run inference on a single text."""
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    inputs = TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    inputs = {k: v.to(DEVICE) for k,v in inputs.items()}

    with torch.no_grad():
        outputs = MODEL(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()

    label = "positive" if pred_class == 1 else "negative"

    return {
        "text": text,
        "label": label,
        "confidence": round(confidence, 4),
    }

# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are healthy."""
    model_health = check_model_health(MODEL, TOKENIZER)
    return HealthResponse(
        status="healthy" if model_health["healthy"] else "unhealthy",
        model_loaded=MODEL is not None,
        model_name=MODEL_INFO.get("name"),
        version=MODEL_INFO.get("version"),
    )

@app.get("/health/ready")
async def readiness_check():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "uptime_seconds": get_uptime()}

@app.get("/health/live")
async def liveness_check():
    return {"status": "alive", "uptime_seconds":get_uptime()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict sentiment for a single text."""
    start = time.time()
    result = predict_sentiment(request.text)
    latency = round((time.time() - start) * 1000, 2)
    logger.info(f"Prediction: '{request.text[:50]}...'->{result['label']} ({result['confidence']}) [{latency}ms]")
    return PredictionResponse(**result)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict sentiment for multiple texts (up to 32)."""
    start = time.time()
    predictions = [predict_sentiment(text) for text in request.texts]
    latency = round((time.time() - start) * 1000, 2)
    logger.info(f"Batch prediction: {len(predictions)} texts [{latency}ms]")
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**p) for p in predictions],
        count = len(predictions),
    )

@app.get("/model/info")
async def model_info():
    """Return model metadata."""
    return {
        **MODEL_INFO, 
        "device": str(DEVICE),
        "uptime_seconds": get_uptime()
    }