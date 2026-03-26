"""Pydantic models for API request and response validation."""
from pydantic import BaseModel, Field
from enum import Enum

class SentimentLabel(str, Enum):
    NEGATIVE = "negative"
    POSITIVE = "positive"

class PredictionRequest(BaseModel):
    """Single text prediction request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text to analyze for seniment",
        examples = ["I love this product!"]
    )

class BatchPredictionRequest(BaseModel):
    """Batch prediction request (up to 32 texts)."""
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of texts to analyze",
    )

class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    text: str
    label: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[PredictionResponse]
    count: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str | None = None
    version: str | None = None
    