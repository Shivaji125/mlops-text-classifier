"""Tests for the FastAPI serving endpoint."""
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client. Model loading may be skipped in test."""
    from src.serving.app import app
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_liveness(self, client):
        """Liveness probe should always return 200."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_health_endpoint_exists(self, client):
        """Health endpoint should exist."""
        response = client.get("/health")
        # 200 if model loaded, 200 with unhealthy status if not
        assert response.status_code == 200


class TestPredictionEndpoints:
    """Test prediction endpoints (requires model to be loaded)."""

    def test_predict_empty_text_rejected(self, client):
        """Empty text should be rejected by validation."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422  # Validation error

    def test_predict_missing_text_rejected(self, client):
        """Missing text field should be rejected."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_batch_empty_list_rejected(self, client):
        """Empty batch should be rejected."""
        response = client.post("/predict/batch", json={"texts": []})
        assert response.status_code == 422

    def test_predict_too_long_text_rejected(self, client):
        """Text exceeding max length should be rejected."""
        long_text = "a" * 1001
        response = client.post("/predict", json={"text": long_text})
        assert response.status_code == 422


class TestModelInfo:
    """Test model info endpoint."""

    def test_model_info_endpoint_exists(self, client):
        """Model info endpoint should exist."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "device" in data
        assert "uptime_seconds" in data