# MLOps Text Classifier

A production-grade MLOps pipeline for BERT-based sentiment classification. This project demonstrates end-to-end ML engineering — from data versioning and experiment tracking to quality gates and automated deployment.

> **Note:** The model itself (fine-tuned BERT) is intentionally simple. The focus is on the infrastructure, tooling, and engineering practices that make ML systems reliable in production.

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  Data       │────▶│  Training   │────▶│  Evaluation  │
│  Pipeline   │     │  Pipeline   │     │  + Quality   │
│  (DVC)      │     │  (MLflow)   │     │    Gates     │
└─────────────┘     └─────────────┘     └──────┬───────┘
                                               │
                                               ▼ pass?
                                        ┌──────────────┐
                                        │   Model      │
                                        │   Registry   │
                                        │   (MLflow)   │
                                        └──────┬───────┘
                                               │
                    ┌──────────────┐     ┌──────▼───────┐
                    │  Monitoring  │◀────│   FastAPI    │
                    │ (Prometheus) │     │   Serving    │
                    └──────────────┘     └──────────────┘
                                               │
                    ┌──────────────┐            │
                    │   CI/CD      │────────────┘
                    │ (GitHub      │
                    │  Actions)    │
                    └──────────────┘
```

## Features

- **Data Versioning** — DVC tracks datasets and ensures reproducible pipelines
- **Data Validation** — Pydantic-based quality checks catch issues before training
- **Experiment Tracking** — MLflow logs hyperparameters, metrics, and model artifacts
- **Quality Gates** — Automated checks block deployment if metrics regress
- **Model Registry** — MLflow versioning with production stage promotion
- **API Serving** — FastAPI endpoint with health checks and input validation
- **Monitoring** — Prometheus metrics for latency, throughput, and prediction drift
- **CI/CD** — GitHub Actions for automated testing and deployment
- **Containerized** — Docker for reproducible environments

## Tech Stack

| Category | Tools |
|----------|-------|
| ML Framework | PyTorch, Hugging Face Transformers |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API | FastAPI, Uvicorn |
| Monitoring | Prometheus |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Testing | pytest |
| Linting | Ruff |

## Project Structure

```
mlops-text-classifier/
├── src/
│   ├── data/
│   │   ├── download.py          # Data fetching with metadata logging
│   │   ├── validate.py          # Data quality checks (nulls, duplicates, distribution)
│   │   └── preprocess.py        # Cleaning, label remapping, train/val/test split
│   ├── training/
│   │   ├── train.py             # Training loop with MLflow tracking
│   │   ├── config.py            # Dataclass-based config loader
│   │   └── evaluate.py          # Quality gates (threshold, regression, overfitting)
│   ├── serving/
│   │   ├── app.py               # FastAPI prediction endpoint
│   │   ├── schemas.py           # Request/response models
│   │   └── health.py            # Health check endpoints
│   ├── monitoring/
│   │   └── metrics.py           # Prometheus metrics
│   └── logging_config.py        # Centralized logging
├── configs/
│   ├── train_config.yaml        # Default training config
│   └── experiments/             # Per-experiment configs
│       ├── exp1_baseline.yaml
│       ├── exp2_higher_lr.yaml
│       ├── exp3_longer_training.yaml
│       └── exp4_distilbert.yaml
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_data_validation.py  # Data pipeline tests
│   ├── test_preprocessing.py    # Preprocessing tests
│   └── test_model.py            # Model smoke tests
├── notebooks/
│   └── training_colab.ipynb     # Colab GPU training notebook
├── dvc.yaml                     # DVC pipeline definition
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Quick Start

### Prerequisites

- Python 3.10+
- Git
- pip

### Setup

```bash
# Clone the repo
git clone https://github.com/shivaji125/mlops-text-classifier.git
cd mlops-text-classifier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init
```

### Run the Data Pipeline

```bash
dvc repro
```

This runs three stages in order:
1. **Download** — Fetches 500 tweets from Sentiment140 dataset
2. **Validate** — Checks for nulls, duplicates, label imbalance
3. **Preprocess** — Cleans text, remaps labels (0/1), splits into train/val/test

### Run Tests

```bash
pytest tests/ -v
```

### Train a Model

```bash
# Single experiment
python src/training/train.py --config configs/experiments/exp1_baseline.yaml

# Or use Colab for GPU training
# Open notebooks/training_colab.ipynb in Google Colab
```

### Run All Experiments

```bash
python src/training/train.py --config configs/experiments/exp1_baseline.yaml
python src/training/train.py --config configs/experiments/exp2_higher_lr.yaml
python src/training/train.py --config configs/experiments/exp3_longer_training.yaml
python src/training/train.py --config configs/experiments/exp4_distilbert.yaml
```

### View MLflow Dashboard

```bash
mlflow ui
# Open http://localhost:5000
```

### Run Quality Gates

```bash
python src/training/evaluate.py --run-id <YOUR_RUN_ID>
```

Quality gates check:
- **Absolute threshold** — F1 and accuracy must exceed 0.80
- **Regression check** — New model can't drop >2% vs production
- **Overfitting check** — Val-test F1 gap must be <5%

### Register Best Model

```python
import mlflow

client = mlflow.tracking.MlflowClient()
mv = mlflow.register_model("runs:/<BEST_RUN_ID>/model", "sentiment-classifier")
client.transition_model_version_stage("sentiment-classifier", mv.version, "Production")
```

## Experiments

| Experiment | Model | LR | Epochs | Batch | Test F1 | Test Acc |
|-----------|-------|-----|--------|-------|---------|----------|
| Baseline | distilbert-base-uncased | 2e-5 | 3 | 8 | — | — |
| Higher LR | distilbert-base-uncased | 5e-5 | 3 | 8 | — | — |
| Longer Training | distilbert-base-uncased | 2e-5 | 5 | 8 | — | — |

> **Fill in the results after running experiments.** Replace the dashes with actual metrics from your MLflow dashboard.

![alt text](image.png)

## Design Decisions

### Why DVC over Git for data?
Parquet files are binary and large. Git stores full copies of every version; DVC stores lightweight hashes and pushes actual data to configurable remote storage (S3, GCS, etc.). This keeps the repo fast while maintaining full data lineage and reproducibility.

### Why validate before training?
In production, data pipelines break silently — a schema change upstream, a null column from a logging bug, a sudden label imbalance. Validation catches these issues before you waste compute on a doomed training run. Our checks cover: null detection, empty strings, label distribution, duplicate rate, and text length sanity.

### Why quality gates?
Automated evaluation prevents deploying a model that's worse than what's already in production. Three gates run automatically: absolute metric thresholds (F1 > 0.80), regression detection (no more than 2% drop vs production), and overfitting detection (val-test gap < 5%). A failed gate returns exit code 1, which blocks CI/CD deployment.

### Why MLflow over W&B?
MLflow is open-source, self-hosted, and doesn't require an external account. For a portfolio project, this means anyone can clone the repo and see the experiment history without signing up for anything. MLflow also includes a model registry with stage management (Staging → Production), which integrates naturally with the quality gate workflow.

## Makefile Commands

```bash
make data       # Run DVC pipeline
make test       # Run all tests
make train      # Train with default config
make serve      # Start FastAPI server
make lint       # Run ruff linter
make clean      # Remove artifacts
```

## License

MIT

## Author

**Shivaji**
- GitHub: [github.com/shivaji125](https://github.com/Shivaji125)
- Email: 01shivaji10@gmail.com