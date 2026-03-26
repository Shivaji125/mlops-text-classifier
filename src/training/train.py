"""Training pipeline with MLflow experiment tracking.
Works on local CPU, local GPU, or Google Colab GPU automatically.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path (works in Colab, local, and DVC)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from src.logging_config import get_logger

logger = get_logger(__name__, log_file="logs/training.log")


# ─── Device Detection ───────────────────────────────────────────────
def get_device():
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple MPS (Metal)")
        return torch.device("mps")
    else:
        logger.info("Using CPU (training will be slower)")
        return torch.device("cpu")


# ─── Dataset ────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# ─── Model Loader ───────────────────────────────────────────────────
def load_model_and_tokenizer(model_name: str, num_labels: int):
    """Load the correct model and tokenizer based on model name."""
    if "distilbert" in model_name:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    return model, tokenizer


# ─── Config Loader ──────────────────────────────────────────────────
def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── Training Loop ──────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Progress logging every 50 batches
        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"  Batch {batch_idx + 1}/{num_batches} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    return total_loss / num_batches


# ─── Evaluation ─────────────────────────────────────────────────────
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
        "precision": precision_score(all_labels, all_preds, average="weighted"),
        "recall": recall_score(all_labels, all_preds, average="weighted"),
    }
    return metrics, all_preds, all_labels


# ─── Main Training Function ─────────────────────────────────────────
def run_training(config_path: str = "configs/train_config.yaml"):
    config = load_config(config_path)
    device = get_device()

    # Set seed for reproducibility
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load data
    logger.info("Loading data...")
    train_df = pd.read_parquet(config["data"]["train_path"])
    val_df = pd.read_parquet(config["data"]["val_path"])
    test_df = pd.read_parquet(config["data"]["test_path"])
    logger.info(f"Data loaded — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load model and tokenizer
    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, config["model"]["num_labels"])
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters — Total: {total_params:,}, Trainable: {trainable_params:,}")

    # Create datasets
    logger.info("Tokenizing datasets...")
    train_dataset = SentimentDataset(
        train_df["text_clean"].tolist(), train_df["label"].tolist(),
        tokenizer, config["model"]["max_length"]
    )
    val_dataset = SentimentDataset(
        val_df["text_clean"].tolist(), val_df["label"].tolist(),
        tokenizer, config["model"]["max_length"]
    )
    test_dataset = SentimentDataset(
        test_df["text_clean"].tolist(), test_df["label"].tolist(),
        tokenizer, config["model"]["max_length"]
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["training"]["batch_size"],
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    total_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["training"]["warmup_ratio"]),
        num_training_steps=total_steps
    )

    # ─── MLflow Tracking ─────────────────────────────────────────────
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        # Log all hyperparameters
        mlflow.log_params({
            "model_name": model_name,
            "max_length": config["model"]["max_length"],
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "warmup_ratio": config["training"]["warmup_ratio"],
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "device": str(device),
            "total_params": total_params,
            "trainable_params": trainable_params,
        })

        # Log config file as artifact
        mlflow.log_artifact(config_path)

        # ─── Training Loop ───────────────────────────────────────────
        best_val_f1 = 0
        best_model_state = None

        for epoch in range(config["training"]["epochs"]):
            logger.info(f"{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
            logger.info(f"{'='*50}")

            # Train
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)

            # Validate
            val_metrics, _, _ = evaluate(model, val_loader, device)

            # Log metrics per epoch
            mlflow.log_metrics({
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "val_f1": round(val_metrics["f1"], 4),
                "val_precision": round(val_metrics["precision"], 4),
                "val_recall": round(val_metrics["recall"], 4),
            }, step=epoch)

            logger.info(
                f"Epoch {epoch + 1} Results — "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # Save best model
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.info(f"New best model! Val F1: {best_val_f1:.4f}")

        # ─── Test Evaluation ─────────────────────────────────────────
        logger.info(f"{'='*50}")
        logger.info("Evaluating best model on test set...")
        model.load_state_dict(best_model_state)
        model.to(device)
        test_metrics, test_preds, test_labels = evaluate(model, test_loader, device)

        # Log final test metrics
        mlflow.log_metrics({
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_f1": round(test_metrics["f1"], 4),
            "test_precision": round(test_metrics["precision"], 4),
            "test_recall": round(test_metrics["recall"], 4),
            "best_val_f1": round(best_val_f1, 4),
        })

        logger.info(
            f"Test Results — "
            f"Acc: {test_metrics['accuracy']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | "
            f"Precision: {test_metrics['precision']:.4f} | "
            f"Recall: {test_metrics['recall']:.4f}"
        )

        # ─── Save Artifacts ──────────────────────────────────────────
        # Confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        report_text = classification_report(
            test_labels, test_preds, target_names=["Negative", "Positive"]
        )
        cm_path = "confusion_matrix.txt"
        with open(cm_path, "w") as f:
            f.write(f"Confusion Matrix:\n{cm}\n\n")
            f.write(report_text)
        mlflow.log_artifact(cm_path)
        logger.info(f"\n{report_text}")

        # Log model
        logger.info("Logging model to MLflow...")
        mlflow.pytorch.log_model(model, "model")

        # Log tokenizer
        tokenizer_dir = "tokenizer_config"
        tokenizer.save_pretrained(tokenizer_dir)
        mlflow.log_artifacts(tokenizer_dir, "tokenizer")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Training complete! MLflow Run ID: {run_id}")
        logger.info(f"Best Val F1: {best_val_f1:.4f} | Test F1: {test_metrics['f1']:.4f}")

        # Clean up temp files
        Path(cm_path).unlink(missing_ok=True)

        return run_id


# ─── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    run_id = run_training(args.config)
    print(f"\nDone! Run ID: {run_id}")