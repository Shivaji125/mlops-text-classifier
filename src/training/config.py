"""
Configuration loader for training pipeline.
Supports YAML config files with CLI override.
"""
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "distilbert/distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 128


@dataclass
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


@dataclass
class DataConfig:
    train_path: str = "data/processed/train.parquet"
    val_path: str = "data/processed/val.parquet"
    test_path: str = "data/processed/test.parquet"


@dataclass
class MLflowConfig:
    experiment_name: str = "sentiment-classifier"
    tracking_uri: str = "mlruns"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


def load_config(config_path: str | None = None) -> Config:
    """
    Load config from YAML file. Falls back to defaults if no file provided.
    """
    config = Config()

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        if "model" in raw:
            config.model = ModelConfig(**raw["model"])
        if "training" in raw:
            config.training = TrainingConfig(**raw["training"])
        if "data" in raw:
            config.data = DataConfig(**raw["data"])
        if "mlflow" in raw:
            config.mlflow = MLflowConfig(**raw["mlflow"])

    return config


def parse_args() -> Config:
    """Parse CLI arguments and load config."""
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()
    return load_config(args.config)