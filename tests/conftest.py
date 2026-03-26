"""Shared test fixtures used across all test files."""
import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_raw_data(tmp_path):
    """Create a minimal valid raw dataset for testing."""
    import random
    random.seed(42)
    
    positive = [f"I love this product {i}" for i in range(300)]
    negative = [f"This is terrible {i}" for i in range(300)]
    
    df = pd.DataFrame({
        "text": positive + negative,
        "sentiment": [4] * 300 + [0] * 300,
    })
    path = tmp_path / "raw_data.parquet"
    df.to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def sample_processed_data(tmp_path):
    """Create minimal processed train/val/test splits."""
    for split in ["train", "val", "test"]:
        df = pd.DataFrame({
            "text_clean": [
                "love this product",
                "absolutely terrible",
                "great experience recommend",
                "worst purchase ever",
            ] * 50,
            "label": [1, 0, 1, 0] * 50,
        })
        path = tmp_path / f"{split}.parquet"
        df.to_parquet(path, index=False)
    return str(tmp_path)


@pytest.fixture
def empty_data(tmp_path):
    """Create a dataset with problematic entries."""
    df = pd.DataFrame({
        "text": ["Good", "", None, "   ", "Fine"] * 100,
        "sentiment": [4, 0, 4, 0, 4] * 100,
    })
    path = tmp_path / "empty_data.parquet"
    df.to_parquet(path, index=False)
    return str(path)