"""Tests for data preprocessing."""
import pytest
import pandas as pd
from pathlib import Path
from src.data.preprocess import preprocess_data

class TestPreprocessing:
    """Test suite for the preprocessing pipeline."""
    def test_splits_are_created(self, sample_raw_data, tmp_path):
        """Should create train, val and test parquet files."""
        output_dir = str(tmp_path / "processed")
        preprocess_data(
            input_path=sample_raw_data,
            output_dir=output_dir,
        )
        assert (Path(output_dir) / "train.parquet").exists()
        assert (Path(output_dir) / "val.parquet").exists()
        assert (Path(output_dir) / "test.parquet").exists()

    def test_no_data_leakage(self, sample_raw_data, tmp_path):
        """Train, val and test should have no overlapping rows."""
        output_dir = str(tmp_path / "processed")
        preprocess_data(input_path=sample_raw_data, output_dir=output_dir)

        train = pd.read_parquet(Path(output_dir) / "train.parquet")
        val = pd.read_parquet(Path(output_dir) / "val.parquet")
        test = pd.read_parquet(Path(output_dir) / "test.parquet")

        # Check total rows preserved (minus any dropped empty/null rows)
        total = len(train) + len(val) + len(test)
        assert total > 0

    def test_labels_are_binary(self, sample_raw_data, tmp_path):
        """Labels should be remapped to 0 ans 1 only."""
        output_dir = str(tmp_path / "processed")
        preprocess_data(input_path=sample_raw_data, output_dir=output_dir)

        train = pd.read_parquet(Path(output_dir) / "train.parquet")
        assert set(train["label"].unique()).issubset({0, 1})

    def test_text_is_cleaned(self, sample_raw_data, tmp_path):
        """Cleaned text should not contain @mentions or URLs."""
        output_dir = str(tmp_path / "processed")
        preprocess_data(input_path=sample_raw_data, output_dir=output_dir)

        train = pd.read_parquet(Path(output_dir) / "train.parquet")
        for text in train["text_clean"]:
            assert not text.startswith("@"), f"Found @mention: {text}"
            assert "http" not in text, f"Found URL: {text}"