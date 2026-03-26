"""Tests for the data validation pipeline."""
import pytest
import pandas as pd
import json
from pathlib import Path
from src.data.validate import validate_data, DataQualityReport


class TestDataValidation:
    """Test suite for data validation checks."""

    def test_valid_data_passes(self, sample_raw_data):
        """Valid dataset should pass all checks."""
        report = validate_data(sample_raw_data)
        assert report.passed is True
        assert report.null_text_count == 0
        assert report.null_label_count == 0
        assert len(report.failure_reasons) == 0

    def test_reports_correct_row_count(self, sample_raw_data):
        """Report should reflect actual dataset size."""
        report = validate_data(sample_raw_data)
        assert report.total_rows == 600

    def test_label_distribution_is_balanced(self, sample_raw_data):
        """Labels should not be severely skewed."""
        report = validate_data(sample_raw_data)
        for label, pct in report.label_distribution.items():
            assert pct >= 0.1, f"Label {label} underrepresented at {pct}"

    def test_null_text_detected(self, tmp_path):
        """Null text entries should be flagged."""
        df = pd.DataFrame({
            "text": ["Good", None, "Bad", None] * 250,
            "sentiment": [4, 0, 4, 0] * 250,
        })
        path = tmp_path / "null_text.parquet"
        df.to_parquet(path, index=False)
        with pytest.raises(SystemExit):
            validate_data(str(path))

    def test_skewed_labels_detected(self, tmp_path):
        """Severely imbalanced labels should fail validation."""
        df = pd.DataFrame({
            "text": ["text"] * 1000,
            "sentiment": [4] * 950 + [0] * 50,  # 95/5 split
        })
        path = tmp_path / "skewed.parquet"
        df.to_parquet(path, index=False)
        with pytest.raises(SystemExit):
            validate_data(str(path))

    def test_high_duplicate_rate_detected(self, tmp_path):
        """Too many duplicates should fail validation."""
        df = pd.DataFrame({
            "text": ["same text"] * 1000,  # 100% duplicates
            "sentiment": [4, 0] * 500,
        })
        path = tmp_path / "dupes.parquet"
        df.to_parquet(path, index=False)
        with pytest.raises(SystemExit):
            validate_data(str(path))

    def test_report_saved_to_disk(self, sample_raw_data, tmp_path, monkeypatch):
        """Validation should save a JSON report."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "raw").mkdir(parents=True)
        report = validate_data(sample_raw_data)
        report_path = tmp_path / "data" / "raw" / "validation_report.json"
        assert report_path.exists()
        saved = json.loads(report_path.read_text())
        assert saved["passed"] is True


class TestDataQualityReport:
    """Test the Pydantic report model itself."""

    def test_report_model_validates(self):
        """Report model should accept valid data."""
        report = DataQualityReport(
            total_rows=1000,
            null_text_count=0,
            null_label_count=0,
            empty_text_count=5,
            label_distribution={"0": 0.5, "4": 0.5},
            avg_text_length=45.2,
            min_text_length=3,
            max_text_length=280,
            duplicate_count=10,
            passed=True,
            failure_reasons=[],
        )
        assert report.total_rows == 1000
        assert report.passed is True