"""Validate raw data before training. Catches data quality isssues early."""
import os
import json
import sys
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_logger


logger = get_logger(__name__, log_file="logs/data_validate.log")

class DataQualityReport(BaseModel):
    """Quality report for the dataset."""
    total_rows: int
    null_text_count: int
    null_label_count: int
    empty_text_count: int
    label_distribution: dict
    avg_text_length: float
    min_text_length: int
    max_text_length: int
    duplicate_count: int
    passed: bool
    failure_reasons: list[str]

def validate_data(data_path: str = "data/raw/sentiment140_500.parquet") -> DataQualityReport:
    """Run validation checks on the raw dataset.
    Returns a quality report. Fails if critical chexks don't pass.
    """
    df = pd.read_parquet(data_path)
    failures = []

    # Check 1: No Null text
    null_text = df["text"].isna().sum()
    if null_text > 0:
        failures.append(f"Found {null_text} null text entries.")
    
    # Check 2: No null labels
    null_label = df["sentiment"].isna().sum()
    if null_label > 0:
        failures.append(f"found{null_label} null label entries.")

    # Check 3: Empty strings
    empty_text = (df["text"].str.strip() == "").sum()
    if empty_text > len(df) * 0.01:
        failures.append(f"Too many empty texts: {empty_text} ({(empty_text/len(df))*100:.1f}%)")

    # Check 4: Label distribution - not extremely skewed
    label_dist = df["sentiment"].value_counts(normalize=True).to_dict()
    for label, pct in label_dist.items():
        if pct < 0.1:   # No class should be  < 10%
            failures.append(f"Label '{label}' is severely underrepresented: {pct:.1%} ")

    # Check 5: Text length sanity
    text_lengths =  df["text"].str.len()
    if text_lengths.min() == 0:
        failures.append("Found zero-length text entries")

    # Check 6: Duplicates
    dup_count = df.duplicated(subset=["text"]).sum()
    if dup_count > len(df) * 0.05:
        failures.append(f"Too many duplicates: {dup_count} ({dup_count/len(df)*100:.1f}%)")

    report = DataQualityReport(
        total_rows=len(df),
        null_text_count=int(null_text),
        null_label_count=int(null_label),
        empty_text_count=int(empty_text),
        label_distribution={str(k): round(v, 4) for k, v in label_dist.items()},
        avg_text_length=round(text_lengths.mean(), 1),
        min_text_length=int(text_lengths.min()),
        max_text_length=int(text_lengths.max()),
        duplicate_count=int(dup_count),
        passed=len(failures) == 0,
        failure_reasons=failures,
    )

    # Save report
    report_path = Path("data/raw/validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    
    if not report.passed:
        logger.error(f"DATA VALIDATION FAILED: {failures}")
        sys.exit(1)
    else:
        logger.info(f"Data validation PASSED: {report.total_rows} rows,"
                    f"dist: {report.label_distribution}")
    return report

if __name__ == "__main__":
    validate_data()