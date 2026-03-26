"""Preprocess and split the validated dataset for training."""
from pathlib import Path
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_logger

logger = get_logger(__name__, log_file="logs/data_preprocess.log")

def preprocess_data(
        input_path: str = "data/raw/sentiment140_500.parquet",
        output_dir: str = "data/processed",
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42,
        ):
    """
    Clean text and split into train/val/test sets.
    Sentiment140 labels: 0 = negative, 4 = positive -> remap to 0/1.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    # Remap labels: sentiment140 uses 0=neg, 4=positive
    df["label"] = df["sentiment"].map({0: 0, 4:1})
    df = df.dropna(subset=["label"])    # Drop any unmapped labels
    df["label"] = df["label"].astype(int)

    # Clean text
    df["text_clean"] = (
        df["text"]
        .str.replace(r"@\w+", "", regex=True)     # Rmeove @mentions
        .str.replace(r"http\S+", "", regex=True)    # Remove URLs
        .str.replace(r"#(\w+)", r"\1", regex=True)     # Remove # but keep word
        .str.replace(r"\s+", " ", regex=True)   # Normalize whitespace
        .str.strip()
    )

    # Remove empty after cleaning
    df = df[df["text_clean"].str.len() > 0]

    # Split train / va; / test
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size),
        random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size / (test_size + val_size),
        random_state=seed, stratify=temp_df["label"]
    )

    # Save splits
    train_df[["text_clean", "label"]].to_parquet(output_path / "train.parquet", index=False)
    val_df[["text_clean", "label"]].to_parquet(output_path / "val.parquet", index=False)
    test_df[["text_clean", "label"]].to_parquet(output_path / "test.parquet", index=False)

    logger.info(f"SPlit size - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Train label dist: {train_df['label'].value_counts(normalize=True).to_dict()}")

if __name__ == "__main__":
    preprocess_data()