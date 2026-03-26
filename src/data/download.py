import os
import json
import sys
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config  import get_logger

logger = get_logger(__name__, log_file="logs/download.log")

def download_data(output_dir: str = "data/raw") -> dict:
    """
    Download Twitter sentiment140 dataset from Hugging Face.
    Returns metadata about the downloaded dataset.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading sentiment140 dataset...")
    dataset = load_dataset("sentiment140", split="train", trust_remote_code=True)

    # Sample 50k for manageable training 
    dataset = dataset.shuffle(seed=42).select(range(500))

    # Save as parquet (better than CSV for typed data)
    save_path = output_path / "sentiment140_500.parquet"
    dataset.to_parquet(str(save_path))

    metadata = {
        "num_samples": len(dataset),
        "columns": dataset.column_names,
        "source": "sentiment140",
        "subset": "500_sample",
        "seed": 42
    }

    meta_path = output_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(dataset)} samples to {save_path}")
    return metadata

if __name__ == "__main__":
    download_data()