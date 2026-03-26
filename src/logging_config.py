"""
Centralized logging configuration.
Import this at the top of any script for consistent log formatting.

Usage:
    from src.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Starting training...")
"""
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | None = None, level=logging.INFO) -> logging.Logger:
    """
    Create a logger with consistent formatting across all modules.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional path to also write logs to a file
        level: Logging level (default INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Format: 2025-01-15 10:30:45 | INFO | src.training.train | Starting epoch 1
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger