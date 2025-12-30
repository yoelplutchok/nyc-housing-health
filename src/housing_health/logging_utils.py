"""
Logging utilities for the NYC Housing Health project.

This module provides standardized logging configuration for all scripts,
ensuring consistent log formatting and file output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .paths import LOGS_DIR


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name (typically __name__ of calling module)
        log_file: Name of log file (will be placed in logs/ directory)
                 If None, only console output is configured
        level: Logging level (default INFO)
        console_output: Whether to also log to console (default True)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_timestamped_log_filename(base_name: str) -> str:
    """
    Generate a log filename with timestamp.
    
    Args:
        base_name: Base name for the log file (e.g., 'download_violations')
    
    Returns:
        Log filename with timestamp (e.g., 'download_violations_2025-01-15_143022.log')
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{base_name}_{timestamp}.log"


def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame") -> None:
    """
    Log summary information about a DataFrame.
    
    Args:
        logger: Logger instance
        df: DataFrame (pandas or geopandas)
        name: Name to use in log messages
    """
    logger.info(f"{name}: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"{name} columns: {list(df.columns)}")
    
    # Log memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1_000_000
    logger.info(f"{name} memory usage: {memory_mb:.2f} MB")


def log_step_start(logger: logging.Logger, step_name: str) -> None:
    """Log the start of a processing step."""
    logger.info("=" * 60)
    logger.info(f"STARTING: {step_name}")
    logger.info("=" * 60)


def log_step_complete(logger: logging.Logger, step_name: str) -> None:
    """Log the completion of a processing step."""
    logger.info("-" * 60)
    logger.info(f"COMPLETED: {step_name}")
    logger.info("-" * 60)

