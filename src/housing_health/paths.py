"""
Path configuration for the NYC Housing Health project.

This module provides standardized paths to all project directories and files,
ensuring consistent file access across all scripts.
"""

from pathlib import Path

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GEO_DIR = DATA_DIR / "geo"
HEALTH_DIR = DATA_DIR / "health"

# Raw data subdirectories
HPD_VIOLATIONS_DIR = RAW_DIR / "hpd_violations"
HPD_COMPLAINTS_DIR = RAW_DIR / "hpd_complaints"
REQUESTS_311_DIR = RAW_DIR / "311_requests"
DOB_VIOLATIONS_DIR = RAW_DIR / "dob_violations"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
INTERACTIVE_DIR = OUTPUTS_DIR / "interactive"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Config and logs
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Documentation
DOCS_DIR = PROJECT_ROOT / "docs"

# Scripts
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Web application
WEB_DIR = PROJECT_ROOT / "web"


def ensure_dirs_exist() -> None:
    """Create all project directories if they don't exist."""
    dirs = [
        RAW_DIR,
        HPD_VIOLATIONS_DIR,
        HPD_COMPLAINTS_DIR,
        REQUESTS_311_DIR,
        DOB_VIOLATIONS_DIR,
        PROCESSED_DIR,
        GEO_DIR,
        HEALTH_DIR,
        FIGURES_DIR,
        INTERACTIVE_DIR,
        TABLES_DIR,
        CONFIGS_DIR,
        LOGS_DIR,
        DOCS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_raw_manifest_path() -> Path:
    """Return path to the raw data manifest file."""
    return RAW_DIR / "manifest.json"


def get_dated_filename(base_name: str, extension: str = "csv") -> str:
    """
    Generate a filename with today's date stamp.
    
    Args:
        base_name: Base name for the file (e.g., 'violations')
        extension: File extension without dot (e.g., 'csv', 'parquet')
    
    Returns:
        Filename with date stamp (e.g., 'violations_2025-01-15.csv')
    """
    from datetime import date
    today = date.today().isoformat()
    return f"{base_name}_{today}.{extension}"


if __name__ == "__main__":
    # Print all paths for verification
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"RAW_DIR: {RAW_DIR}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")
    print(f"GEO_DIR: {GEO_DIR}")
    print(f"HEALTH_DIR: {HEALTH_DIR}")
    print(f"OUTPUTS_DIR: {OUTPUTS_DIR}")

