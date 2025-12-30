"""
I/O utilities for the NYC Housing Health project.

This module provides standardized functions for reading and writing data files,
with consistent error handling and logging.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# GeoPandas is optional - only needed for geo operations
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    gpd = None
    HAS_GEOPANDAS = False

from .paths import get_raw_manifest_path


def read_csv(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file with standard settings.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        DataFrame with the CSV contents
    """
    return pd.read_csv(filepath, **kwargs)


def write_csv(df: pd.DataFrame, filepath: Path, index: bool = False, **kwargs) -> None:
    """
    Write a DataFrame to CSV with standard settings.
    
    Args:
        df: DataFrame to write
        filepath: Output path
        index: Whether to include index (default False)
        **kwargs: Additional arguments passed to df.to_csv
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index, **kwargs)


def read_geojson(filepath: Path):
    """
    Read a GeoJSON file.
    
    Args:
        filepath: Path to GeoJSON file
    
    Returns:
        GeoDataFrame with the geographic data
    
    Raises:
        ImportError: If geopandas is not installed
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for read_geojson. Install with: pip install geopandas")
    return gpd.read_file(filepath)


def write_geojson(gdf, filepath: Path) -> None:
    """
    Write a GeoDataFrame to GeoJSON.
    
    Args:
        gdf: GeoDataFrame to write
        filepath: Output path
    
    Raises:
        ImportError: If geopandas is not installed
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for write_geojson. Install with: pip install geopandas")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(filepath, driver="GeoJSON")


def read_parquet(filepath: Path) -> pd.DataFrame:
    """
    Read a Parquet file.
    
    Args:
        filepath: Path to Parquet file
    
    Returns:
        DataFrame with the data
    """
    return pd.read_parquet(filepath)


def write_parquet(df: pd.DataFrame, filepath: Path, **kwargs) -> None:
    """
    Write a DataFrame to Parquet.
    
    Args:
        df: DataFrame to write
        filepath: Output path
        **kwargs: Additional arguments passed to df.to_parquet
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, **kwargs)


def calculate_file_hash(filepath: Path, algorithm: str = "md5") -> str:
    """
    Calculate hash of a file for verification.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm (default 'md5')
    
    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def update_manifest(
    filename: str,
    source_url: str,
    row_count: int,
    file_hash: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    """
    Update the raw data manifest with download information.
    
    Args:
        filename: Name of downloaded file
        source_url: URL data was downloaded from
        row_count: Number of records
        file_hash: MD5 hash of file (optional)
        notes: Additional notes (optional)
    """
    manifest_path = get_raw_manifest_path()
    
    # Load existing manifest or create new
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"downloads": []}
    
    # Add new entry
    entry = {
        "filename": filename,
        "source_url": source_url,
        "download_date": datetime.now().isoformat(),
        "row_count": row_count,
    }
    if file_hash:
        entry["file_hash"] = file_hash
    if notes:
        entry["notes"] = notes
    
    manifest["downloads"].append(entry)
    manifest["last_updated"] = datetime.now().isoformat()
    
    # Write back
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def read_manifest() -> dict[str, Any]:
    """
    Read the raw data manifest.
    
    Returns:
        Manifest dictionary
    """
    manifest_path = get_raw_manifest_path()
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {"downloads": []}

