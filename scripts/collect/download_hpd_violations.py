"""
Download HPD Housing Maintenance Code Violations from NYC Open Data.

Dataset: Housing Maintenance Code Violations
Dataset ID: wvxf-dwi5
URL: https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Violations/wvxf-dwi5

This script downloads violations from 2019 onwards in chunks using the Socrata API.
"""

import os
import sys
from datetime import date
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from dotenv import load_dotenv
from sodapy import Socrata
import pandas as pd
from tqdm import tqdm

from housing_health.paths import HPD_VIOLATIONS_DIR, get_dated_filename, ensure_dirs_exist
from housing_health.io_utils import write_csv, update_manifest, calculate_file_hash
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
    log_dataframe_info,
)

# Configuration
DATASET_ID = "wvxf-dwi5"
DOMAIN = "data.cityofnewyork.us"
START_DATE = "2019-01-01"
CHUNK_SIZE = 50000  # Socrata max per request


def download_violations():
    """Download HPD violations data from NYC Open Data."""
    
    # Load environment variables
    load_dotenv()
    app_token = os.getenv("SOCRATA_APP_TOKEN")
    
    # Setup logging
    log_file = get_timestamped_log_filename("download_hpd_violations")
    logger = setup_logger(__name__, log_file=log_file)
    
    log_step_start(logger, "Download HPD Violations")
    
    # Ensure directories exist
    ensure_dirs_exist()
    
    # Initialize Socrata client
    logger.info(f"Connecting to NYC Open Data (token: {'set' if app_token else 'not set'})")
    client = Socrata(DOMAIN, app_token, timeout=300)
    
    # Get total count first
    logger.info(f"Querying dataset {DATASET_ID} for violations since {START_DATE}...")
    count_query = f"inspectiondate >= '{START_DATE}'"
    
    try:
        count_result = client.get(DATASET_ID, select="COUNT(*)", where=count_query)
        total_records = int(count_result[0]["COUNT"])
        logger.info(f"Total records to download: {total_records:,}")
    except Exception as e:
        logger.warning(f"Could not get count, will download until exhausted: {e}")
        total_records = None
    
    # Download in chunks
    all_data = []
    offset = 0
    
    if total_records:
        pbar = tqdm(total=total_records, desc="Downloading violations")
    else:
        pbar = tqdm(desc="Downloading violations")
    
    while True:
        try:
            logger.info(f"Fetching chunk at offset {offset:,}...")
            chunk = client.get(
                DATASET_ID,
                where=count_query,
                limit=CHUNK_SIZE,
                offset=offset,
                order="inspectiondate DESC"
            )
            
            if not chunk:
                logger.info("No more records to fetch.")
                break
            
            all_data.extend(chunk)
            pbar.update(len(chunk))
            offset += len(chunk)
            
            logger.info(f"  Retrieved {len(chunk):,} records (total so far: {len(all_data):,})")
            
            if len(chunk) < CHUNK_SIZE:
                logger.info("Received partial chunk, download complete.")
                break
                
        except Exception as e:
            logger.error(f"Error fetching chunk at offset {offset}: {e}")
            raise
    
    pbar.close()
    client.close()
    
    # Convert to DataFrame
    logger.info("Converting to DataFrame...")
    df = pd.DataFrame.from_records(all_data)
    log_dataframe_info(logger, df, "HPD Violations")
    
    # Save to file
    filename = get_dated_filename("hpd_violations", "csv")
    filepath = HPD_VIOLATIONS_DIR / filename
    
    logger.info(f"Saving to {filepath}...")
    write_csv(df, filepath)
    
    # Calculate file hash and update manifest
    file_hash = calculate_file_hash(filepath)
    file_size_mb = filepath.stat().st_size / 1_000_000
    
    update_manifest(
        filename=filename,
        source_url=f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json",
        row_count=len(df),
        file_hash=file_hash,
        notes=f"Violations from {START_DATE} to present. File size: {file_size_mb:.1f}MB"
    )
    
    log_step_complete(logger, "Download HPD Violations")
    
    # Summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(df):,}")
    logger.info(f"Date range: {df['inspectiondate'].min()} to {df['inspectiondate'].max()}")
    logger.info(f"File: {filepath}")
    logger.info(f"File size: {file_size_mb:.1f} MB")
    logger.info(f"Columns: {len(df.columns)}")
    
    # Show violation class distribution
    if 'class' in df.columns:
        logger.info("\nViolation Class Distribution:")
        class_counts = df['class'].value_counts()
        for cls, count in class_counts.items():
            pct = count / len(df) * 100
            logger.info(f"  Class {cls}: {count:,} ({pct:.1f}%)")
    
    return df


if __name__ == "__main__":
    df = download_violations()
    print(f"\n✅ Successfully downloaded {len(df):,} HPD violations!")

