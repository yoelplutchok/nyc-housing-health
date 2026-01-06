"""
Download DOB ECB Violations from NYC Open Data.

Dataset: DOB ECB Violations
Dataset ID: 6bgk-3dad
URL: https://data.cityofnewyork.us/Housing-Development/DOB-ECB-Violations/6bgk-3dad

This script downloads DOB violations from 2019 onwards, filtering for health-related
violations (hazardous severity, structural safety, etc.).
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

from housing_health.paths import DOB_VIOLATIONS_DIR, get_dated_filename, ensure_dirs_exist
from housing_health.io_utils import write_csv, update_manifest, calculate_file_hash
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
    log_dataframe_info,
)

# Configuration
DATASET_ID = "6bgk-3dad"
DOMAIN = "data.cityofnewyork.us"
START_DATE = "2019-01-01"
CHUNK_SIZE = 50000  # Socrata max per request
DATE_COLUMN = "issue_date"  # DOB uses issue_date with underscore

# Health-related keywords for filtering violations
HEALTH_KEYWORDS = [
    'asbestos', 'unsafe', 'structural', 'fire', 'egress', 'exit',
    'gas', 'carbon monoxide', 'hazard', 'safety', 'boiler', 'elevator',
    'collapse', 'dangerous', 'emergency', 'vacate'
]


def download_dob_violations():
    """Download DOB ECB violations data from NYC Open Data."""

    # Load environment variables
    load_dotenv()
    app_token = os.getenv("SOCRATA_APP_TOKEN")

    # Setup logging
    log_file = get_timestamped_log_filename("download_dob_violations")
    logger = setup_logger(__name__, log_file=log_file)

    log_step_start(logger, "Download DOB ECB Violations")

    # Ensure directories exist
    ensure_dirs_exist()

    # Initialize Socrata client
    logger.info(f"Connecting to NYC Open Data (token: {'set' if app_token else 'not set'})")
    client = Socrata(DOMAIN, app_token, timeout=300)

    # Get total count first
    logger.info(f"Querying dataset {DATASET_ID} for violations since {START_DATE}...")
    count_query = f"{DATE_COLUMN} >= '{START_DATE}'"

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
        pbar = tqdm(total=total_records, desc="Downloading DOB violations")
    else:
        pbar = tqdm(desc="Downloading DOB violations")

    while True:
        try:
            logger.info(f"Fetching chunk at offset {offset:,}...")
            chunk = client.get(
                DATASET_ID,
                where=count_query,
                limit=CHUNK_SIZE,
                offset=offset,
                order=f"{DATE_COLUMN} DESC"
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
    log_dataframe_info(logger, df, "DOB ECB Violations (raw)")

    # Flag health-related violations
    logger.info("Flagging health-related violations...")
    df = flag_health_related_violations(df, logger)

    # Save to file
    filename = get_dated_filename("dob_violations", "csv")
    filepath = DOB_VIOLATIONS_DIR / filename

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
        notes=f"DOB ECB Violations from {START_DATE} to present. File size: {file_size_mb:.1f}MB"
    )

    log_step_complete(logger, "Download DOB ECB Violations")

    # Summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(df):,}")
    if 'issue_date' in df.columns:
        logger.info(f"Date range: {df['issue_date'].min()} to {df['issue_date'].max()}")
    logger.info(f"File: {filepath}")
    logger.info(f"File size: {file_size_mb:.1f} MB")
    logger.info(f"Columns: {len(df.columns)}")

    # Show severity distribution
    if 'severity' in df.columns:
        logger.info("\nSeverity Distribution:")
        severity_counts = df['severity'].value_counts()
        for sev, count in severity_counts.items():
            pct = count / len(df) * 100
            logger.info(f"  {sev}: {count:,} ({pct:.1f}%)")

    # Show health-related count
    if 'is_health_related' in df.columns:
        health_count = df['is_health_related'].sum()
        logger.info(f"\nHealth-related violations: {health_count:,} ({health_count/len(df)*100:.1f}%)")

    return df


def flag_health_related_violations(df, logger):
    """Flag violations that are health/safety related."""

    # Create combined text field for keyword search
    # DOB uses underscored column names
    text_cols = []
    for col in ['violation_description', 'violation_type']:
        if col in df.columns:
            text_cols.append(col)

    # Also check section law descriptions (there can be up to 10)
    for i in range(1, 11):
        col = f'section_law_description{i}'
        if col in df.columns:
            text_cols.append(col)

    if text_cols:
        df['_combined_text'] = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
    else:
        df['_combined_text'] = ''

    # Flag hazardous severity
    df['is_hazardous'] = False
    if 'severity' in df.columns:
        df['is_hazardous'] = df['severity'].str.lower().str.contains('hazardous', na=False)

    # Flag by keywords
    keyword_pattern = '|'.join(HEALTH_KEYWORDS)
    df['has_health_keyword'] = df['_combined_text'].str.contains(keyword_pattern, case=False, na=False)

    # Combined health flag
    df['is_health_related'] = df['is_hazardous'] | df['has_health_keyword']

    # Clean up temp column
    df.drop(columns=['_combined_text'], inplace=True)

    logger.info(f"  Hazardous severity: {df['is_hazardous'].sum():,}")
    logger.info(f"  Health keyword matches: {df['has_health_keyword'].sum():,}")
    logger.info(f"  Total health-related: {df['is_health_related'].sum():,}")

    return df


if __name__ == "__main__":
    df = download_dob_violations()
    print(f"\n✅ Successfully downloaded {len(df):,} DOB ECB violations!")
    print(f"   Health-related: {df['is_health_related'].sum():,}")
