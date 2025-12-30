"""
Clean and process HPD Housing Maintenance Code Complaints and Problems.

This script:
1. Loads raw complaints data
2. Parses dates
3. Categorizes by Major/Minor Category for health-related issues
4. Creates health-specific flags
5. Validates coordinates
6. Saves cleaned file

Input: data/raw/hpd_complaints/hpd_complaints_*.csv
Output: data/processed/complaints_clean.csv
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np

from housing_health.paths import PROCESSED_DIR, ensure_dirs_exist
from housing_health.io_utils import write_csv
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
    log_dataframe_info,
)

# Health-related major categories
HEALTH_MAJOR_CATEGORIES = [
    'HEAT/HOT WATER',
    'UNSANITARY CONDITION',
    'WATER LEAK',
    'PLUMBING',
    'PAINT/PLASTER',
    'VENTILATION',
    'GENERAL',
]

# Problem types of interest
PROBLEM_TYPES = {
    'Emergency': 3,
    'Immediate Emergency': 4,
    'Hazardous': 2,
    'Non Emergency': 1,
}

# NYC bounding box for coordinate validation
NYC_BOUNDS = {
    'lat_min': 40.4774,
    'lat_max': 41.0000,
    'lon_min': -74.2591,
    'lon_max': -73.7002,
}


def validate_coordinates(lat: float, lon: float) -> bool:
    """Check if coordinates are within NYC bounds."""
    if pd.isna(lat) or pd.isna(lon):
        return False
    return (NYC_BOUNDS['lat_min'] <= lat <= NYC_BOUNDS['lat_max'] and
            NYC_BOUNDS['lon_min'] <= lon <= NYC_BOUNDS['lon_max'])


def categorize_complaint(major_cat: str, minor_cat: str) -> dict:
    """Categorize complaint based on major/minor category."""
    result = {
        'is_heat': False,
        'is_mold': False,
        'is_pests': False,
        'is_lead': False,
        'is_water': False,
    }
    
    if pd.isna(major_cat):
        return result
    
    major_upper = str(major_cat).upper()
    minor_upper = str(minor_cat).upper() if pd.notna(minor_cat) else ''
    
    # Heat
    if 'HEAT' in major_upper or 'HOT WATER' in major_upper:
        result['is_heat'] = True
    
    # Mold
    if 'MOLD' in major_upper or 'MOLD' in minor_upper or 'MILDEW' in minor_upper:
        result['is_mold'] = True
    
    # Pests
    pest_keywords = ['ROACH', 'RODENT', 'MICE', 'MOUSE', 'RAT', 'PEST', 'VERMIN', 'BED BUG', 'BEDBUG']
    if any(kw in major_upper for kw in pest_keywords) or any(kw in minor_upper for kw in pest_keywords):
        result['is_pests'] = True
    
    # Lead/Paint
    if 'PAINT' in major_upper or 'LEAD' in major_upper or 'LEAD' in minor_upper:
        result['is_lead'] = True
    
    # Water/Plumbing
    if 'WATER' in major_upper or 'PLUMBING' in major_upper or 'LEAK' in major_upper:
        result['is_water'] = True
    
    return result


def clean_complaints():
    """Clean and process HPD complaints data."""
    
    # Setup logging
    log_file = get_timestamped_log_filename("clean_complaints")
    logger = setup_logger(__name__, log_file=log_file)
    
    log_step_start(logger, "Clean HPD Complaints")
    
    # Ensure directories exist
    ensure_dirs_exist()
    
    # Find the complaints file
    complaints_dir = Path("data/raw/hpd_complaints")
    complaints_files = list(complaints_dir.glob("hpd_complaints_*.csv"))
    
    if not complaints_files:
        logger.error("No complaints file found!")
        raise FileNotFoundError("No HPD complaints file found")
    
    input_file = max(complaints_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading complaints from {input_file}")
    
    # Load data
    logger.info("Reading CSV...")
    df = pd.read_csv(input_file, low_memory=False)
    log_dataframe_info(logger, df, "Raw complaints")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    logger.info(f"Columns: {list(df.columns)}")
    
    # Parse dates
    logger.info("Parsing dates...")
    date_columns = ['received_date', 'complaint_status_date', 'problem_status_date']
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            valid_dates = df[col].notna().sum()
            logger.info(f"  {col}: {valid_dates:,} valid dates")
    
    # Analyze Major Categories
    logger.info("\nMajor Category Distribution:")
    if 'major_category' in df.columns:
        cat_counts = df['major_category'].value_counts().head(15)
        for cat, count in cat_counts.items():
            logger.info(f"  {cat}: {count:,}")
    
    # Create health category flags
    logger.info("\nCreating health category flags...")
    
    # Initialize flags using vectorized operations (much faster!)
    major_col = 'major_category' if 'major_category' in df.columns else None
    minor_col = 'minor_category' if 'minor_category' in df.columns else None
    
    # Create uppercase versions for matching
    if major_col:
        major_upper = df[major_col].fillna('').str.upper()
    else:
        major_upper = pd.Series([''] * len(df))
    
    if minor_col:
        minor_upper = df[minor_col].fillna('').str.upper()
    else:
        minor_upper = pd.Series([''] * len(df))
    
    # Heat
    df['is_heat'] = major_upper.str.contains('HEAT|HOT WATER', regex=True, na=False)
    
    # Mold
    df['is_mold'] = (major_upper.str.contains('MOLD|MILDEW', regex=True, na=False) | 
                     minor_upper.str.contains('MOLD|MILDEW', regex=True, na=False))
    
    # Pests
    pest_pattern = 'ROACH|RODENT|MICE|MOUSE|RAT|PEST|VERMIN|BED BUG|BEDBUG'
    df['is_pests'] = (major_upper.str.contains(pest_pattern, regex=True, na=False) | 
                      minor_upper.str.contains(pest_pattern, regex=True, na=False))
    
    # Lead/Paint
    df['is_lead'] = (major_upper.str.contains('PAINT|LEAD', regex=True, na=False) | 
                     minor_upper.str.contains('LEAD', regex=True, na=False))
    
    # Water/Plumbing
    df['is_water'] = major_upper.str.contains('WATER|PLUMBING|LEAK', regex=True, na=False)
    
    # Log flag distribution
    for flag in ['is_heat', 'is_mold', 'is_pests', 'is_lead', 'is_water']:
        count = df[flag].sum()
        pct = count / len(df) * 100
        logger.info(f"  {flag}: {count:,} ({pct:.1f}%)")
    
    # Create combined health flag
    df['is_health_related'] = df['is_heat'] | df['is_mold'] | df['is_pests'] | df['is_lead'] | df['is_water']
    health_count = df['is_health_related'].sum()
    logger.info(f"Total health-related: {health_count:,} ({health_count/len(df)*100:.1f}%)")
    
    # Assign severity weight based on problem type
    logger.info("\nAssigning severity weights...")
    type_col = 'type' if 'type' in df.columns else None
    if type_col:
        df['severity_weight'] = df[type_col].map(PROBLEM_TYPES).fillna(1)
        logger.info(f"Type distribution:\n{df[type_col].value_counts()}")
    else:
        df['severity_weight'] = 1
    
    # Validate coordinates
    logger.info("\nValidating coordinates...")
    lat_col = 'latitude' if 'latitude' in df.columns else None
    lon_col = 'longitude' if 'longitude' in df.columns else None
    
    if lat_col and lon_col:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        df['valid_coords'] = df.apply(
            lambda row: validate_coordinates(row['latitude'], row['longitude']),
            axis=1
        )
        valid_count = df['valid_coords'].sum()
        logger.info(f"  Valid coordinates: {valid_count:,} ({valid_count/len(df)*100:.1f}%)")
    else:
        df['valid_coords'] = False
    
    # Standardize BBL format
    logger.info("Standardizing BBL format...")
    if 'bbl' in df.columns:
        df['bbl'] = df['bbl'].astype(str).str.strip()
        valid_bbl = df['bbl'].notna() & (df['bbl'] != '') & (df['bbl'] != 'nan')
        logger.info(f"  Valid BBL: {valid_bbl.sum():,} ({valid_bbl.sum()/len(df)*100:.1f}%)")
    
    # Save cleaned file
    output_file = PROCESSED_DIR / "complaints_clean.csv"
    logger.info(f"\nSaving cleaned data to {output_file}...")
    write_csv(df, output_file)
    
    log_step_complete(logger, "Clean HPD Complaints")
    
    # Summary
    logger.info("=" * 60)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Input records: {len(df):,}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / 1_000_000:.1f} MB")
    
    if 'received_date' in df.columns:
        logger.info(f"Date range: {df['received_date'].min()} to {df['received_date'].max()}")
    
    logger.info("\nHealth complaint breakdown:")
    logger.info(f"  Heat: {df['is_heat'].sum():,}")
    logger.info(f"  Mold: {df['is_mold'].sum():,}")
    logger.info(f"  Pests: {df['is_pests'].sum():,}")
    logger.info(f"  Lead/Paint: {df['is_lead'].sum():,}")
    logger.info(f"  Water/Plumbing: {df['is_water'].sum():,}")
    
    return df


if __name__ == "__main__":
    df = clean_complaints()
    print(f"\n✅ Cleaned {len(df):,} complaints!")
    print(f"   Health-related: {df['is_health_related'].sum():,}")

