"""
Clean and process HPD Housing Maintenance Code Violations.

This script:
1. Loads raw violations data
2. Parses dates to datetime format
3. Creates health-specific violation flags (lead, mold, pests, heat)
4. Assigns severity weights by class
5. Validates coordinates
6. Saves cleaned file

Input: data/raw/hpd_violations/hpd_violations_*.csv
Output: data/processed/violations_clean.csv
"""

import sys
from pathlib import Path
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from tqdm import tqdm

from housing_health.paths import HPD_VIOLATIONS_DIR, PROCESSED_DIR, ensure_dirs_exist
from housing_health.io_utils import write_csv
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
    log_dataframe_info,
)

# Keywords for health-related violations (expanded for better coverage)
# Note: These patterns are matched case-insensitively against the NOV description

LEAD_KEYWORDS = [
    # Direct lead references
    'lead', 'lead-based', 'lead paint', 'leadpaint', 'lead based',
    # Testing and compliance
    'xrf', 'epa-hud', 'abatement', 'lead-safe',
    # Specific NYC lead laws
    'local law 1', 'll1', 'lead hazard', 'lead dust',
    # Child safety related
    'child-occupied', 'window sill', 'windowsill', 'chipping paint',
    'peeling paint', 'flaking paint', 'deteriorated paint',
]

MOLD_KEYWORDS = [
    # Direct mold/mildew
    'mold', 'mould', 'mildew', 'fungus', 'fungi', 'musty',
    # Water damage (often leads to mold)
    'moisture', 'water damage', 'water stain', 'dampness', 'wet wall',
    'leak', 'leaking', 'water infiltration', 'condensation',
    # Visible signs
    'black spot', 'discoloration', 'growth',
]

PEST_KEYWORDS = [
    # Insects
    'roach', 'cockroach', 'roaches', 'bedbug', 'bed bug', 'bedbugs',
    # Rodents
    'rodent', 'mice', 'mouse', 'rat', 'rats',
    # General pest terms
    'pest', 'vermin', 'infestation', 'infested', 'exterminate',
    # Other pests
    'ant', 'ants', 'flea', 'fleas', 'fly', 'flies',
    # Evidence
    'droppings', 'gnaw marks', 'burrow',
]

HEAT_KEYWORDS = [
    # Heat/heating
    'heat', 'no heat', 'lack of heat', 'heating', 'heated',
    'heating system', 'heat supply', 'heat not provided',
    # Hot water
    'hot water', 'no hot water', 'lack of hot water', 'hotwater',
    # Equipment
    'boiler', 'furnace', 'radiator', 'thermostat', 'steam',
    'heating equipment', 'burner', 'pilot light',
    # Temperature
    'temperature', 'cold', 'inadequate heat',
]

# Severity weights by class
CLASS_WEIGHTS = {
    'A': 1,  # Non-hazardous
    'B': 2,  # Hazardous
    'C': 3,  # Immediately hazardous
}

# NYC bounding box for coordinate validation
NYC_BOUNDS = {
    'lat_min': 40.4774,
    'lat_max': 41.0000,
    'lon_min': -74.2591,
    'lon_max': -73.7002,
}


def contains_keywords(text: str, keywords: list) -> bool:
    """Check if text contains any of the keywords (case-insensitive)."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def validate_coordinates(lat: float, lon: float) -> bool:
    """Check if coordinates are within NYC bounds."""
    if pd.isna(lat) or pd.isna(lon):
        return False
    return (NYC_BOUNDS['lat_min'] <= lat <= NYC_BOUNDS['lat_max'] and
            NYC_BOUNDS['lon_min'] <= lon <= NYC_BOUNDS['lon_max'])


def clean_violations():
    """Clean and process HPD violations data."""
    
    # Setup logging
    log_file = get_timestamped_log_filename("clean_violations")
    logger = setup_logger(__name__, log_file=log_file)
    
    log_step_start(logger, "Clean HPD Violations")
    
    # Ensure directories exist
    ensure_dirs_exist()
    
    # Find the most recent violations file
    violations_files = list(HPD_VIOLATIONS_DIR.glob("hpd_violations_*.csv"))
    if not violations_files:
        # Check in raw directory directly
        violations_files = list(Path("data/raw/hpd_violations").glob("hpd_violations_*.csv"))
    
    if not violations_files:
        logger.error("No violations file found!")
        raise FileNotFoundError("No HPD violations file found")
    
    input_file = max(violations_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading violations from {input_file}")
    
    # Load data
    logger.info("Reading CSV (this may take a minute for large files)...")
    df = pd.read_csv(input_file, low_memory=False)
    log_dataframe_info(logger, df, "Raw violations")
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    logger.info(f"Columns: {list(df.columns)}")
    
    # Parse dates
    logger.info("Parsing dates...")
    date_columns = ['inspectiondate', 'approveddate', 'currentstatusdate', 
                    'originalcertifybydate', 'originalcorrectbydate', 
                    'novissueddate', 'certifieddate']
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            valid_dates = df[col].notna().sum()
            logger.info(f"  {col}: {valid_dates:,} valid dates")
    
    # Standardize violation class
    logger.info("Standardizing violation class...")
    if 'class' in df.columns:
        df['class'] = df['class'].astype(str).str.upper().str.strip()
        class_dist = df['class'].value_counts()
        logger.info(f"Class distribution:\n{class_dist}")
    
    # Create health-specific violation flags
    logger.info("Creating health violation flags...")
    description_col = 'novdescription' if 'novdescription' in df.columns else None
    
    if description_col:
        tqdm.pandas(desc="Flagging lead violations")
        df['is_lead'] = df[description_col].progress_apply(
            lambda x: contains_keywords(x, LEAD_KEYWORDS)
        )
        
        tqdm.pandas(desc="Flagging mold violations")
        df['is_mold'] = df[description_col].progress_apply(
            lambda x: contains_keywords(x, MOLD_KEYWORDS)
        )
        
        tqdm.pandas(desc="Flagging pest violations")
        df['is_pests'] = df[description_col].progress_apply(
            lambda x: contains_keywords(x, PEST_KEYWORDS)
        )
        
        tqdm.pandas(desc="Flagging heat violations")
        df['is_heat'] = df[description_col].progress_apply(
            lambda x: contains_keywords(x, HEAT_KEYWORDS)
        )
        
        # Log flag distribution
        for flag in ['is_lead', 'is_mold', 'is_pests', 'is_heat']:
            count = df[flag].sum()
            pct = count / len(df) * 100
            logger.info(f"  {flag}: {count:,} ({pct:.1f}%)")
    else:
        logger.warning("No description column found for keyword flagging")
        df['is_lead'] = False
        df['is_mold'] = False
        df['is_pests'] = False
        df['is_heat'] = False
    
    # Create severity weight
    logger.info("Assigning severity weights...")
    df['severity_weight'] = df['class'].map(CLASS_WEIGHTS).fillna(1)
    
    # Validate coordinates
    logger.info("Validating coordinates...")
    if 'latitude' in df.columns and 'longitude' in df.columns:
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
        logger.warning("No coordinate columns found")
    
    # Standardize BBL format
    logger.info("Standardizing BBL format...")
    if 'bbl' in df.columns:
        df['bbl'] = df['bbl'].astype(str).str.strip()
        valid_bbl = df['bbl'].notna() & (df['bbl'] != '') & (df['bbl'] != 'nan')
        logger.info(f"  Valid BBL: {valid_bbl.sum():,} ({valid_bbl.sum()/len(df)*100:.1f}%)")
    
    # Create any_health_violation flag
    df['is_health_related'] = df['is_lead'] | df['is_mold'] | df['is_pests'] | df['is_heat']
    health_count = df['is_health_related'].sum()
    logger.info(f"Total health-related violations: {health_count:,} ({health_count/len(df)*100:.1f}%)")
    
    # Save cleaned file
    output_file = PROCESSED_DIR / "violations_clean.csv"
    logger.info(f"Saving cleaned data to {output_file}...")
    write_csv(df, output_file)
    
    log_step_complete(logger, "Clean HPD Violations")
    
    # Summary statistics
    logger.info("=" * 60)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Input records: {len(df):,}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / 1_000_000:.1f} MB")
    
    if 'inspectiondate' in df.columns:
        logger.info(f"Date range: {df['inspectiondate'].min()} to {df['inspectiondate'].max()}")
    
    logger.info("\nHealth violation breakdown:")
    logger.info(f"  Lead: {df['is_lead'].sum():,}")
    logger.info(f"  Mold: {df['is_mold'].sum():,}")
    logger.info(f"  Pests: {df['is_pests'].sum():,}")
    logger.info(f"  Heat: {df['is_heat'].sum():,}")
    
    return df


if __name__ == "__main__":
    df = clean_violations()
    print(f"\n✅ Cleaned {len(df):,} violations!")
    print(f"   Health-related: {df['is_health_related'].sum():,}")

