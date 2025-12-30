"""
Join health data (asthma, lead poisoning) to neighborhood aggregated data.

This script:
1. Loads NTA-aggregated violations/complaints data
2. Joins childhood asthma rates (by NTA name matching)
3. Joins lead poisoning rates (by borough, since lead data is at UHF42 level)
4. Saves enhanced neighborhood dataset

Input: 
  - data/processed/nta_aggregated.csv
  - data/health/childhood_asthma_*.csv
  - data/health/lead_poisoning_*.csv

Output: data/processed/nta_with_health.csv
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from difflib import SequenceMatcher

from housing_health.paths import PROCESSED_DIR, HEALTH_DIR, ensure_dirs_exist
from housing_health.io_utils import write_csv
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
    log_dataframe_info,
)


def fuzzy_match_score(s1: str, s2: str) -> float:
    """Calculate fuzzy match score between two strings."""
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def clean_nta_name(name: str) -> str:
    """Clean and normalize NTA name for matching."""
    if pd.isna(name):
        return ''
    # Remove parenthetical suffixes like (North), (East), etc.
    name = str(name)
    # Remove content in parentheses for matching
    import re
    name = re.sub(r'\s*\([^)]*\)', '', name)
    return name.strip().lower()


def match_nta_names(violations_ntas: list, health_ntas: list, threshold: float = 0.7):
    """
    Create a mapping between violations NTA names and health data NTA names.
    Returns a dict mapping violations NTA -> health NTA.
    """
    mapping = {}
    
    for v_nta in violations_ntas:
        v_clean = clean_nta_name(v_nta)
        if not v_clean:
            continue
            
        best_match = None
        best_score = 0
        
        for h_nta in health_ntas:
            h_clean = clean_nta_name(h_nta)
            if not h_clean:
                continue
            
            # Try exact match first
            if v_clean == h_clean:
                best_match = h_nta
                best_score = 1.0
                break
            
            # Fuzzy match
            score = fuzzy_match_score(v_clean, h_clean)
            if score > best_score:
                best_score = score
                best_match = h_nta
        
        if best_score >= threshold:
            mapping[v_nta] = best_match
    
    return mapping


def load_asthma_data(logger):
    """Load and process childhood asthma data."""
    asthma_files = list(HEALTH_DIR.glob("childhood_asthma_*.csv"))
    if not asthma_files:
        logger.warning("No asthma data file found")
        return None
    
    asthma_file = max(asthma_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading asthma data from {asthma_file}")
    
    df = pd.read_csv(asthma_file)
    
    # Get most recent time period
    if 'TimePeriod' in df.columns:
        most_recent = df['TimePeriod'].max()
        df = df[df['TimePeriod'] == most_recent]
        logger.info(f"  Using time period: {most_recent}")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Extract NTA-level data
    if 'GeoType' in df.columns:
        df = df[df['GeoType'].str.contains('NTA', case=False, na=False)]
    
    # Rename columns
    asthma_df = df[['Geography', 'Average annual rate per 10,000']].copy()
    asthma_df.columns = ['nta_health', 'asthma_rate_per_10k']
    asthma_df['asthma_rate_per_10k'] = pd.to_numeric(asthma_df['asthma_rate_per_10k'], errors='coerce')
    
    logger.info(f"  Loaded {len(asthma_df)} NTA asthma records")
    logger.info(f"  Rate range: {asthma_df['asthma_rate_per_10k'].min():.1f} - {asthma_df['asthma_rate_per_10k'].max():.1f} per 10k")
    
    return asthma_df


def load_lead_data(logger):
    """Load and process lead poisoning data."""
    lead_files = list(HEALTH_DIR.glob("lead_poisoning_*.csv"))
    if not lead_files:
        logger.warning("No lead poisoning data file found")
        return None
    
    lead_file = max(lead_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading lead data from {lead_file}")
    
    df = pd.read_csv(lead_file)
    
    # Get most recent time period
    if 'time_period' in df.columns:
        most_recent = df['time_period'].max()
        df = df[df['time_period'] == most_recent]
        logger.info(f"  Using time period: {most_recent}")
    
    # Get borough-level data (since UHF42 doesn't map easily to NTA2020)
    if 'geo_type' in df.columns:
        borough_df = df[df['geo_type'] == 'Borough'].copy()
    else:
        borough_df = df
    
    # Find the rate column
    rate_col = None
    for col in df.columns:
        if 'Rate' in col and 'BLL>=5' in col and 'per 1,000' in col:
            rate_col = col
            break
    
    if rate_col is None:
        # Try alternate column names
        for col in df.columns:
            if 'Rate' in col and 'BLL' in col:
                rate_col = col
                break
    
    if rate_col:
        lead_df = borough_df[['geo_area_name', rate_col]].copy()
        lead_df.columns = ['borough', 'lead_rate_per_1k']
        lead_df['lead_rate_per_1k'] = pd.to_numeric(lead_df['lead_rate_per_1k'], errors='coerce')
        
        # Clean borough names
        lead_df['borough'] = lead_df['borough'].str.strip()
        
        logger.info(f"  Loaded {len(lead_df)} borough lead records")
        logger.info(f"  Rate range: {lead_df['lead_rate_per_1k'].min():.1f} - {lead_df['lead_rate_per_1k'].max():.1f} per 1k")
        
        return lead_df
    else:
        logger.warning("Could not find rate column in lead data")
        return None


def join_health_data():
    """Main function to join health data to neighborhood data."""
    
    # Setup logging
    log_file = get_timestamped_log_filename("join_health")
    logger = setup_logger(__name__, log_file=log_file)
    
    log_step_start(logger, "Join Health Data")
    
    # Ensure directories exist
    ensure_dirs_exist()
    
    # Load NTA aggregated data
    nta_file = PROCESSED_DIR / "nta_aggregated.csv"
    logger.info(f"Loading NTA data from {nta_file}")
    nta_df = pd.read_csv(nta_file)
    log_dataframe_info(logger, nta_df, "NTA aggregated")
    
    initial_count = len(nta_df)
    
    # Load and join asthma data
    asthma_df = load_asthma_data(logger)
    if asthma_df is not None:
        # Create mapping between NTA names
        violations_ntas = nta_df['nta'].dropna().unique().tolist()
        health_ntas = asthma_df['nta_health'].dropna().unique().tolist()
        
        logger.info("Creating NTA name mapping...")
        nta_mapping = match_nta_names(violations_ntas, health_ntas, threshold=0.6)
        logger.info(f"  Matched {len(nta_mapping)} / {len(violations_ntas)} NTAs")
        
        # Apply mapping
        nta_df['nta_health_match'] = nta_df['nta'].map(nta_mapping)
        
        # Join asthma data
        nta_df = nta_df.merge(
            asthma_df,
            left_on='nta_health_match',
            right_on='nta_health',
            how='left'
        )
        
        # Clean up
        nta_df.drop(['nta_health_match', 'nta_health'], axis=1, errors='ignore', inplace=True)
        
        matched = nta_df['asthma_rate_per_10k'].notna().sum()
        logger.info(f"  Joined asthma data for {matched} / {len(nta_df)} NTAs")
    
    # Load and join lead data (by borough)
    lead_df = load_lead_data(logger)
    if lead_df is not None and 'borough' in nta_df.columns:
        nta_df = nta_df.merge(lead_df, on='borough', how='left')
        matched = nta_df['lead_rate_per_1k'].notna().sum()
        logger.info(f"  Joined lead data for {matched} / {len(nta_df)} NTAs (by borough)")
    
    # Calculate health risk score (combined health indicator)
    logger.info("Calculating health risk composite score...")
    
    # Normalize scores to 0-100 scale
    if 'asthma_rate_per_10k' in nta_df.columns:
        nta_df['asthma_pctl'] = nta_df['asthma_rate_per_10k'].rank(pct=True) * 100
    if 'lead_rate_per_1k' in nta_df.columns:
        nta_df['lead_pctl'] = nta_df['lead_rate_per_1k'].rank(pct=True) * 100
    
    # Save
    output_file = PROCESSED_DIR / "nta_with_health.csv"
    logger.info(f"Saving to {output_file}...")
    write_csv(nta_df, output_file)
    
    log_step_complete(logger, "Join Health Data")
    
    # Summary
    logger.info("=" * 60)
    logger.info("HEALTH DATA JOIN SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total NTAs: {len(nta_df)}")
    
    if 'asthma_rate_per_10k' in nta_df.columns:
        asthma_matched = nta_df['asthma_rate_per_10k'].notna().sum()
        logger.info(f"Asthma data matched: {asthma_matched} NTAs")
        logger.info(f"  Mean rate: {nta_df['asthma_rate_per_10k'].mean():.1f} per 10k")
        
        # Top 5 by asthma
        logger.info("\nTop 5 NTAs by Asthma Rate:")
        top5 = nta_df.nlargest(5, 'asthma_rate_per_10k')[['nta', 'borough', 'asthma_rate_per_10k']]
        for _, row in top5.iterrows():
            logger.info(f"  {row['nta']} ({row['borough']}): {row['asthma_rate_per_10k']:.1f}")
    
    if 'lead_rate_per_1k' in nta_df.columns:
        logger.info("\nLead Poisoning Rates by Borough:")
        for _, row in lead_df.iterrows():
            logger.info(f"  {row['borough']}: {row['lead_rate_per_1k']:.1f} per 1k")
    
    return nta_df


if __name__ == "__main__":
    df = join_health_data()
    print(f"\n✅ Health data joined for {len(df)} neighborhoods!")

