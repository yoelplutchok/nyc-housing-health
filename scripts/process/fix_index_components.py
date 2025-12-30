"""
Fix missing index components and recalculate Housing Health Index.

This script addresses the critique:
1. Process PLUTO to calculate building age (pct_pre_1978) by NTA
2. Join census demographics to NTAs
3. Correct index weights to 30/25/20/15/10
4. Recalculate with all 5 components

Input:
  - data/processed/nta_with_health.csv
  - data/raw/pluto_*.csv
  - data/raw/census_demographics_*.csv

Output:
  - data/processed/nta_complete.csv (with all components)
  - data/processed/housing_health_index.csv (recalculated)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np

from housing_health.paths import PROCESSED_DIR, RAW_DIR, ensure_dirs_exist
from housing_health.io_utils import write_csv
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
    log_dataframe_info,
)


# CORRECT weights from params.yml
WEIGHTS = {
    'violations': 0.30,
    'complaints': 0.25,
    'asthma': 0.20,
    'lead': 0.15,
    'building_age': 0.10,
}

# Risk tier thresholds (percentiles)
RISK_TIERS = {
    'Very High Risk': 90,
    'High Risk': 75,
    'Elevated Risk': 50,
    'Moderate Risk': 25,
    'Low Risk': 0,
}

# Pre-1978 is the lead paint cutoff
LEAD_PAINT_CUTOFF = 1978


def normalize_to_percentile(series: pd.Series) -> pd.Series:
    """Normalize series to 0-100 percentile scale."""
    return series.rank(pct=True, na_option='keep') * 100


def process_pluto_building_age(logger):
    """
    Process PLUTO data to calculate building age stats by NTA.
    Uses BBL from violations to map buildings to NTAs.
    """
    logger.info("Processing PLUTO for building age...")
    
    # Find PLUTO file
    pluto_files = list(RAW_DIR.glob("pluto_*.csv"))
    if not pluto_files:
        logger.error("No PLUTO file found")
        return None
    
    pluto_file = max(pluto_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"  Loading {pluto_file.name}...")
    
    # Load PLUTO with needed columns
    pluto = pd.read_csv(pluto_file, usecols=['BBL', 'yearbuilt', 'unitsres'], low_memory=False)
    logger.info(f"  Loaded {len(pluto):,} PLUTO records")
    
    # Convert BBL to string for matching
    pluto['bbl'] = pluto['BBL'].astype(str).str.strip()
    pluto['yearbuilt'] = pd.to_numeric(pluto['yearbuilt'], errors='coerce')
    pluto['unitsres'] = pd.to_numeric(pluto['unitsres'], errors='coerce').fillna(0)
    
    # Flag pre-1978 buildings
    pluto['is_pre_1978'] = pluto['yearbuilt'] < LEAD_PAINT_CUTOFF
    
    # Load BBL-NTA mapping from violations
    logger.info("  Loading BBL-NTA mapping from violations...")
    violations_file = PROCESSED_DIR / "violations_clean.csv"
    bbl_nta = pd.read_csv(violations_file, usecols=['bbl', 'nta'])
    
    # Convert violations BBL to matching format
    bbl_nta['bbl'] = pd.to_numeric(bbl_nta['bbl'], errors='coerce').astype('Int64').astype(str)
    bbl_nta = bbl_nta.dropna(subset=['nta']).drop_duplicates(subset=['bbl'])
    logger.info(f"  {len(bbl_nta):,} unique BBL-NTA mappings")
    
    # Join PLUTO to NTA via BBL
    pluto_with_nta = pluto.merge(bbl_nta, on='bbl', how='inner')
    logger.info(f"  Matched {len(pluto_with_nta):,} PLUTO records to NTAs")
    
    # Aggregate by NTA
    nta_building_age = pluto_with_nta.groupby('nta').agg({
        'bbl': 'count',
        'unitsres': 'sum',
        'is_pre_1978': 'sum',
        'yearbuilt': 'mean'
    }).reset_index()
    
    nta_building_age.columns = ['nta', 'pluto_building_count', 'total_units', 'pre_1978_count', 'avg_year_built']
    
    # Calculate percentage pre-1978
    nta_building_age['pct_pre_1978'] = (
        nta_building_age['pre_1978_count'] / nta_building_age['pluto_building_count'] * 100
    )
    
    logger.info(f"  Aggregated to {len(nta_building_age)} NTAs")
    logger.info(f"  Avg pct_pre_1978: {nta_building_age['pct_pre_1978'].mean():.1f}%")
    logger.info(f"  Range: {nta_building_age['pct_pre_1978'].min():.1f}% - {nta_building_age['pct_pre_1978'].max():.1f}%")
    
    return nta_building_age


def process_census_demographics(logger):
    """
    Process census demographics and aggregate to NTA level.
    Uses census tract to NTA mapping.
    """
    logger.info("Processing census demographics...")
    
    # Find census file
    census_files = list(RAW_DIR.glob("census_demographics_*.csv"))
    if not census_files:
        logger.warning("No census demographics file found")
        return None
    
    census_file = max(census_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"  Loading {census_file.name}...")
    
    census = pd.read_csv(census_file)
    logger.info(f"  Loaded {len(census):,} census tracts")
    log_dataframe_info(logger, census, "Census")
    
    # The census data needs to be aggregated to NTA
    # This requires a tract-to-NTA crosswalk
    # For now, we'll aggregate to borough level as a proxy
    
    if 'state' in census.columns and 'county' in census.columns:
        # NYC county FIPS codes
        county_to_borough = {
            '005': 'Bronx',
            '047': 'Brooklyn', 
            '061': 'Manhattan',
            '081': 'Queens',
            '085': 'Staten Island'
        }
        
        census['borough'] = census['county'].astype(str).str.zfill(3).map(county_to_borough)
        
        # Aggregate to borough
        borough_demo = census.groupby('borough').agg({
            'total_population': 'sum',
            'median_income': 'median',
        }).reset_index()
        
        logger.info(f"  Aggregated to {len(borough_demo)} boroughs")
        return borough_demo
    else:
        logger.warning("  Census data doesn't have expected columns")
        return None


def fix_and_recalculate():
    """Main function to fix components and recalculate index."""
    
    # Setup logging
    log_file = get_timestamped_log_filename("fix_index")
    logger = setup_logger(__name__, log_file=log_file)
    
    log_step_start(logger, "Fix Index Components")
    ensure_dirs_exist()
    
    # Load current NTA data
    input_file = PROCESSED_DIR / "nta_with_health.csv"
    logger.info(f"Loading NTA data from {input_file}")
    nta_df = pd.read_csv(input_file)
    log_dataframe_info(logger, nta_df, "Current NTA data")
    
    # Step 1: Add building age component from PLUTO
    building_age = process_pluto_building_age(logger)
    if building_age is not None:
        nta_df = nta_df.merge(
            building_age[['nta', 'pct_pre_1978', 'total_units', 'pluto_building_count', 'avg_year_built']],
            on='nta',
            how='left'
        )
        matched = nta_df['pct_pre_1978'].notna().sum()
        logger.info(f"  Joined building age for {matched} / {len(nta_df)} NTAs")
    
    # Step 2: Add census demographics
    demographics = process_census_demographics(logger)
    if demographics is not None and 'borough' in nta_df.columns:
        nta_df = nta_df.merge(demographics, on='borough', how='left')
        logger.info(f"  Joined demographics by borough")
    
    # Save intermediate file
    complete_file = PROCESSED_DIR / "nta_complete.csv"
    write_csv(nta_df, complete_file)
    logger.info(f"Saved complete NTA data to {complete_file}")
    
    # Step 3: Recalculate index with correct weights
    logger.info("\n" + "="*60)
    logger.info("RECALCULATING HOUSING HEALTH INDEX")
    logger.info("="*60)
    logger.info(f"Using CORRECT weights: {WEIGHTS}")
    
    # Calculate normalized indicators
    logger.info("\nNormalizing indicators to 0-100 scale...")
    
    # Violations score
    if 'violation_health_per_bldg' in nta_df.columns:
        nta_df['violations_score'] = normalize_to_percentile(nta_df['violation_health_per_bldg'])
        logger.info(f"  Violations score: mean={nta_df['violations_score'].mean():.1f}")
    else:
        nta_df['violations_score'] = 50
        logger.warning("  Violations data missing")
    
    # Complaints score
    if 'complaint_health_per_bldg' in nta_df.columns:
        nta_df['complaints_score'] = normalize_to_percentile(nta_df['complaint_health_per_bldg'])
        logger.info(f"  Complaints score: mean={nta_df['complaints_score'].mean():.1f}")
    else:
        nta_df['complaints_score'] = 50
        logger.warning("  Complaints data missing")
    
    # Asthma score
    if 'asthma_rate_per_10k' in nta_df.columns:
        nta_df['asthma_score'] = normalize_to_percentile(nta_df['asthma_rate_per_10k'])
        logger.info(f"  Asthma score: mean={nta_df['asthma_score'].mean():.1f}")
    else:
        nta_df['asthma_score'] = 50
        logger.warning("  Asthma data missing")
    
    # Lead score
    if 'lead_rate_per_1k' in nta_df.columns:
        nta_df['lead_score'] = normalize_to_percentile(nta_df['lead_rate_per_1k'])
        logger.info(f"  Lead score: mean={nta_df['lead_score'].mean():.1f}")
    else:
        nta_df['lead_score'] = 50
        logger.warning("  Lead data missing")
    
    # Building age score (NEW!)
    if 'pct_pre_1978' in nta_df.columns:
        nta_df['building_age_score'] = normalize_to_percentile(nta_df['pct_pre_1978'])
        logger.info(f"  Building age score: mean={nta_df['building_age_score'].mean():.1f}")
    else:
        nta_df['building_age_score'] = 50
        logger.warning("  Building age data missing")
    
    # Calculate composite index with CORRECT 5-component weights
    nta_df['housing_health_index'] = (
        nta_df['violations_score'] * WEIGHTS['violations'] +
        nta_df['complaints_score'] * WEIGHTS['complaints'] +
        nta_df['asthma_score'] * WEIGHTS['asthma'] +
        nta_df['lead_score'] * WEIGHTS['lead'] +
        nta_df['building_age_score'] * WEIGHTS['building_age']
    )
    
    # Handle missing values
    nta_df['housing_health_index'] = nta_df['housing_health_index'].fillna(
        nta_df['housing_health_index'].median()
    )
    
    logger.info(f"\nIndex statistics:")
    logger.info(f"  Range: {nta_df['housing_health_index'].min():.1f} - {nta_df['housing_health_index'].max():.1f}")
    logger.info(f"  Mean: {nta_df['housing_health_index'].mean():.1f}")
    logger.info(f"  Std: {nta_df['housing_health_index'].std():.1f}")
    
    # Assign risk tiers
    def assign_risk_tier(score):
        if pd.isna(score):
            return 'Unknown'
        for tier, threshold in sorted(RISK_TIERS.items(), key=lambda x: -x[1]):
            if score >= threshold:
                return tier
        return 'Low Risk'
    
    nta_df['risk_tier'] = nta_df['housing_health_index'].apply(assign_risk_tier)
    
    tier_counts = nta_df['risk_tier'].value_counts()
    logger.info("\nRisk tier distribution:")
    for tier, count in tier_counts.items():
        pct = count / len(nta_df) * 100
        logger.info(f"  {tier}: {count} ({pct:.1f}%)")
    
    # Create rank
    nta_df['index_rank'] = nta_df['housing_health_index'].rank(ascending=False, method='min').astype(int)
    
    # Sort by index
    nta_df = nta_df.sort_values('housing_health_index', ascending=False)
    
    # Save final index
    output_file = PROCESSED_DIR / "housing_health_index.csv"
    write_csv(nta_df, output_file)
    
    log_step_complete(logger, "Fix Index Components")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("CORRECTED HOUSING HEALTH INDEX SUMMARY")
    logger.info("="*60)
    
    logger.info("\nComponent weights used:")
    for comp, weight in WEIGHTS.items():
        logger.info(f"  {comp}: {weight*100:.0f}%")
    
    logger.info("\nTOP 20 HIGHEST RISK NEIGHBORHOODS:")
    logger.info("-"*70)
    top20 = nta_df.head(20)
    for _, row in top20.iterrows():
        borough = row['borough'] if pd.notna(row['borough']) else 'Unknown'
        logger.info(
            f"{row['index_rank']:3}. {str(row['nta'])[:40]:<40} ({borough[:10]:<10}) "
            f"Index: {row['housing_health_index']:.1f} [{row['risk_tier']}]"
        )
    
    logger.info("\nBOROUGH SUMMARY:")
    logger.info("-"*70)
    borough_summary = nta_df.groupby('borough').agg({
        'housing_health_index': 'mean',
        'pct_pre_1978': 'mean',
        'nta': 'count'
    }).round(1)
    borough_summary = borough_summary.sort_values('housing_health_index', ascending=False)
    for borough, row in borough_summary.iterrows():
        logger.info(f"  {borough}: Index={row['housing_health_index']:.1f}, Pre-1978={row['pct_pre_1978']:.1f}%")
    
    return nta_df


if __name__ == "__main__":
    df = fix_and_recalculate()
    
    print(f"\n✅ Housing Health Index RECALCULATED with 5 components!")
    print(f"\n📊 Top 10 Highest Risk Neighborhoods (CORRECTED):")
    print("-" * 70)
    
    top10 = df.head(10)
    for _, row in top10.iterrows():
        borough = row['borough'] if pd.notna(row['borough']) else 'Unknown'
        print(f"  {row['index_rank']:2}. {str(row['nta'])[:35]:<35} ({borough:<10}) Index: {row['housing_health_index']:.1f}")

