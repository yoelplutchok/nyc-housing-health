"""
Calculate the NYC Child Health Housing Index (CHHI).

This script:
1. Loads neighborhood data with violations, complaints, health outcomes, building age, and demographics
2. Normalizes all indicators to 0-100 scale
3. Applies 311 bias adjustment using income data
4. Calculates composite Housing Health Index using configurable weights
5. Categorizes neighborhoods into risk tiers
6. Saves final analysis dataset

Input: data/processed/nta_with_demographics.csv
Output: data/processed/housing_health_index.csv
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

# Index weights (from configs/params.yml - must sum to 1.0)
# These weights reflect the relative importance of each component
WEIGHTS = {
    'violations': 0.30,      # Housing violations weight (direct measure of hazards)
    'complaints': 0.25,      # Housing complaints weight (resident-reported issues)
    'asthma': 0.20,          # Childhood asthma weight (primary health outcome)
    'lead': 0.15,            # Lead poisoning weight (critical but affecting fewer)
    'building_age': 0.10,    # Building age weight (proxy for lead/asbestos risk)
}

# Risk tier thresholds (percentiles)
RISK_TIERS = {
    'Very High Risk': 90,
    'High Risk': 75,
    'Elevated Risk': 50,
    'Moderate Risk': 25,
    'Low Risk': 0,
}


def normalize_to_percentile(series: pd.Series) -> pd.Series:
    """Normalize series to 0-100 percentile scale."""
    return series.rank(pct=True, na_option='keep') * 100


def calculate_index():
    """Calculate the Housing Health Index."""

    # Setup logging
    log_file = get_timestamped_log_filename("calculate_index")
    logger = setup_logger(__name__, log_file=log_file)

    log_step_start(logger, "Calculate Housing Health Index")

    # Ensure directories exist
    ensure_dirs_exist()

    # Load data (use improved health file if available)
    input_file = PROCESSED_DIR / "nta_with_health_improved.csv"
    if not input_file.exists():
        input_file = PROCESSED_DIR / "nta_with_demographics.csv"
        logger.warning("Improved health file not found, using demographics file")
    if not input_file.exists():
        input_file = PROCESSED_DIR / "nta_with_health.csv"
        logger.warning("Demographics file not found, using health file without building age")

    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    log_dataframe_info(logger, df, "Input data")

    # Calculate normalized indicators
    logger.info("Normalizing indicators to 0-100 scale...")
    logger.info(f"  Component weights: {WEIGHTS}")
    logger.info(f"  Total weight: {sum(WEIGHTS.values()):.2f}")

    # Violations indicator (use health violations per building)
    if 'violation_health_per_bldg' in df.columns:
        df['violations_score'] = normalize_to_percentile(df['violation_health_per_bldg'])
        logger.info(f"  Violations score: mean={df['violations_score'].mean():.1f} (weight: {WEIGHTS['violations']})")
    else:
        df['violations_score'] = 50  # Default neutral
        logger.warning("  Violations data not available, using default score")

    # Complaints indicator - use bias-adjusted score if available
    if 'complaints_score_adjusted' in df.columns:
        df['complaints_score'] = df['complaints_score_adjusted']
        logger.info(f"  Complaints score (bias-adjusted): mean={df['complaints_score'].mean():.1f} (weight: {WEIGHTS['complaints']})")
    elif 'complaint_health_per_bldg' in df.columns:
        df['complaints_score'] = normalize_to_percentile(df['complaint_health_per_bldg'])
        logger.info(f"  Complaints score (unadjusted): mean={df['complaints_score'].mean():.1f} (weight: {WEIGHTS['complaints']})")
    else:
        df['complaints_score'] = 50
        logger.warning("  Complaints data not available, using default score")

    # Asthma indicator
    if 'asthma_rate_per_10k' in df.columns:
        df['asthma_score'] = normalize_to_percentile(df['asthma_rate_per_10k'])
        logger.info(f"  Asthma score: mean={df['asthma_score'].mean():.1f} (weight: {WEIGHTS['asthma']})")
    else:
        df['asthma_score'] = 50
        logger.warning("  Asthma data not available, using default score")

    # Lead indicator
    if 'lead_rate_per_1k' in df.columns:
        df['lead_score'] = normalize_to_percentile(df['lead_rate_per_1k'])
        logger.info(f"  Lead score: mean={df['lead_score'].mean():.1f} (weight: {WEIGHTS['lead']})")
    else:
        df['lead_score'] = 50
        logger.warning("  Lead data not available, using default score")

    # Building age indicator (% pre-1978)
    if 'building_age_score' in df.columns:
        # Use pre-calculated building age score
        logger.info(f"  Building age score: mean={df['building_age_score'].mean():.1f} (weight: {WEIGHTS['building_age']})")
    elif 'pct_pre_1978' in df.columns:
        df['building_age_score'] = normalize_to_percentile(df['pct_pre_1978'])
        logger.info(f"  Building age score (calculated): mean={df['building_age_score'].mean():.1f} (weight: {WEIGHTS['building_age']})")
    else:
        df['building_age_score'] = 50
        logger.warning("  Building age data not available, using default score")

    # Calculate composite Housing Health Index (all 5 components)
    logger.info("\nCalculating composite Child Health Housing Index (CHHI)...")

    df['housing_health_index'] = (
        df['violations_score'] * WEIGHTS['violations'] +
        df['complaints_score'] * WEIGHTS['complaints'] +
        df['asthma_score'] * WEIGHTS['asthma'] +
        df['lead_score'] * WEIGHTS['lead'] +
        df['building_age_score'] * WEIGHTS['building_age']
    )
    
    # Handle missing values
    df['housing_health_index'] = df['housing_health_index'].fillna(df['housing_health_index'].median())
    
    logger.info(f"  Index range: {df['housing_health_index'].min():.1f} - {df['housing_health_index'].max():.1f}")
    logger.info(f"  Index mean: {df['housing_health_index'].mean():.1f}")
    logger.info(f"  Index std: {df['housing_health_index'].std():.1f}")
    
    # Assign risk tiers
    logger.info("\nAssigning risk tiers...")
    
    def assign_risk_tier(score):
        if pd.isna(score):
            return 'Unknown'
        for tier, threshold in sorted(RISK_TIERS.items(), key=lambda x: -x[1]):
            if score >= threshold:
                return tier
        return 'Low Risk'
    
    df['risk_tier'] = df['housing_health_index'].apply(assign_risk_tier)
    
    tier_counts = df['risk_tier'].value_counts()
    logger.info("Risk tier distribution:")
    for tier, count in tier_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  {tier}: {count} ({pct:.1f}%)")
    
    # Create rank column
    df['index_rank'] = df['housing_health_index'].rank(ascending=False, method='min').astype(int)
    
    # Sort by index (highest risk first)
    df = df.sort_values('housing_health_index', ascending=False)
    
    # Save
    output_file = PROCESSED_DIR / "housing_health_index.csv"
    logger.info(f"\nSaving to {output_file}...")
    write_csv(df, output_file)
    
    log_step_complete(logger, "Calculate Housing Health Index")
    
    # Summary
    logger.info("=" * 60)
    logger.info("HOUSING HEALTH INDEX SUMMARY")
    logger.info("=" * 60)
    
    # Top 20 highest risk neighborhoods
    logger.info("\nTOP 20 HIGHEST RISK NEIGHBORHOODS:")
    logger.info("-" * 60)
    top20 = df.head(20)
    for _, row in top20.iterrows():
        logger.info(
            f"{row['index_rank']:3}. {row['nta'][:40]:<40} ({row['borough'][:10]:<10}) "
            f"Index: {row['housing_health_index']:.1f} [{row['risk_tier']}]"
        )
    
    # Borough summary
    logger.info("\nBOROUGH SUMMARY:")
    logger.info("-" * 60)
    borough_summary = df.groupby('borough').agg({
        'housing_health_index': 'mean',
        'violation_health_per_bldg': 'mean',
        'asthma_rate_per_10k': 'mean',
        'nta': 'count'
    }).round(1)
    borough_summary.columns = ['Mean Index', 'Violations/Bldg', 'Asthma Rate', 'NTA Count']
    borough_summary = borough_summary.sort_values('Mean Index', ascending=False)
    for borough, row in borough_summary.iterrows():
        logger.info(f"  {borough}: Index={row['Mean Index']:.1f}, Violations={row['Violations/Bldg']:.1f}, Asthma={row['Asthma Rate']:.1f}")
    
    return df


if __name__ == "__main__":
    df = calculate_index()
    
    print(f"\n✅ Housing Health Index calculated for {len(df)} neighborhoods!")
    print(f"\n📊 Top 10 Highest Risk Neighborhoods:")
    print("-" * 70)
    
    top10 = df.head(10)
    for _, row in top10.iterrows():
        print(f"  {row['index_rank']:2}. {row['nta'][:35]:<35} ({row['borough']:<10}) Index: {row['housing_health_index']:.1f}")

