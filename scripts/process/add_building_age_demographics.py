"""
Add building age and demographics data to NTA aggregated data.

This script:
1. Joins PLUTO building age data (pct_pre_1978) to NTAs
2. Aggregates census demographics from tracts to NTAs
3. Saves enhanced neighborhood dataset

Input:
  - data/processed/nta_with_health.csv
  - data/raw/pluto_*.csv
  - data/raw/census_demographics_*.csv
  - data/geo/nta_boundaries_*.geojson

Output: data/processed/nta_with_demographics.csv
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np
import json

from housing_health.paths import PROCESSED_DIR, RAW_DIR, GEO_DIR, HEALTH_DIR, ensure_dirs_exist
from housing_health.io_utils import write_csv
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
    log_dataframe_info,
)


def load_pluto_building_age(logger):
    """
    Load PLUTO data and calculate building age statistics.
    Returns DataFrame with BBL-level year built data.
    """
    pluto_files = list(RAW_DIR.glob("pluto_*.csv"))
    if not pluto_files:
        logger.warning("No PLUTO file found")
        return None

    pluto_file = max(pluto_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading PLUTO data from {pluto_file}")

    # Read only needed columns
    df = pd.read_csv(
        pluto_file,
        usecols=['borough', 'BBL', 'yearbuilt', 'unitsres'],
        low_memory=False
    )
    df.columns = df.columns.str.lower()

    logger.info(f"  Loaded {len(df):,} property records")

    # Clean data
    df['yearbuilt'] = pd.to_numeric(df['yearbuilt'], errors='coerce')
    df['unitsres'] = pd.to_numeric(df['unitsres'], errors='coerce').fillna(0)

    # Filter to valid year built (1800-2025)
    valid_years = (df['yearbuilt'] >= 1800) & (df['yearbuilt'] <= 2025)
    df = df[valid_years]
    logger.info(f"  Valid year built records: {len(df):,}")

    # Filter to residential buildings (at least 1 unit)
    df = df[df['unitsres'] >= 1]
    logger.info(f"  Residential buildings: {len(df):,}")

    # Calculate building age indicators
    df['is_pre_1950'] = (df['yearbuilt'] < 1950).astype(int)
    df['is_pre_1978'] = (df['yearbuilt'] < 1978).astype(int)
    df['building_age'] = 2025 - df['yearbuilt']

    return df


def load_nta_boundaries():
    """Load NTA boundaries to get borough info and NTA names."""
    geo_files = list(GEO_DIR.glob("nta_boundaries_*.geojson"))
    if not geo_files:
        return None

    geo_file = max(geo_files, key=lambda x: x.stat().st_mtime)

    with open(geo_file, 'r') as f:
        geojson = json.load(f)

    nta_info = []
    for feature in geojson['features']:
        props = feature['properties']
        nta_info.append({
            'nta': props.get('ntaname'),
            'nta_code': props.get('nta2020'),
            'borough': props.get('boroname'),
        })

    return pd.DataFrame(nta_info)


def calculate_building_age_by_nta(pluto_df, violations_file, logger):
    """
    Calculate building age statistics by NTA.
    Uses violations BBL-NTA mapping since PLUTO doesn't have NTA directly.
    """
    logger.info("Calculating building age statistics by NTA...")

    # Load BBL-NTA mapping from violations
    logger.info("  Loading BBL-NTA mapping from violations...")
    bbl_nta = pd.read_csv(violations_file, usecols=['bbl', 'nta'], low_memory=False)
    bbl_nta = bbl_nta.dropna()

    # Standardize BBL format
    bbl_nta['bbl'] = bbl_nta['bbl'].astype(float).astype(int).astype(str)
    pluto_df['bbl'] = pluto_df['bbl'].astype(str)

    # Get unique BBL-NTA pairs
    bbl_nta_unique = bbl_nta.drop_duplicates()
    logger.info(f"  Unique BBL-NTA mappings: {len(bbl_nta_unique):,}")

    # Join PLUTO with NTA mapping
    pluto_with_nta = pluto_df.merge(bbl_nta_unique, on='bbl', how='inner')
    logger.info(f"  Matched buildings with NTA: {len(pluto_with_nta):,}")

    # Aggregate by NTA
    nta_building_age = pluto_with_nta.groupby('nta').agg({
        'bbl': 'count',
        'unitsres': 'sum',
        'is_pre_1950': 'sum',
        'is_pre_1978': 'sum',
        'building_age': 'mean',
        'yearbuilt': 'median'
    }).reset_index()

    nta_building_age.columns = [
        'nta', 'pluto_building_count', 'total_units',
        'buildings_pre_1950', 'buildings_pre_1978',
        'avg_building_age', 'median_year_built'
    ]

    # Calculate percentages
    nta_building_age['pct_pre_1950'] = (
        nta_building_age['buildings_pre_1950'] / nta_building_age['pluto_building_count'] * 100
    )
    nta_building_age['pct_pre_1978'] = (
        nta_building_age['buildings_pre_1978'] / nta_building_age['pluto_building_count'] * 100
    )

    logger.info(f"  Building age calculated for {len(nta_building_age)} NTAs")
    logger.info(f"  Avg pct_pre_1978: {nta_building_age['pct_pre_1978'].mean():.1f}%")

    return nta_building_age


def load_and_aggregate_census(logger):
    """
    Load census demographics and aggregate to borough level.
    Note: We aggregate to borough since tract-to-NTA crosswalk is complex.
    """
    census_files = list(RAW_DIR.glob("census_demographics_*.csv"))
    if not census_files:
        logger.warning("No census demographics file found")
        return None

    census_file = max(census_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading census demographics from {census_file}")

    df = pd.read_csv(census_file)
    logger.info(f"  Loaded {len(df):,} census tracts")

    # Map county codes to boroughs
    county_to_borough = {
        '005': 'Bronx',
        '047': 'Brooklyn',
        '061': 'Manhattan',
        '081': 'Queens',
        '085': 'Staten Island'
    }

    df['county'] = df['county'].astype(str).str.zfill(3)
    df['borough'] = df['county'].map(county_to_borough)

    # Clean numeric columns
    for col in ['total_population', 'white_alone', 'black_alone', 'hispanic', 'median_income', 'poverty_below']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter out missing/invalid income data
    df = df[df['median_income'] > 0]

    # Aggregate to borough level
    borough_demo = df.groupby('borough').agg({
        'total_population': 'sum',
        'white_alone': 'sum',
        'black_alone': 'sum',
        'hispanic': 'sum',
        'median_income': 'median',  # median of tract medians
        'poverty_below': 'sum',
    }).reset_index()

    # Calculate percentages
    borough_demo['pct_white'] = borough_demo['white_alone'] / borough_demo['total_population'] * 100
    borough_demo['pct_black'] = borough_demo['black_alone'] / borough_demo['total_population'] * 100
    borough_demo['pct_hispanic'] = borough_demo['hispanic'] / borough_demo['total_population'] * 100

    logger.info(f"  Aggregated demographics for {len(borough_demo)} boroughs")

    return borough_demo


def add_building_age_and_demographics():
    """Main function to add building age and demographics."""

    # Setup logging
    log_file = get_timestamped_log_filename("add_building_demographics")
    logger = setup_logger(__name__, log_file=log_file)

    log_step_start(logger, "Add Building Age and Demographics")

    # Ensure directories exist
    ensure_dirs_exist()

    # Load current NTA data
    input_file = PROCESSED_DIR / "nta_with_health.csv"
    logger.info(f"Loading NTA data from {input_file}")
    nta_df = pd.read_csv(input_file)
    log_dataframe_info(logger, nta_df, "Input NTA data")

    initial_cols = set(nta_df.columns)

    # Load and process PLUTO for building age
    pluto_df = load_pluto_building_age(logger)
    if pluto_df is not None:
        violations_file = PROCESSED_DIR / "violations_clean.csv"
        building_age_df = calculate_building_age_by_nta(pluto_df, violations_file, logger)

        # Join to NTA data
        nta_df = nta_df.merge(
            building_age_df[['nta', 'total_units', 'pct_pre_1950', 'pct_pre_1978', 'avg_building_age', 'median_year_built']],
            on='nta',
            how='left'
        )
        logger.info(f"  Joined building age for {nta_df['pct_pre_1978'].notna().sum()} NTAs")

    # Load and join census demographics (by borough)
    borough_demo = load_and_aggregate_census(logger)
    if borough_demo is not None and 'borough' in nta_df.columns:
        nta_df = nta_df.merge(
            borough_demo[['borough', 'median_income', 'pct_white', 'pct_black', 'pct_hispanic']],
            on='borough',
            how='left'
        )
        logger.info(f"  Joined demographics for {nta_df['median_income'].notna().sum()} NTAs")

    # Calculate building age score (percentile)
    if 'pct_pre_1978' in nta_df.columns:
        nta_df['building_age_score'] = nta_df['pct_pre_1978'].rank(pct=True, na_option='keep') * 100
        logger.info(f"  Building age score: mean={nta_df['building_age_score'].mean():.1f}")

    # Calculate 311 bias adjustment factor (using income)
    if 'median_income' in nta_df.columns:
        INCOME_ELASTICITY = 0.3
        city_median_income = nta_df['median_income'].median()
        logger.info(f"  City median income: ${city_median_income:,.0f}")

        nta_df['income_adjustment'] = (nta_df['median_income'] / city_median_income) ** INCOME_ELASTICITY

        # Adjusted complaint rate (higher for low-income areas)
        if 'complaint_health_per_bldg' in nta_df.columns:
            nta_df['complaint_health_adjusted'] = nta_df['complaint_health_per_bldg'] / nta_df['income_adjustment']
            nta_df['complaints_score_adjusted'] = nta_df['complaint_health_adjusted'].rank(pct=True, na_option='keep') * 100
            logger.info(f"  Adjusted complaints score: mean={nta_df['complaints_score_adjusted'].mean():.1f}")

    # Save
    output_file = PROCESSED_DIR / "nta_with_demographics.csv"
    logger.info(f"\nSaving to {output_file}...")
    write_csv(nta_df, output_file)

    new_cols = set(nta_df.columns) - initial_cols
    logger.info(f"  Added columns: {sorted(new_cols)}")

    log_step_complete(logger, "Add Building Age and Demographics")

    # Summary
    logger.info("=" * 60)
    logger.info("BUILDING AGE & DEMOGRAPHICS SUMMARY")
    logger.info("=" * 60)

    if 'pct_pre_1978' in nta_df.columns:
        logger.info("\nBuilding Age (Pre-1978 %):")
        logger.info(f"  Mean: {nta_df['pct_pre_1978'].mean():.1f}%")
        logger.info(f"  Median: {nta_df['pct_pre_1978'].median():.1f}%")
        logger.info(f"  Range: {nta_df['pct_pre_1978'].min():.1f}% - {nta_df['pct_pre_1978'].max():.1f}%")

        # Top 5 oldest neighborhoods
        logger.info("\nTop 5 Oldest Neighborhoods (by % pre-1978):")
        top5 = nta_df.nlargest(5, 'pct_pre_1978')[['nta', 'borough', 'pct_pre_1978', 'median_year_built']]
        for _, row in top5.iterrows():
            logger.info(f"  {row['nta']}: {row['pct_pre_1978']:.1f}% pre-1978")

    if 'median_income' in nta_df.columns:
        logger.info("\nMedian Income by Borough:")
        borough_income = nta_df.groupby('borough')['median_income'].first()
        for borough, income in borough_income.items():
            logger.info(f"  {borough}: ${income:,.0f}")

    return nta_df


if __name__ == "__main__":
    df = add_building_age_and_demographics()
    print(f"\n✅ Added building age and demographics for {len(df)} neighborhoods!")

    if 'pct_pre_1978' in df.columns:
        print(f"\n📊 Building Age Summary:")
        print(f"  Mean pre-1978: {df['pct_pre_1978'].mean():.1f}%")
        print(f"  Median pre-1978: {df['pct_pre_1978'].median():.1f}%")
