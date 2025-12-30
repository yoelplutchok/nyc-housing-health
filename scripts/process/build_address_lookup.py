"""
Build building-level lookup database for address search feature.

This script:
1. Aggregates violations by building (BBL)
2. Aggregates complaints by building
3. Joins PLUTO data for building characteristics (year built, units)
4. Calculates building-level risk scores
5. Creates SQLite database for fast lookups

Input:
  - data/processed/violations_clean.csv
  - data/processed/complaints_clean.csv
  - data/raw/pluto_*.csv
  - data/processed/housing_health_index.csv (for neighborhood context)

Output: data/processed/building_lookup.db (SQLite)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np
import sqlite3
import gc
import json

from housing_health.paths import PROCESSED_DIR, RAW_DIR, ensure_dirs_exist

# Path to NTA boundary GeoJSON
GEO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "geo"
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
    log_dataframe_info,
)

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
# These values can be adjusted to tune the data processing and scoring.
# Consider moving these to a YAML config file for easier management.

# Processing settings
CHUNK_SIZE = 500_000  # Number of rows to process at once (memory vs speed tradeoff)
NEARBY_HOUSE_NUM_RANGE = 10  # For address search: +/- this many house numbers

# Time period thresholds (in days)
ONE_YEAR_DAYS = 365
TWO_YEAR_DAYS = 730

# Violation severity weights by class
# Class C = Immediately hazardous (most serious)
# Class B = Hazardous
# Class A = Non-hazardous
# This is the ONLY factor used in the weighted score - simple and defensible
CLASS_SEVERITY_WEIGHTS = {
    'C': 3.0,  # Immediately hazardous - 3x weight
    'B': 2.0,  # Hazardous - 2x weight
    'A': 1.0,  # Non-hazardous - 1x weight
    'I': 2.5,  # Orders (vacate/repair) - between B and C
}

# Weight for rodent inspection failures (DOHMH data)
# Weighted same as Class B violations (health hazard)
RODENT_FAILURE_WEIGHT = 2.0

# Risk tier thresholds (percentile-based)
RISK_TIER_THRESHOLDS = {
    'Very High Risk': 95,  # Top 5%
    'High Risk': 85,       # Top 15%
    'Elevated Risk': 70,   # Top 30%
    'Moderate Risk': 50,   # Top 50%
    'Low Risk': 0,         # Bottom 50%
}

# Building risk score weights (100% building-specific - no neighborhood weighting)
# Weights based on health impact severity for children
# These should sum to approximately 1.0
BUILDING_WEIGHTS = {
    'lead_violations': 0.25,   # Most dangerous - brain damage, developmental issues
    'mold_violations': 0.18,   # Respiratory issues, especially for children with asthma
    'pest_violations': 0.12,   # Disease vectors from HPD violations
    'rodent_failures': 0.10,   # DOHMH rat inspections - disease vectors
    'heat_violations': 0.08,   # Safety concern but less direct health impact
    'complaints': 0.12,        # Indicates issues even without formal citations
    'bedbug_history': 0.05,    # Quality of life, allergens
    'building_age': 0.10,      # Pre-1978 lead paint risk (90% of NYC buildings)
}

# NYC geographic bounds (for coordinate validation)
NYC_BOUNDS = {
    'lat_min': 40.4774,
    'lat_max': 41.0000,
    'lon_min': -74.2591,
    'lon_max': -73.7002,
}

# Pre-1978 cutoff for lead paint risk
LEAD_PAINT_CUTOFF_YEAR = 1978


def assign_nta_from_coordinates(buildings_df, logger):
    """
    Assign NTA (Neighborhood Tabulation Area) to buildings using spatial join.

    Uses building lat/long from PLUTO to find which NTA polygon contains each building.
    This fills in NTA for buildings that don't have violations (and thus no NTA from HPD data).
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        logger.warning("geopandas not installed - skipping spatial NTA assignment")
        logger.warning("Install with: pip install geopandas")
        return buildings_df

    # Find NTA boundary file
    nta_files = list(GEO_DIR.glob("nta_boundaries_*.geojson"))
    if not nta_files:
        logger.warning("No NTA boundary GeoJSON found in data/geo/ - skipping spatial join")
        return buildings_df

    nta_file = max(nta_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading NTA boundaries from {nta_file}")

    # Load NTA boundaries
    nta_gdf = gpd.read_file(nta_file)
    logger.info(f"  Loaded {len(nta_gdf)} NTA polygons")

    # Get NTA code and name columns
    nta_code_col = 'nta2020' if 'nta2020' in nta_gdf.columns else 'ntacode'
    nta_name_col = 'ntaname' if 'ntaname' in nta_gdf.columns else nta_code_col

    # Count buildings that need NTA assignment
    needs_nta = buildings_df['nta'].isna() | (buildings_df['nta'] == '')
    has_coords = buildings_df['latitude'].notna() & buildings_df['longitude'].notna()
    can_assign = needs_nta & has_coords

    logger.info(f"  Buildings needing NTA: {needs_nta.sum():,}")
    logger.info(f"  Buildings with coordinates: {has_coords.sum():,}")
    logger.info(f"  Buildings that can be assigned: {can_assign.sum():,}")

    if can_assign.sum() == 0:
        logger.info("  No buildings to assign - all have NTA or lack coordinates")
        return buildings_df

    # Create GeoDataFrame from buildings needing NTA
    buildings_to_assign = buildings_df[can_assign].copy()
    geometry = [Point(xy) for xy in zip(buildings_to_assign['longitude'], buildings_to_assign['latitude'])]
    buildings_gdf = gpd.GeoDataFrame(buildings_to_assign, geometry=geometry, crs="EPSG:4326")

    # Ensure NTA boundaries are in same CRS
    if nta_gdf.crs != buildings_gdf.crs:
        nta_gdf = nta_gdf.to_crs(buildings_gdf.crs)

    # Perform spatial join
    logger.info("  Performing spatial join (this may take a few minutes)...")
    joined = gpd.sjoin(buildings_gdf, nta_gdf[[nta_code_col, nta_name_col, 'geometry']],
                       how='left', predicate='within')

    # Update NTA in original dataframe
    # Use nta_name for consistency with existing data
    nta_mapping = joined.set_index(buildings_gdf.index)[nta_name_col]
    buildings_df.loc[can_assign, 'nta'] = nta_mapping

    # Also store NTA code if we want it later
    if nta_code_col != nta_name_col:
        nta_code_mapping = joined.set_index(buildings_gdf.index)[nta_code_col]
        buildings_df.loc[can_assign, 'nta_code'] = nta_code_mapping

    # Report results
    newly_assigned = buildings_df.loc[can_assign, 'nta'].notna().sum()
    still_missing = buildings_df['nta'].isna().sum()

    logger.info(f"  Assigned NTA to {newly_assigned:,} buildings via spatial join")
    logger.info(f"  Still missing NTA: {still_missing:,} (likely outside NTA boundaries or missing coords)")

    return buildings_df


def aggregate_violations_by_building(violations_file, logger):
    """Aggregate violations to building level using chunked processing."""

    logger.info("Aggregating violations by building (BBL)...")

    cols_to_read = [
        'violationid', 'bbl', 'buildingid', 'nta', 'boro', 'housenumber', 'streetname', 'zip',
        'class', 'is_lead', 'is_mold', 'is_pests', 'is_heat', 'is_health_related',
        'severity_weight', 'inspectiondate', 'violationstatus'
    ]

    agg_accum = None
    total_rows = 0
    deduped_rows = 0

    chunk_iter = pd.read_csv(
        violations_file,
        usecols=lambda c: c in cols_to_read,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for i, chunk in enumerate(chunk_iter):
        total_rows += len(chunk)

        # Remove duplicate violation IDs (data quality fix)
        if 'violationid' in chunk.columns:
            before_len = len(chunk)
            chunk = chunk.drop_duplicates(subset=['violationid'], keep='first')
            deduped_rows += before_len - len(chunk)

        # Filter to valid BBL
        chunk = chunk[chunk['bbl'].notna()]
        chunk['bbl'] = chunk['bbl'].astype(float).astype(int).astype(str)

        # Standardize address fields before grouping
        chunk['housenumber'] = chunk['housenumber'].astype(str).str.strip().str.upper()
        chunk['streetname'] = chunk['streetname'].astype(str).str.strip().str.upper()
        
        # Create open/closed flags
        chunk['is_open'] = (chunk['violationstatus'].str.upper() == 'OPEN').astype(int)
        chunk['is_closed'] = (chunk['violationstatus'].str.upper() == 'CLOSE').astype(int)

        # Parse inspection date for time-based filtering
        chunk['inspectiondate'] = pd.to_datetime(chunk['inspectiondate'], errors='coerce')
        from datetime import datetime, timedelta
        now = datetime.now()
        one_year_ago = now - timedelta(days=ONE_YEAR_DAYS)
        two_years_ago = now - timedelta(days=TWO_YEAR_DAYS)

        # Time-based flags
        chunk['is_last_1yr'] = (chunk['inspectiondate'] >= one_year_ago).astype(int)
        chunk['is_last_2yr'] = (chunk['inspectiondate'] >= two_years_ago).astype(int)
        chunk['is_1_to_2yr'] = ((chunk['inspectiondate'] >= two_years_ago) &
                                (chunk['inspectiondate'] < one_year_ago)).astype(int)
        chunk['is_older'] = (chunk['inspectiondate'] < two_years_ago).astype(int)

        # Create violation class flags (A, B, C, I)
        chunk['class_upper'] = chunk['class'].astype(str).str.upper().str.strip()
        chunk['is_class_a'] = (chunk['class_upper'] == 'A').astype(int)
        chunk['is_class_b'] = (chunk['class_upper'] == 'B').astype(int)
        chunk['is_class_c'] = (chunk['class_upper'] == 'C').astype(int)
        chunk['is_class_i'] = (chunk['class_upper'] == 'I').astype(int)

        # Create open violation flags for each category
        chunk['lead_open'] = (chunk['is_lead'] == 1) & (chunk['is_open'] == 1)
        chunk['mold_open'] = (chunk['is_mold'] == 1) & (chunk['is_open'] == 1)
        chunk['pest_open'] = (chunk['is_pests'] == 1) & (chunk['is_open'] == 1)
        chunk['heat_open'] = (chunk['is_heat'] == 1) & (chunk['is_open'] == 1)

        # Time-based violation counts
        chunk['violations_1yr'] = chunk['is_last_1yr']
        chunk['violations_2yr'] = chunk['is_last_2yr']
        chunk['violations_open_1yr'] = chunk['is_open'] * chunk['is_last_1yr']
        chunk['violations_open_2yr'] = chunk['is_open'] * chunk['is_last_2yr']

        # ============================================================
        # WEIGHTED VIOLATION SCORE CALCULATION
        # Uses ONLY severity (class) weighting - simple and defensible
        # Class C = 3x, Class B = 2x, Class A = 1x, Class I = 2.5x
        # Open/closed status and recency are shown separately in UI
        # ============================================================

        # Severity weight based on class (the ONLY factor in weighted score)
        chunk['weighted_score'] = chunk['class_upper'].map(CLASS_SEVERITY_WEIGHTS).fillna(1.0)

        # Aggregate by BBL + address (to handle multiple addresses per BBL)
        chunk_agg = chunk.groupby(['bbl', 'housenumber', 'streetname']).agg({
            'buildingid': 'first',
            'nta': 'first',
            'boro': 'first',
            'zip': 'first',
            'class': 'count',  # total violations
            'is_open': 'sum',  # open violations
            'is_closed': 'sum',  # closed violations
            'is_class_a': 'sum',
            'is_class_b': 'sum',
            'is_class_c': 'sum',
            'is_class_i': 'sum',
            'is_lead': 'sum',
            'is_mold': 'sum',
            'is_pests': 'sum',
            'is_heat': 'sum',
            'lead_open': 'sum',
            'mold_open': 'sum',
            'pest_open': 'sum',
            'heat_open': 'sum',
            'is_health_related': 'sum',
            'severity_weight': 'sum',
            'inspectiondate': 'max',  # most recent
            # Time-based counts
            'violations_1yr': 'sum',
            'violations_2yr': 'sum',
            'violations_open_1yr': 'sum',
            'violations_open_2yr': 'sum',
            # Weighted score (combines severity, status, recency)
            'weighted_score': 'sum',
        }).reset_index()

        chunk_agg.rename(columns={
            'class': 'violation_count',
            'is_open': 'violations_open',
            'is_closed': 'violations_closed',
            'is_class_a': 'class_a_count',
            'is_class_b': 'class_b_count',
            'is_class_c': 'class_c_count',
            'is_class_i': 'class_i_count',
            'is_lead': 'lead_violations',
            'is_mold': 'mold_violations',
            'is_pests': 'pest_violations',
            'is_heat': 'heat_violations',
            'lead_open': 'lead_violations_open',
            'mold_open': 'mold_violations_open',
            'pest_open': 'pest_violations_open',
            'heat_open': 'heat_violations_open',
            'is_health_related': 'health_violations',
            'severity_weight': 'violation_severity',
            'inspectiondate': 'last_violation_date',
        }, inplace=True)

        if agg_accum is None:
            agg_accum = chunk_agg
        else:
            # Combine with accumulator (sum numeric, keep first for text)
            combined = pd.concat([agg_accum, chunk_agg])
            agg_accum = combined.groupby(['bbl', 'housenumber', 'streetname']).agg({
                'buildingid': 'first',
                'nta': 'first',
                'boro': 'first',
                'zip': 'first',
                'violation_count': 'sum',
                'violations_open': 'sum',
                'violations_closed': 'sum',
                'class_a_count': 'sum',
                'class_b_count': 'sum',
                'class_c_count': 'sum',
                'class_i_count': 'sum',
                'lead_violations': 'sum',
                'mold_violations': 'sum',
                'pest_violations': 'sum',
                'heat_violations': 'sum',
                'lead_violations_open': 'sum',
                'mold_violations_open': 'sum',
                'pest_violations_open': 'sum',
                'heat_violations_open': 'sum',
                'health_violations': 'sum',
                'violation_severity': 'sum',
                'last_violation_date': 'max',
                # Time-based counts
                'violations_1yr': 'sum',
                'violations_2yr': 'sum',
                'violations_open_1yr': 'sum',
                'violations_open_2yr': 'sum',
                # Weighted score
                'weighted_score': 'sum',
            }).reset_index()

        if (i + 1) % 5 == 0:
            logger.info(f"  Processed {total_rows:,} violation records...")
        gc.collect()

    logger.info(f"  Total violations processed: {total_rows:,}")
    logger.info(f"  Unique buildings: {len(agg_accum):,}")

    return agg_accum


def aggregate_complaints_by_building(complaints_file, logger):
    """Aggregate complaints to building level using chunked processing."""

    logger.info("Aggregating complaints by building (BBL)...")

    cols_to_read = [
        'bbl', 'complaint_id', 'problem_id', 'house_number', 'street_name',
        'is_heat', 'is_mold', 'is_pests', 'is_lead', 'is_water', 'is_health_related',
        'severity_weight', 'received_date'
    ]

    agg_accum = None
    complaint_ids_per_key = {}
    total_rows = 0

    chunk_iter = pd.read_csv(
        complaints_file,
        usecols=lambda c: c in cols_to_read,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for i, chunk in enumerate(chunk_iter):
        total_rows += len(chunk)

        # Filter to valid BBL
        chunk = chunk[chunk['bbl'].notna()]
        chunk['bbl'] = chunk['bbl'].astype(float).astype(int).astype(str)
        
        # Standardize address fields
        chunk['house_number'] = chunk['house_number'].astype(str).str.strip().str.upper()
        chunk['street_name'] = chunk['street_name'].astype(str).str.strip().str.upper()

        # Track unique complaints per BBL+address
        for key, group in chunk.groupby(['bbl', 'house_number', 'street_name']):
            if key not in complaint_ids_per_key:
                complaint_ids_per_key[key] = set()
            complaint_ids_per_key[key].update(group['complaint_id'].dropna().unique())

        # Aggregate by BBL + address
        chunk_agg = chunk.groupby(['bbl', 'house_number', 'street_name']).agg({
            'problem_id': 'count',
            'is_heat': 'sum',
            'is_mold': 'sum',
            'is_pests': 'sum',
            'is_lead': 'sum',
            'is_water': 'sum',
            'is_health_related': 'sum',
            'severity_weight': 'sum',
            'received_date': 'max',
        }).reset_index()

        chunk_agg.rename(columns={
            'problem_id': 'problem_count',
            'is_heat': 'complaint_heat',
            'is_mold': 'complaint_mold',
            'is_pests': 'complaint_pests',
            'is_lead': 'complaint_lead',
            'is_water': 'complaint_water',
            'is_health_related': 'complaint_health',
            'severity_weight': 'complaint_severity',
            'received_date': 'last_complaint_date',
        }, inplace=True)

        if agg_accum is None:
            agg_accum = chunk_agg
        else:
            combined = pd.concat([agg_accum, chunk_agg])
            agg_accum = combined.groupby(['bbl', 'house_number', 'street_name']).agg({
                'problem_count': 'sum',
                'complaint_heat': 'sum',
                'complaint_mold': 'sum',
                'complaint_pests': 'sum',
                'complaint_lead': 'sum',
                'complaint_water': 'sum',
                'complaint_health': 'sum',
                'complaint_severity': 'sum',
                'last_complaint_date': 'max',
            }).reset_index()

        if (i + 1) % 5 == 0:
            logger.info(f"  Processed {total_rows:,} complaint records...")
        gc.collect()

    # Add complaint count (unique complaints per address)
    agg_accum['complaint_count'] = agg_accum.apply(
        lambda row: len(complaint_ids_per_key.get((row['bbl'], row['house_number'], row['street_name']), set())),
        axis=1
    )

    logger.info(f"  Total complaints processed: {total_rows:,}")
    logger.info(f"  Unique buildings with complaints: {len(agg_accum):,}")

    return agg_accum


def aggregate_rodent_inspections(logger):
    """Aggregate DOHMH rodent inspections by BBL."""

    rodent_file = RAW_DIR / "rodent_inspections.csv"
    if not rodent_file.exists():
        logger.warning("No rodent inspection data found")
        return None

    logger.info("Aggregating rodent inspections by building...")

    # Read rodent data
    df = pd.read_csv(rodent_file, low_memory=False,
                     usecols=['bbl', 'result', 'inspection_date'])

    # Clean BBL - convert to int first to remove .0 suffix
    df = df[df['bbl'].notna()]
    df['bbl'] = pd.to_numeric(df['bbl'], errors='coerce')
    df = df[df['bbl'].notna() & (df['bbl'] > 0)]
    df['bbl'] = df['bbl'].astype(int).astype(str)

    # Count failures (rat activity or failed)
    df['is_failure'] = df['result'].str.contains('Rat Activity|Failed', case=False, na=False).astype(int)

    # Aggregate by BBL
    agg = df.groupby('bbl').agg({
        'result': 'count',  # total inspections
        'is_failure': 'sum',  # failed inspections
        'inspection_date': 'max',  # most recent
    }).reset_index()

    agg.rename(columns={
        'result': 'rodent_inspections',
        'is_failure': 'rodent_failures',
        'inspection_date': 'last_rodent_inspection',
    }, inplace=True)

    logger.info(f"  Rodent inspections: {len(df):,} records")
    logger.info(f"  Buildings with inspections: {len(agg):,}")
    logger.info(f"  Buildings with failures: {(agg['rodent_failures'] > 0).sum():,}")

    return agg


def aggregate_bedbug_reports(logger):
    """Aggregate bedbug reports by BBL."""

    bedbug_file = RAW_DIR / "bedbug_reports.csv"
    if not bedbug_file.exists():
        logger.warning("No bedbug report data found")
        return None

    logger.info("Aggregating bedbug reports by building...")

    # Read bedbug data
    df = pd.read_csv(bedbug_file, low_memory=False,
                     usecols=['bbl', 'infested_dwelling_unit_count', 'filing_date'])

    # Clean BBL - convert to int first to remove .0 suffix
    df = df[df['bbl'].notna()]
    df['bbl'] = pd.to_numeric(df['bbl'], errors='coerce')
    df = df[df['bbl'].notna() & (df['bbl'] > 0)]
    df['bbl'] = df['bbl'].astype(int).astype(str)

    # Clean infestation count
    df['infested_dwelling_unit_count'] = pd.to_numeric(
        df['infested_dwelling_unit_count'], errors='coerce').fillna(0)

    # Aggregate by BBL - sum total infestations, count reports
    agg = df.groupby('bbl').agg({
        'infested_dwelling_unit_count': 'sum',
        'filing_date': ['count', 'max'],
    }).reset_index()

    agg.columns = ['bbl', 'bedbug_infested_units', 'bedbug_reports', 'last_bedbug_report']

    # Create binary flag for any history
    agg['has_bedbug_history'] = (agg['bedbug_infested_units'] > 0).astype(int)

    logger.info(f"  Bedbug reports: {len(df):,} records")
    logger.info(f"  Buildings with reports: {len(agg):,}")
    logger.info(f"  Buildings with infestations: {(agg['has_bedbug_history'] > 0).sum():,}")

    return agg


def load_pluto_data(logger):
    """Load PLUTO data for building characteristics."""

    pluto_files = list(RAW_DIR.glob("pluto_*.csv"))
    if not pluto_files:
        logger.warning("No PLUTO file found")
        return None

    pluto_file = max(pluto_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading PLUTO data from {pluto_file}")

    # Read only needed columns
    cols_to_read = ['BBL', 'borough', 'address', 'yearbuilt', 'unitsres', 'unitstotal',
                    'numfloors', 'bldgclass', 'latitude', 'longitude']

    df = pd.read_csv(
        pluto_file,
        usecols=lambda c: c.upper() in [col.upper() for col in cols_to_read],
        low_memory=False
    )
    df.columns = df.columns.str.lower()

    # Standardize BBL
    df['bbl'] = df['bbl'].astype(str)

    # Clean numeric columns
    df['yearbuilt'] = pd.to_numeric(df['yearbuilt'], errors='coerce')
    df['unitsres'] = pd.to_numeric(df['unitsres'], errors='coerce').fillna(0)
    df['numfloors'] = pd.to_numeric(df['numfloors'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Filter to valid year built
    valid_years = (df['yearbuilt'] >= 1800) & (df['yearbuilt'] <= 2025)
    df.loc[~valid_years, 'yearbuilt'] = np.nan

    # Calculate building age indicators
    df['is_pre_1978'] = (df['yearbuilt'] < 1978).astype(int)
    df['building_age'] = 2025 - df['yearbuilt']

    logger.info(f"  Loaded {len(df):,} PLUTO records")

    # Rename for clarity
    df.rename(columns={
        'borough': 'pluto_borough',
        'address': 'pluto_address',
    }, inplace=True)

    return df


def smart_percentile(series):
    """
    Calculate percentile that treats 0 as 'best' (low percentile).

    Buildings with 0 violations get percentile based on how many buildings also have 0.
    Buildings with >0 violations get percentile 50-100 based on their rank among non-zero.

    This ensures a building with 0 violations shows as ~1st-10th percentile (better than most),
    rather than ~48th percentile (which is misleading).
    """
    result = pd.Series(index=series.index, dtype=float)

    zero_mask = series == 0
    non_zero_mask = series > 0

    n_total = len(series)
    n_zeros = zero_mask.sum()
    n_non_zero = non_zero_mask.sum()

    if n_zeros > 0:
        # Buildings with 0 get low percentiles (1 to proportion_of_zeros * 50)
        # If 76% have 0, they get spread across 1-38th percentile
        zero_pct_max = (n_zeros / n_total) * 50
        result[zero_mask] = zero_pct_max / 2  # Average for zeros

    if n_non_zero > 0:
        # Non-zero buildings get percentiles from 50 to 100
        non_zero_pct = series[non_zero_mask].rank(pct=True, method='average') * 50 + 50
        result[non_zero_mask] = non_zero_pct

    return result


def calculate_building_risk_scores(buildings_df, neighborhood_df, logger):
    """Calculate percentile rankings for each building (citywide and by neighborhood)."""

    logger.info("Calculating building percentile rankings...")

    # Fill missing values with 0 for all count columns
    count_cols = ['lead_violations', 'mold_violations', 'pest_violations', 'heat_violations',
                  'violations_open', 'violations_closed', 'violation_count',
                  'lead_violations_open', 'mold_violations_open', 'pest_violations_open', 'heat_violations_open',
                  'complaint_count', 'rodent_failures', 'rodent_inspections',
                  'bedbug_reports', 'bedbug_infested_units',
                  'class_a_count', 'class_b_count', 'class_c_count', 'class_i_count']
    for col in count_cols:
        if col in buildings_df.columns:
            buildings_df[col] = buildings_df[col].fillna(0)

    # Get current year dynamically
    from datetime import datetime
    current_year = datetime.now().year

    # ========== CITYWIDE PERCENTILES ==========
    logger.info("  Calculating citywide percentiles (using smart percentile for zeros)...")

    # Violation categories (total) - using smart percentile
    for col in ['lead_violations', 'mold_violations', 'pest_violations', 'heat_violations']:
        if col in buildings_df.columns:
            buildings_df[f'{col}_pct'] = smart_percentile(buildings_df[col])
            # Keep legacy score column for backward compatibility
            buildings_df[f'{col}_score'] = buildings_df[f'{col}_pct']

    # Open violations specifically
    if 'violations_open' in buildings_df.columns:
        buildings_df['violations_open_pct'] = smart_percentile(buildings_df['violations_open'])

    # Class-based percentiles (A=non-hazardous, B=hazardous, C=immediately hazardous)
    for class_col in ['class_a_count', 'class_b_count', 'class_c_count', 'class_i_count']:
        if class_col in buildings_df.columns:
            buildings_df[class_col] = buildings_df[class_col].fillna(0)
            buildings_df[f'{class_col}_pct'] = smart_percentile(buildings_df[class_col])

    # Total violations percentile
    if 'violation_count' in buildings_df.columns:
        buildings_df['violation_count_pct'] = smart_percentile(buildings_df['violation_count'])

    # ========== TIME-BASED PERCENTILES ==========
    logger.info("  Calculating time-based percentiles...")
    for time_col in ['violations_1yr', 'violations_2yr', 'violations_open_1yr', 'violations_open_2yr']:
        if time_col in buildings_df.columns:
            buildings_df[time_col] = buildings_df[time_col].fillna(0)
            buildings_df[f'{time_col}_pct'] = smart_percentile(buildings_df[time_col])

    # ========== WEIGHTED SCORE PERCENTILES ==========
    # weighted_score combines: severity (class), status (open/closed), and recency
    logger.info("  Calculating weighted score percentiles...")
    if 'weighted_score' in buildings_df.columns:
        buildings_df['weighted_score'] = buildings_df['weighted_score'].fillna(0)
        buildings_df['weighted_score_pct'] = smart_percentile(buildings_df['weighted_score'])

    # ========== VIOLATIONS PER UNIT (normalized by building size) ==========
    logger.info("  Calculating violations per unit...")
    if 'violation_count' in buildings_df.columns and 'unitsres' in buildings_df.columns:
        # Avoid division by zero - use max(units, 1)
        units_safe = buildings_df['unitsres'].clip(lower=1)
        buildings_df['violations_per_unit'] = buildings_df['violation_count'] / units_safe
        buildings_df['violations_per_unit_pct'] = smart_percentile(buildings_df['violations_per_unit'])

        # Also calculate open violations per unit
        if 'violations_open' in buildings_df.columns:
            buildings_df['open_violations_per_unit'] = buildings_df['violations_open'] / units_safe
            buildings_df['open_violations_per_unit_pct'] = smart_percentile(buildings_df['open_violations_per_unit'])

        # Complaints per unit
        if 'complaint_count' in buildings_df.columns:
            buildings_df['complaints_per_unit'] = buildings_df['complaint_count'] / units_safe
            buildings_df['complaints_per_unit_pct'] = smart_percentile(buildings_df['complaints_per_unit'])

        # Weighted score per unit (the most comprehensive metric)
        if 'weighted_score' in buildings_df.columns:
            buildings_df['weighted_score_per_unit'] = buildings_df['weighted_score'] / units_safe
            buildings_df['weighted_score_per_unit_pct'] = smart_percentile(buildings_df['weighted_score_per_unit'])

    # ========== COMBINED SCORE (HPD violations + Rodent failures) ==========
    # Combines weighted violation score with rodent inspection failures
    logger.info("  Calculating combined score (violations + rodent failures)...")
    if 'weighted_score' in buildings_df.columns:
        buildings_df['weighted_score'] = buildings_df['weighted_score'].fillna(0)

        # Add rodent failures to the score (weighted at 2.0 each, same as Class B)
        if 'rodent_failures' in buildings_df.columns:
            buildings_df['rodent_failures'] = buildings_df['rodent_failures'].fillna(0)
            buildings_df['combined_score'] = (
                buildings_df['weighted_score'] +
                buildings_df['rodent_failures'] * RODENT_FAILURE_WEIGHT
            )
            logger.info(f"    Added rodent failures to score (weight={RODENT_FAILURE_WEIGHT})")
        else:
            buildings_df['combined_score'] = buildings_df['weighted_score']
            logger.info("    No rodent data - using violations only")

        buildings_df['combined_score_pct'] = smart_percentile(buildings_df['combined_score'])

    # ========== ADJUSTED PERCENTILE (primary comparison metric) ==========
    # For multi-unit buildings (>1 unit): use per-unit combined score
    # For single-family homes: use raw combined score
    # This ensures fair comparison across different building sizes
    logger.info("  Calculating adjusted percentile (primary metric)...")
    if 'combined_score' in buildings_df.columns and 'unitsres' in buildings_df.columns:
        # Calculate per-unit combined score
        units_safe = buildings_df['unitsres'].clip(lower=1)
        buildings_df['combined_score_per_unit'] = buildings_df['combined_score'] / units_safe
        buildings_df['combined_score_per_unit_pct'] = smart_percentile(buildings_df['combined_score_per_unit'])

        # Start with raw combined score percentile
        buildings_df['adjusted_score_pct'] = buildings_df['combined_score_pct']

        # For buildings with >1 unit, use per-unit percentile instead
        multi_unit_mask = buildings_df['unitsres'] > 1
        buildings_df.loc[multi_unit_mask, 'adjusted_score_pct'] = \
            buildings_df.loc[multi_unit_mask, 'combined_score_per_unit_pct']

        logger.info(f"    Multi-unit buildings using per-unit score: {multi_unit_mask.sum():,}")
        logger.info(f"    Single-unit buildings using raw score: {(~multi_unit_mask).sum():,}")

    # Complaints
    if 'complaint_count' in buildings_df.columns:
        buildings_df['complaints_pct'] = smart_percentile(buildings_df['complaint_count'])
        buildings_df['complaints_score'] = buildings_df['complaints_pct']
    else:
        buildings_df['complaints_pct'] = 0
        buildings_df['complaints_score'] = 0

    # Rodent failures
    if 'rodent_failures' in buildings_df.columns:
        buildings_df['rodent_failures_pct'] = smart_percentile(buildings_df['rodent_failures'])
        buildings_df['rodent_score'] = buildings_df['rodent_failures_pct']
    else:
        buildings_df['rodent_failures_pct'] = 0
        buildings_df['rodent_score'] = 0

    # Bedbug (binary - has history or not)
    if 'has_bedbug_history' in buildings_df.columns:
        buildings_df['has_bedbug_history'] = buildings_df['has_bedbug_history'].fillna(0)
        buildings_df['bedbug_score'] = buildings_df['has_bedbug_history'] * 100
    else:
        buildings_df['bedbug_score'] = 0

    # Building age percentile (by year built)
    if 'yearbuilt' in buildings_df.columns and 'is_pre_1978' in buildings_df.columns:
        # Older buildings rank higher (higher percentile = older)
        # First filter valid years
        valid_years = buildings_df['yearbuilt'].notna() & (buildings_df['yearbuilt'] > 1800)
        buildings_df['building_age_pct'] = 50  # Default
        buildings_df.loc[valid_years, 'building_age_pct'] = (
            (current_year - buildings_df.loc[valid_years, 'yearbuilt']).rank(pct=True, method='average') * 100
        )
        buildings_df['building_age_score'] = buildings_df['building_age_pct']
    else:
        buildings_df['building_age_pct'] = 50
        buildings_df['building_age_score'] = 50

    # ========== NEIGHBORHOOD PERCENTILES ==========
    logger.info("  Calculating neighborhood percentiles...")
    
    # Calculate percentiles within each NTA
    pct_cols_map = {
        'violation_count': 'violation_count_nhood_pct',
        'lead_violations': 'lead_violations_nhood_pct',
        'mold_violations': 'mold_violations_nhood_pct',
        'pest_violations': 'pest_violations_nhood_pct',
        'heat_violations': 'heat_violations_nhood_pct',
        'violations_open': 'violations_open_nhood_pct',
        'complaint_count': 'complaints_nhood_pct',
        'rodent_failures': 'rodent_failures_nhood_pct',
    }
    
    for src_col, pct_col in pct_cols_map.items():
        if src_col in buildings_df.columns and 'nta' in buildings_df.columns:
            buildings_df[pct_col] = buildings_df.groupby('nta')[src_col].transform(
                lambda x: x.rank(pct=True, method='average') * 100
            )
            # Fill NaN (e.g., NTAs with only 1 building) with citywide percentile
            citywide_col = src_col.replace('complaint_count', 'complaints') + '_pct'
            if citywide_col in buildings_df.columns:
                buildings_df[pct_col] = buildings_df[pct_col].fillna(buildings_df[citywide_col])
            else:
                buildings_df[pct_col] = buildings_df[pct_col].fillna(50)
        else:
            buildings_df[pct_col] = 50
    
    # ========== LEGACY COMPOSITE SCORES (for backward compatibility) ==========
    # Fill any remaining NaN in score columns
    score_cols = ['lead_violations_score', 'mold_violations_score', 'pest_violations_score',
                  'heat_violations_score', 'complaints_score', 'rodent_score', 'bedbug_score',
                  'building_age_score']
    for col in score_cols:
        if col in buildings_df.columns:
            buildings_df[col] = buildings_df[col].fillna(50)

    # Calculate building-specific composite score (kept for backward compatibility)
    buildings_df['building_score'] = (
        buildings_df['lead_violations_score'] * BUILDING_WEIGHTS['lead_violations'] +
        buildings_df['mold_violations_score'] * BUILDING_WEIGHTS['mold_violations'] +
        buildings_df['pest_violations_score'] * BUILDING_WEIGHTS['pest_violations'] +
        buildings_df['rodent_score'] * BUILDING_WEIGHTS['rodent_failures'] +
        buildings_df['heat_violations_score'] * BUILDING_WEIGHTS['heat_violations'] +
        buildings_df['complaints_score'] * BUILDING_WEIGHTS['complaints'] +
        buildings_df['bedbug_score'] * BUILDING_WEIGHTS['bedbug_history'] +
        buildings_df['building_age_score'] * BUILDING_WEIGHTS['building_age']
    )

    # Store neighborhood context
    if neighborhood_df is not None and 'housing_health_index' in neighborhood_df.columns:
        nta_index = neighborhood_df.set_index('nta')['housing_health_index'].to_dict()
        buildings_df['neighborhood_index'] = buildings_df['nta'].map(nta_index)
    else:
        buildings_df['neighborhood_index'] = None

    # Final risk score = building composite (kept for backward compatibility)
    buildings_df['risk_score'] = buildings_df['building_score']

    # Risk tier based on score
    has_issues = (buildings_df['violation_count'].fillna(0) > 0) | (buildings_df['complaint_count'].fillna(0) > 0)
    
    def assign_tier(row):
        score = row['risk_score']
        has_data = row['has_issues']
        
        if pd.isna(score):
            return 'Unknown'
        elif not has_data:
            return 'Low Risk'
        elif score >= 75:
            return 'High Risk'
        elif score >= 50:
            return 'Elevated Risk'
        elif score >= 25:
            return 'Moderate Risk'
        else:
            return 'Low Risk'

    buildings_df['has_issues'] = has_issues
    buildings_df['risk_tier'] = buildings_df.apply(assign_tier, axis=1)
    buildings_df.drop(columns=['has_issues'], inplace=True)

    logger.info(f"  Citywide percentile calculations complete")
    logger.info(f"  Neighborhood percentile calculations complete")

    return buildings_df


def create_sqlite_database(buildings_df, output_path, logger, nonres_df=None):
    """Create SQLite database for fast address lookups.

    Args:
        buildings_df: DataFrame with residential building data
        output_path: Path to save SQLite database
        logger: Logger instance
        nonres_df: Optional DataFrame with non-residential buildings (for lookup feedback)
    """

    logger.info(f"Creating SQLite database at {output_path}...")

    # Select columns for database
    db_columns = [
        'bbl', 'buildingid', 'nta', 'nta_code', 'boro', 'housenumber', 'streetname', 'zip',
        'pluto_address', 'latitude', 'longitude',
        'yearbuilt', 'unitsres', 'numfloors', 'is_pre_1978',
        # Violation counts (total and by type)
        'violation_count', 'violations_open', 'violations_closed',
        # Violation class breakdown (A=non-hazardous, B=hazardous, C=immediately hazardous, I=orders)
        'class_a_count', 'class_b_count', 'class_c_count', 'class_i_count',
        'lead_violations', 'lead_violations_open',
        'mold_violations', 'mold_violations_open',
        'pest_violations', 'pest_violations_open',
        'heat_violations', 'heat_violations_open',
        'health_violations', 'last_violation_date',
        # Complaint data
        'complaint_count', 'complaint_health', 'last_complaint_date',
        # Rodent data
        'rodent_inspections', 'rodent_failures', 'last_rodent_inspection',
        # Bedbug data
        'bedbug_reports', 'bedbug_infested_units', 'has_bedbug_history',
        # Time-based violation counts
        'violations_1yr', 'violations_2yr', 'violations_open_1yr', 'violations_open_2yr',
        'violations_1yr_pct', 'violations_2yr_pct', 'violations_open_1yr_pct', 'violations_open_2yr_pct',
        # Weighted score (HPD violations weighted by severity)
        'weighted_score', 'weighted_score_pct',
        'weighted_score_per_unit', 'weighted_score_per_unit_pct',
        # Combined score (weighted violations + rodent failures)
        'combined_score', 'combined_score_pct',
        'combined_score_per_unit', 'combined_score_per_unit_pct',
        'adjusted_score_pct',  # Primary comparison metric (per-unit for multi-unit, raw for single)
        # Per-unit normalized metrics
        'violations_per_unit', 'violations_per_unit_pct',
        'open_violations_per_unit', 'open_violations_per_unit_pct',
        'complaints_per_unit', 'complaints_per_unit_pct',
        # Percentile scores for citywide comparison
        'violation_count_pct', 'lead_violations_pct', 'mold_violations_pct', 'pest_violations_pct',
        'heat_violations_pct', 'violations_open_pct', 'complaints_pct',
        'rodent_failures_pct', 'building_age_pct',
        # Class-based percentiles
        'class_a_count_pct', 'class_b_count_pct', 'class_c_count_pct', 'class_i_count_pct',
        # Neighborhood percentiles
        'violation_count_nhood_pct', 'lead_violations_nhood_pct', 'mold_violations_nhood_pct',
        'pest_violations_nhood_pct', 'heat_violations_nhood_pct', 'violations_open_nhood_pct',
        'complaints_nhood_pct', 'rodent_failures_nhood_pct',
        # Legacy score columns (for backward compatibility)
        'lead_violations_score', 'mold_violations_score', 'pest_violations_score',
        'heat_violations_score', 'complaints_score', 'rodent_score', 'bedbug_score',
        'building_age_score', 'building_score',
        'risk_score', 'neighborhood_index', 'risk_tier'
    ]

    # Keep only columns that exist
    available_cols = [c for c in db_columns if c in buildings_df.columns]
    db_df = buildings_df[available_cols].copy()

    # Create database
    conn = sqlite3.connect(output_path)

    # Create main buildings table
    db_df.to_sql('buildings', conn, index=False, if_exists='replace')

    # Create indexes for fast lookups
    cursor = conn.cursor()
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bbl ON buildings(bbl)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_address ON buildings(housenumber, streetname, boro)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_nta ON buildings(nta)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_zip ON buildings(zip)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk ON buildings(risk_score)')

    # Create summary stats table
    summary = pd.DataFrame([{
        'total_buildings': len(db_df),
        'buildings_with_violations': (db_df['violation_count'] > 0).sum(),
        'buildings_high_risk': (db_df['risk_tier'] == 'High Risk').sum(),
        'avg_risk_score': db_df['risk_score'].mean(),
        'max_violations': db_df['violation_count'].max(),
        'created_date': pd.Timestamp.now().isoformat()
    }])
    summary.to_sql('summary', conn, index=False, if_exists='replace')

    # Create non-residential buildings table (for address lookup feedback)
    if nonres_df is not None and len(nonres_df) > 0:
        logger.info(f"  Adding {len(nonres_df):,} non-residential buildings to database...")

        # Parse address into house number and street name
        nonres_df = nonres_df.copy()
        nonres_df['housenumber'] = nonres_df['pluto_address'].str.extract(r'^(\d+[-\d]*)', expand=False).fillna('')
        nonres_df['streetname'] = nonres_df['pluto_address'].str.replace(r'^\d+[-\d]*\s*', '', regex=True).str.strip()
        nonres_df['housenumber'] = nonres_df['housenumber'].str.upper().str.strip()
        nonres_df['streetname'] = nonres_df['streetname'].str.upper().str.strip()
        nonres_df['boro'] = nonres_df['bbl'].astype(str).str[0]

        # Keep only essential columns for the lookup
        nonres_cols = ['bbl', 'housenumber', 'streetname', 'boro', 'pluto_address', 'bldgclass']
        nonres_db = nonres_df[[c for c in nonres_cols if c in nonres_df.columns]].copy()

        nonres_db.to_sql('nonresidential_buildings', conn, index=False, if_exists='replace')

        # Create index for fast lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nonres_address ON nonresidential_buildings(housenumber, streetname, boro)')

        logger.info(f"  Non-residential buildings table created")

    conn.commit()
    conn.close()

    logger.info(f"  Database created with {len(db_df):,} residential buildings")
    logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return output_path


def build_address_lookup():
    """Main function to build address lookup database."""

    # Setup logging
    log_file = get_timestamped_log_filename("build_address_lookup")
    logger = setup_logger(__name__, log_file=log_file)

    log_step_start(logger, "Build Address Lookup Database")

    # Ensure directories exist
    ensure_dirs_exist()

    # Load PLUTO data first - this is our base of ALL residential buildings
    pluto_df = load_pluto_data(logger)
    if pluto_df is None:
        raise ValueError("PLUTO data required - cannot build lookup without it")

    # Save non-residential buildings before filtering (for address lookup feedback)
    nonres_df = pluto_df[pluto_df['unitsres'] == 0].copy()
    logger.info(f"  Non-residential buildings in PLUTO: {len(nonres_df):,}")

    # Filter to residential buildings only (at least 1 residential unit)
    pluto_df = pluto_df[pluto_df['unitsres'] >= 1].copy()
    logger.info(f"  Residential buildings in PLUTO: {len(pluto_df):,}")

    # Aggregate violations by building
    violations_file = PROCESSED_DIR / "violations_clean.csv"
    violations_agg = aggregate_violations_by_building(violations_file, logger)
    gc.collect()

    # Aggregate complaints by building
    complaints_file = PROCESSED_DIR / "complaints_clean.csv"
    complaints_agg = aggregate_complaints_by_building(complaints_file, logger)
    gc.collect()

    # Rename complaints address columns to match violations
    complaints_agg.rename(columns={
        'house_number': 'housenumber',
        'street_name': 'streetname'
    }, inplace=True)

    # Start from PLUTO as the base and LEFT JOIN violations and complaints
    logger.info("Building comprehensive lookup from PLUTO base...")

    # Parse PLUTO address into house number and street name
    # PLUTO address format is typically "123 STREET NAME"
    pluto_df['housenumber'] = pluto_df['pluto_address'].str.extract(r'^(\d+[-\d]*)', expand=False).fillna('')
    pluto_df['streetname'] = pluto_df['pluto_address'].str.replace(r'^\d+[-\d]*\s*', '', regex=True).str.strip()
    pluto_df['housenumber'] = pluto_df['housenumber'].str.upper().str.strip()
    pluto_df['streetname'] = pluto_df['streetname'].str.upper().str.strip()

    # Prepare PLUTO columns for base
    buildings_df = pluto_df[['bbl', 'housenumber', 'streetname', 'pluto_address', 'pluto_borough',
                             'yearbuilt', 'unitsres', 'numfloors', 'is_pre_1978', 'building_age',
                             'latitude', 'longitude']].copy()

    # Extract borough code from BBL (first digit)
    buildings_df['boro'] = buildings_df['bbl'].str[0]

    logger.info(f"  Starting with {len(buildings_df):,} PLUTO residential buildings")

    # LEFT JOIN violations (buildings without violations will have NaN, which becomes 0)
    # Include time-based columns and weighted score if available
    base_cols = ['bbl', 'buildingid', 'nta', 'zip',
                 'violation_count', 'violations_open', 'violations_closed',
                 'class_a_count', 'class_b_count', 'class_c_count', 'class_i_count',
                 'lead_violations', 'lead_violations_open',
                 'mold_violations', 'mold_violations_open',
                 'pest_violations', 'pest_violations_open',
                 'heat_violations', 'heat_violations_open',
                 'health_violations',
                 'violation_severity', 'last_violation_date',
                 'violations_1yr', 'violations_2yr', 'violations_open_1yr', 'violations_open_2yr',
                 'weighted_score']  # New: weighted score combining severity, status, recency
    available_cols = [c for c in base_cols if c in violations_agg.columns]
    violations_for_join = violations_agg[available_cols].copy()

    # Group by BBL to get one row per building (some have multiple addresses)
    agg_dict = {
        'buildingid': 'first',
        'nta': 'first',
        'zip': 'first',
        'violation_count': 'sum',
        'violations_open': 'sum',
        'violations_closed': 'sum',
        'class_a_count': 'sum',
        'class_b_count': 'sum',
        'class_c_count': 'sum',
        'class_i_count': 'sum',
        'lead_violations': 'sum',
        'lead_violations_open': 'sum',
        'mold_violations': 'sum',
        'mold_violations_open': 'sum',
        'pest_violations': 'sum',
        'pest_violations_open': 'sum',
        'heat_violations': 'sum',
        'heat_violations_open': 'sum',
        'health_violations': 'sum',
        'violation_severity': 'sum',
        'last_violation_date': 'max',
    }
    # Add time-based columns to aggregation if they exist
    for time_col in ['violations_1yr', 'violations_2yr', 'violations_open_1yr', 'violations_open_2yr']:
        if time_col in violations_for_join.columns:
            agg_dict[time_col] = 'sum'

    # Add weighted_score to aggregation if it exists
    if 'weighted_score' in violations_for_join.columns:
        agg_dict['weighted_score'] = 'sum'

    violations_by_bbl = violations_for_join.groupby('bbl').agg(agg_dict).reset_index()

    buildings_df = buildings_df.merge(violations_by_bbl, on='bbl', how='left')
    logger.info(f"  Buildings with violations: {buildings_df['violation_count'].notna().sum():,}")

    # LEFT JOIN complaints
    complaints_for_join = complaints_agg[['bbl', 'problem_count', 'complaint_count',
                                           'complaint_heat', 'complaint_mold', 'complaint_pests',
                                           'complaint_lead', 'complaint_water', 'complaint_health',
                                           'complaint_severity', 'last_complaint_date']].copy()
    complaints_by_bbl = complaints_for_join.groupby('bbl').agg({
        'problem_count': 'sum',
        'complaint_count': 'sum',
        'complaint_heat': 'sum',
        'complaint_mold': 'sum',
        'complaint_pests': 'sum',
        'complaint_lead': 'sum',
        'complaint_water': 'sum',
        'complaint_health': 'sum',
        'complaint_severity': 'sum',
        'last_complaint_date': 'max',
    }).reset_index()

    buildings_df = buildings_df.merge(complaints_by_bbl, on='bbl', how='left')
    logger.info(f"  Buildings with complaints: {buildings_df['complaint_count'].notna().sum():,}")

    # LEFT JOIN rodent inspections
    rodent_agg = aggregate_rodent_inspections(logger)
    if rodent_agg is not None:
        buildings_df = buildings_df.merge(rodent_agg, on='bbl', how='left')
        logger.info(f"  Buildings with rodent data: {buildings_df['rodent_inspections'].notna().sum():,}")
    gc.collect()

    # LEFT JOIN bedbug reports
    bedbug_agg = aggregate_bedbug_reports(logger)
    if bedbug_agg is not None:
        buildings_df = buildings_df.merge(bedbug_agg, on='bbl', how='left')
        logger.info(f"  Buildings with bedbug data: {buildings_df['bedbug_reports'].notna().sum():,}")
    gc.collect()

    # Assign NTA to buildings missing it using spatial join with coordinates
    logger.info("Assigning NTA to buildings via spatial join...")
    buildings_df = assign_nta_from_coordinates(buildings_df, logger)
    gc.collect()

    # Load neighborhood index for context
    index_file = PROCESSED_DIR / "housing_health_index.csv"
    if index_file.exists():
        neighborhood_df = pd.read_csv(index_file)
    else:
        neighborhood_df = None
        logger.warning("Neighborhood index not found, using default neighborhood scores")

    # Calculate risk scores
    buildings_df = calculate_building_risk_scores(buildings_df, neighborhood_df, logger)

    # Fill NaN values
    numeric_cols = buildings_df.select_dtypes(include=[np.number]).columns
    buildings_df[numeric_cols] = buildings_df[numeric_cols].fillna(0)

    # Create SQLite database (include non-residential for address lookup feedback)
    db_path = PROCESSED_DIR / "building_lookup.db"
    create_sqlite_database(buildings_df, db_path, logger, nonres_df=nonres_df)

    # Also save as CSV for reference
    csv_path = PROCESSED_DIR / "building_lookup.csv"
    buildings_df.to_csv(csv_path, index=False)
    logger.info(f"  Also saved as CSV: {csv_path}")

    log_step_complete(logger, "Build Address Lookup Database")

    # Summary
    logger.info("=" * 60)
    logger.info("BUILDING LOOKUP DATABASE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total buildings: {len(buildings_df):,}")
    logger.info(f"Buildings with violations: {(buildings_df['violation_count'] > 0).sum():,}")
    logger.info(f"Buildings with complaints: {(buildings_df['complaint_count'] > 0).sum():,}")

    logger.info(f"\nRisk tier distribution:")
    for tier, count in buildings_df['risk_tier'].value_counts().items():
        pct = count / len(buildings_df) * 100
        logger.info(f"  {tier}: {count:,} ({pct:.1f}%)")

    logger.info(f"\nTop 10 highest risk buildings:")
    top10 = buildings_df.nlargest(10, 'risk_score')
    for _, row in top10.iterrows():
        addr = f"{row.get('housenumber', '')} {row.get('streetname', '')}".strip()
        logger.info(f"  {addr} ({row.get('boro', 'Unknown')}): Score={row['risk_score']:.1f}, Violations={row.get('violation_count', 0):.0f}")

    return buildings_df


if __name__ == "__main__":
    df = build_address_lookup()

    print(f"\n✅ Building lookup database created with {len(df):,} buildings!")
    print(f"\n📊 Risk Tier Distribution:")
    for tier, count in df['risk_tier'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {tier}: {count:,} ({pct:.1f}%)")
