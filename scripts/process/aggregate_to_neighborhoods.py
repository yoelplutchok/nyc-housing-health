"""
Aggregate violations and complaints to NTA (Neighborhood Tabulation Area) level.

This script:
1. Loads cleaned violations and complaints data
2. Aggregates counts and rates by NTA
3. Creates severity-weighted indices
4. Joins with NTA boundaries for geographic data
5. Saves neighborhood-level summary

Input: 
  - data/processed/violations_clean.csv
  - data/processed/complaints_clean.csv
  - data/geo/nta_boundaries_*.geojson
  - data/raw/pluto_*.csv (for residential unit counts)
  
Output: data/processed/nta_aggregated.csv
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np
import json
import gc

from housing_health.paths import PROCESSED_DIR, GEO_DIR, ensure_dirs_exist
from housing_health.io_utils import write_csv
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
)

# Define only the columns needed for aggregation (memory optimization)
VIOLATIONS_COLS = ['violationid', 'nta', 'class', 'is_lead', 'is_mold', 'is_pests', 
                   'is_heat', 'is_health_related', 'severity_weight']

COMPLAINTS_COLS = ['complaint_id', 'problem_id', 'nta', 'is_heat', 'is_mold', 
                   'is_pests', 'is_lead', 'is_water', 'is_health_related', 'severity_weight']

# Chunk size for processing large files
CHUNK_SIZE = 500_000


def load_nta_boundaries():
    """Load NTA boundaries GeoJSON to get NTA names and boroughs."""
    geo_files = list(GEO_DIR.glob("nta_boundaries_*.geojson"))
    if not geo_files:
        return None
    
    geo_file = max(geo_files, key=lambda x: x.stat().st_mtime)
    
    with open(geo_file, 'r') as f:
        geojson = json.load(f)
    
    # Extract NTA info from features
    # Note: violations use 'ntaname' as the NTA identifier, so we join on that
    nta_info = []
    for feature in geojson['features']:
        props = feature['properties']
        nta_info.append({
            'nta': props.get('ntaname'),  # Match to violations NTA field (which is the name)
            'nta_code': props.get('nta2020'),
            'borough': props.get('boroname'),
        })
    
    return pd.DataFrame(nta_info)


def load_pluto_for_units(bbl_nta_mapping: pd.DataFrame):
    """
    Load PLUTO data to get residential unit counts by NTA.
    Uses BBL-to-NTA mapping from violations/complaints since PLUTO doesn't have NTA.
    
    Args:
        bbl_nta_mapping: DataFrame with 'bbl' and 'nta' columns from violations data
    """
    pluto_files = list(Path("data/raw").glob("pluto_*.csv"))
    if not pluto_files:
        return None
    
    pluto_file = max(pluto_files, key=lambda x: x.stat().st_mtime)
    
    # Read only needed columns
    cols_to_read = ['bbl', 'unitsres', 'yearbuilt']
    df = pd.read_csv(pluto_file, usecols=lambda c: c.upper() in [col.upper() for col in cols_to_read], low_memory=False)
    df.columns = df.columns.str.lower()
    
    # Standardize BBL
    df['bbl'] = df['bbl'].astype(str).str.strip()
    
    # Convert units to numeric
    df['unitsres'] = pd.to_numeric(df['unitsres'], errors='coerce').fillna(0)
    df['yearbuilt'] = pd.to_numeric(df['yearbuilt'], errors='coerce')
    
    # Join with BBL-NTA mapping
    bbl_nta = bbl_nta_mapping[['bbl', 'nta']].drop_duplicates()
    bbl_nta['bbl'] = bbl_nta['bbl'].astype(str).str.strip()
    
    df = df.merge(bbl_nta, on='bbl', how='inner')
    
    # Aggregate by NTA
    nta_units = df.groupby('nta').agg({
        'unitsres': 'sum',
        'bbl': 'count',  # building count
        'yearbuilt': lambda x: (pd.to_numeric(x, errors='coerce') < 1978).sum()  # pre-1978 buildings
    }).reset_index()
    
    nta_units.columns = ['nta', 'total_units', 'building_count', 'pre_1978_buildings']
    
    return nta_units


def aggregate_violations_chunked(violations_file: Path, logger) -> pd.DataFrame:
    """Aggregate violations to NTA level using chunked processing."""
    
    logger.info("Aggregating violations by NTA (chunked)...")
    
    # Initialize accumulators
    agg_accum = None
    class_accum = None
    total_rows = 0
    valid_rows = 0
    
    # Process in chunks
    chunk_iter = pd.read_csv(
        violations_file, 
        usecols=VIOLATIONS_COLS,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )
    
    for i, chunk in enumerate(chunk_iter):
        total_rows += len(chunk)
        
        # Filter to records with valid NTA
        chunk_valid = chunk[chunk['nta'].notna() & (chunk['nta'] != '')]
        valid_rows += len(chunk_valid)
        
        if len(chunk_valid) == 0:
            continue
        
        # Aggregate this chunk
        chunk_agg = chunk_valid.groupby('nta').agg({
            'violationid': 'count',
            'is_lead': 'sum',
            'is_mold': 'sum',
            'is_pests': 'sum',
            'is_heat': 'sum',
            'is_health_related': 'sum',
            'severity_weight': 'sum',
        }).reset_index()
        
        # Class counts for this chunk
        chunk_class = chunk_valid.groupby(['nta', 'class']).size().unstack(fill_value=0).reset_index()
        
        # Merge with accumulator
        if agg_accum is None:
            agg_accum = chunk_agg
            class_accum = chunk_class
        else:
            agg_accum = pd.concat([agg_accum, chunk_agg]).groupby('nta').sum().reset_index()
            class_accum = pd.concat([class_accum, chunk_class]).groupby('nta').sum().reset_index()
        
        logger.info(f"  Processed chunk {i+1}: {total_rows:,} rows so far...")
    
    logger.info(f"  Records with valid NTA: {valid_rows:,} / {total_rows:,}")
    
    # Rename columns
    agg_accum.columns = [
        'nta',
        'violation_count',
        'violation_lead',
        'violation_mold',
        'violation_pests',
        'violation_heat',
        'violation_health_total',
        'violation_severity_score',
    ]
    
    # Rename class columns
    class_accum.columns = ['nta'] + [f'violation_class_{c}' for c in class_accum.columns[1:]]
    
    # Merge
    agg = agg_accum.merge(class_accum, on='nta', how='left')
    
    logger.info(f"  Aggregated to {len(agg)} NTAs")
    
    return agg


def aggregate_complaints_chunked(complaints_file: Path, logger) -> pd.DataFrame:
    """Aggregate complaints to NTA level using chunked processing."""
    
    logger.info("Aggregating complaints by NTA (chunked)...")
    
    # Initialize accumulators
    agg_accum = None
    complaint_ids_per_nta = {}  # Track unique complaint IDs per NTA
    total_rows = 0
    valid_rows = 0
    
    # Process in chunks
    chunk_iter = pd.read_csv(
        complaints_file, 
        usecols=COMPLAINTS_COLS,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )
    
    for i, chunk in enumerate(chunk_iter):
        total_rows += len(chunk)
        
        # Filter to records with valid NTA
        chunk_valid = chunk[chunk['nta'].notna() & (chunk['nta'] != '')]
        valid_rows += len(chunk_valid)
        
        if len(chunk_valid) == 0:
            continue
        
        # Track unique complaint IDs per NTA (for accurate nunique)
        for nta, group in chunk_valid.groupby('nta'):
            if nta not in complaint_ids_per_nta:
                complaint_ids_per_nta[nta] = set()
            complaint_ids_per_nta[nta].update(group['complaint_id'].dropna().unique())
        
        # Aggregate this chunk (problem_id count, not nunique for complaint_id yet)
        chunk_agg = chunk_valid.groupby('nta').agg({
            'problem_id': 'count',
            'is_heat': 'sum',
            'is_mold': 'sum',
            'is_pests': 'sum',
            'is_lead': 'sum',
            'is_water': 'sum',
            'is_health_related': 'sum',
            'severity_weight': 'sum',
        }).reset_index()
        
        # Merge with accumulator
        if agg_accum is None:
            agg_accum = chunk_agg
        else:
            agg_accum = pd.concat([agg_accum, chunk_agg]).groupby('nta').sum().reset_index()
        
        logger.info(f"  Processed chunk {i+1}: {total_rows:,} rows so far...")
    
    logger.info(f"  Records with valid NTA: {valid_rows:,} / {total_rows:,}")
    
    # Add unique complaint counts
    complaint_counts = pd.DataFrame([
        {'nta': nta, 'complaint_count': len(ids)} 
        for nta, ids in complaint_ids_per_nta.items()
    ])
    
    # Merge complaint counts with aggregated data
    agg = complaint_counts.merge(agg_accum, on='nta', how='outer')
    
    # Rename columns
    agg.columns = [
        'nta',
        'complaint_count',
        'problem_count',
        'complaint_heat',
        'complaint_mold',
        'complaint_pests',
        'complaint_lead',
        'complaint_water',
        'complaint_health_total',
        'complaint_severity_score',
    ]
    
    logger.info(f"  Aggregated to {len(agg)} NTAs")
    
    return agg


def aggregate_to_neighborhoods():
    """Main function to aggregate data to neighborhood level."""
    
    # Setup logging
    log_file = get_timestamped_log_filename("aggregate_neighborhoods")
    logger = setup_logger(__name__, log_file=log_file)
    
    log_step_start(logger, "Aggregate to NTA Neighborhoods")
    
    # Ensure directories exist
    ensure_dirs_exist()
    
    # Aggregate violations (chunked - no full load)
    violations_file = PROCESSED_DIR / "violations_clean.csv"
    logger.info(f"Processing violations from {violations_file}...")
    violations_agg = aggregate_violations_chunked(violations_file, logger)
    gc.collect()  # Free memory
    
    # Aggregate complaints (chunked - no full load)
    complaints_file = PROCESSED_DIR / "complaints_clean.csv"
    logger.info(f"Processing complaints from {complaints_file}...")
    complaints_agg = aggregate_complaints_chunked(complaints_file, logger)
    gc.collect()  # Free memory
    
    # Merge violations and complaints
    logger.info("Merging violations and complaints...")
    nta_df = violations_agg.merge(complaints_agg, on='nta', how='outer')
    logger.info(f"  Combined NTAs: {len(nta_df)}")
    
    # Load and merge NTA info
    logger.info("Loading NTA boundaries for names...")
    nta_info = load_nta_boundaries()
    if nta_info is not None:
        nta_df = nta_df.merge(nta_info, on='nta', how='left')
        logger.info(f"  Added NTA names and boroughs")
    
    # Get building counts from violations data (unique BBLs per NTA)
    # This serves as a proxy for unit counts
    logger.info("Calculating building counts per NTA...")
    violations_file = PROCESSED_DIR / "violations_clean.csv"
    bbl_nta = pd.read_csv(violations_file, usecols=['bbl', 'nta'])
    bbl_nta = bbl_nta.dropna()
    bbl_nta['bbl'] = bbl_nta['bbl'].astype(str).str.strip()
    
    building_counts = bbl_nta.groupby('nta').agg({
        'bbl': 'nunique'
    }).reset_index()
    building_counts.columns = ['nta', 'building_count']
    
    nta_df = nta_df.merge(building_counts, on='nta', how='left')
    logger.info(f"  Added building counts for {len(building_counts)} NTAs")
    
    # Fill NaN values
    numeric_cols = nta_df.select_dtypes(include=[np.number]).columns
    nta_df[numeric_cols] = nta_df[numeric_cols].fillna(0)
    
    # Calculate rates per building (since we have building counts)
    logger.info("Calculating rates per building...")
    if 'building_count' in nta_df.columns:
        nta_df['bldg_safe'] = nta_df['building_count'].replace(0, np.nan)
        
        rate_cols = [
            ('violation_count', 'violation_per_bldg'),
            ('violation_health_total', 'violation_health_per_bldg'),
            ('complaint_count', 'complaint_per_bldg'),
            ('complaint_health_total', 'complaint_health_per_bldg'),
        ]
        
        for count_col, rate_col in rate_cols:
            if count_col in nta_df.columns:
                nta_df[rate_col] = nta_df[count_col] / nta_df['bldg_safe']
        
        nta_df.drop('bldg_safe', axis=1, inplace=True)
    
    # Calculate percentiles for ranking
    logger.info("Calculating percentile rankings...")
    rank_cols = ['violation_per_bldg', 'complaint_per_bldg', 'violation_health_per_bldg', 'complaint_health_per_bldg']
    for col in rank_cols:
        if col in nta_df.columns:
            nta_df[f'{col}_pctl'] = nta_df[col].rank(pct=True) * 100
    
    # Sort by health violations
    nta_df = nta_df.sort_values('violation_health_total', ascending=False)
    
    # Save
    output_file = PROCESSED_DIR / "nta_aggregated.csv"
    logger.info(f"Saving to {output_file}...")
    write_csv(nta_df, output_file)
    
    log_step_complete(logger, "Aggregate to NTA Neighborhoods")
    
    # Summary
    logger.info("=" * 60)
    logger.info("AGGREGATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total NTAs: {len(nta_df)}")
    logger.info(f"Output file: {output_file}")
    
    if 'violation_count' in nta_df.columns:
        logger.info(f"\nViolations:")
        logger.info(f"  Total: {nta_df['violation_count'].sum():,.0f}")
        logger.info(f"  Health-related: {nta_df['violation_health_total'].sum():,.0f}")
        logger.info(f"  Avg per NTA: {nta_df['violation_count'].mean():,.0f}")
    
    if 'complaint_count' in nta_df.columns:
        logger.info(f"\nComplaints:")
        logger.info(f"  Total: {nta_df['complaint_count'].sum():,.0f}")
        logger.info(f"  Health-related: {nta_df['complaint_health_total'].sum():,.0f}")
        logger.info(f"  Avg per NTA: {nta_df['complaint_count'].mean():,.0f}")
    
    # Top 10 neighborhoods by health issues
    if 'nta' in nta_df.columns and 'violation_health_per_bldg' in nta_df.columns:
        logger.info("\nTop 10 NTAs by Health Violations per Building:")
        top10 = nta_df.nlargest(10, 'violation_health_per_bldg')[['nta', 'borough', 'violation_health_per_bldg', 'building_count']]
        for _, row in top10.iterrows():
            logger.info(f"  {row['nta']} ({row['borough']}): {row['violation_health_per_bldg']:.1f} per bldg")
    
    return nta_df


if __name__ == "__main__":
    df = aggregate_to_neighborhoods()
    print(f"\n✅ Aggregated to {len(df)} neighborhoods!")

