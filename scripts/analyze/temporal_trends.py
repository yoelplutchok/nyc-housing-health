"""
Temporal Trend Analysis: Housing Conditions Over Time

This script examines how housing conditions have changed over time.

Analyses:
1. Annual violation trends
2. Violation type trends (lead, mold, pests, heat)
3. Neighborhood improvement/worsening

Output: outputs/tables/temporal_trends.csv
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats
import warnings

from housing_health.paths import PROCESSED_DIR, RAW_DIR, HPD_VIOLATIONS_DIR, TABLES_DIR, ensure_dirs_exist
from housing_health.io_utils import write_csv
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
)

warnings.filterwarnings('ignore')


def load_violations_with_dates(logger):
    """Load violations data with date parsing."""
    
    # First try processed file
    processed_file = PROCESSED_DIR / "violations_clean.csv"
    if processed_file.exists():
        logger.info(f"Loading from {processed_file}")
        df = pd.read_csv(processed_file, parse_dates=['inspectiondate'], low_memory=False)
        return df
    
    # Fallback to raw
    raw_files = list(HPD_VIOLATIONS_DIR.glob("hpd_violations_*.csv"))
    if raw_files:
        latest = max(raw_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Loading from {latest}")
        df = pd.read_csv(latest, low_memory=False)
        df['inspectiondate'] = pd.to_datetime(df['inspectiondate'], errors='coerce')
        return df
    
    return None


def calculate_trend(years, values):
    """Calculate linear trend (slope) for time series."""
    if len(years) < 2:
        return np.nan, np.nan
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
    return slope, p_value


def temporal_trend_analysis():
    """Run full temporal trend analysis."""
    
    # Setup
    log_file = get_timestamped_log_filename("temporal_trends")
    logger = setup_logger(__name__, log_file=log_file)
    log_step_start(logger, "Temporal Trend Analysis")
    ensure_dirs_exist()
    
    # Load violations data
    logger.info("Loading violations data...")
    df = load_violations_with_dates(logger)
    
    if df is None:
        logger.error("Could not load violations data")
        return None
    
    logger.info(f"Loaded {len(df):,} violations")
    
    # Extract year
    df['year'] = df['inspectiondate'].dt.year
    df = df[df['year'].notna() & (df['year'] >= 2019) & (df['year'] <= 2025)]
    logger.info(f"Violations from 2019-2025: {len(df):,}")
    
    all_trends = []
    
    # =========================================================================
    # 1. CITYWIDE ANNUAL TRENDS
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("1. CITYWIDE ANNUAL TRENDS")
    logger.info("="*60)
    
    yearly = df.groupby('year').agg({
        'violationid': 'count',
        'is_lead': 'sum',
        'is_mold': 'sum',
        'is_pests': 'sum',
        'is_heat': 'sum',
        'is_health_related': 'sum',
    }).reset_index()
    
    yearly.columns = ['year', 'total', 'lead', 'mold', 'pests', 'heat', 'health_total']
    
    logger.info("\nAnnual Violation Counts:")
    for _, row in yearly.iterrows():
        logger.info(f"  {int(row['year'])}: {int(row['total']):,} total, {int(row['health_total']):,} health-related")
    
    # Calculate trends
    for col in ['total', 'lead', 'mold', 'pests', 'heat', 'health_total']:
        slope, p_val = calculate_trend(yearly['year'].values, yearly[col].values)
        trend_dir = "increasing" if slope > 0 else "decreasing"
        sig = "significantly " if p_val < 0.05 else ""
        
        all_trends.append({
            'category': f'Citywide {col.replace("_", " ").title()}',
            'trend_direction': trend_dir,
            'slope_per_year': slope,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'start_year': int(yearly['year'].min()),
            'end_year': int(yearly['year'].max()),
            'start_value': int(yearly.iloc[0][col]),
            'end_value': int(yearly.iloc[-1][col]),
            'pct_change': (yearly.iloc[-1][col] - yearly.iloc[0][col]) / yearly.iloc[0][col] * 100 if yearly.iloc[0][col] > 0 else np.nan,
        })
        
        if col in ['total', 'health_total']:
            pct_change = (yearly.iloc[-1][col] - yearly.iloc[0][col]) / yearly.iloc[0][col] * 100
            logger.info(f"\n  {col.title()} trend: {sig}{trend_dir}")
            logger.info(f"    Slope: {slope:,.0f} violations/year")
            logger.info(f"    Change: {pct_change:+.1f}% from {int(yearly['year'].min())} to {int(yearly['year'].max())}")
    
    # =========================================================================
    # 2. VIOLATION TYPE TRENDS
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("2. HEALTH VIOLATION TYPE TRENDS")
    logger.info("="*60)
    
    logger.info("\nYear-over-Year Health Violations:")
    logger.info(f"{'Year':<6} {'Lead':<10} {'Mold':<10} {'Pests':<10} {'Heat':<10}")
    logger.info("-" * 50)
    
    for _, row in yearly.iterrows():
        logger.info(f"{int(row['year']):<6} {int(row['lead']):<10,} {int(row['mold']):<10,} {int(row['pests']):<10,} {int(row['heat']):<10,}")
    
    # =========================================================================
    # 3. BOROUGH TRENDS
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("3. BOROUGH TRENDS")
    logger.info("="*60)
    
    if 'boro' in df.columns:
        borough_yearly = df.groupby(['year', 'boro']).agg({
            'violationid': 'count',
            'is_health_related': 'sum',
        }).reset_index()
        
        borough_yearly.columns = ['year', 'borough', 'total', 'health']
        
        for borough in df['boro'].dropna().unique():
            boro_data = borough_yearly[borough_yearly['borough'] == borough].sort_values('year')
            if len(boro_data) >= 2:
                slope, p_val = calculate_trend(boro_data['year'].values, boro_data['total'].values)
                trend_dir = "↑" if slope > 0 else "↓"
                
                pct_change = (boro_data.iloc[-1]['total'] - boro_data.iloc[0]['total']) / boro_data.iloc[0]['total'] * 100 if boro_data.iloc[0]['total'] > 0 else 0
                
                logger.info(f"  {borough}: {trend_dir} {pct_change:+.1f}% ({boro_data.iloc[0]['total']:,} → {boro_data.iloc[-1]['total']:,})")
                
                all_trends.append({
                    'category': f'{borough} Total Violations',
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                    'slope_per_year': slope,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'start_year': int(boro_data['year'].min()),
                    'end_year': int(boro_data['year'].max()),
                    'start_value': int(boro_data.iloc[0]['total']),
                    'end_value': int(boro_data.iloc[-1]['total']),
                    'pct_change': pct_change,
                })
    
    # =========================================================================
    # 4. VIOLATION CLASS TRENDS
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("4. VIOLATION CLASS TRENDS")
    logger.info("="*60)
    
    if 'class' in df.columns:
        class_yearly = df.groupby(['year', 'class']).size().reset_index(name='count')
        
        for vclass in ['A', 'B', 'C']:
            class_data = class_yearly[class_yearly['class'] == vclass].sort_values('year')
            if len(class_data) >= 2:
                slope, p_val = calculate_trend(class_data['year'].values, class_data['count'].values)
                trend_dir = "↑" if slope > 0 else "↓"
                pct_change = (class_data.iloc[-1]['count'] - class_data.iloc[0]['count']) / class_data.iloc[0]['count'] * 100 if class_data.iloc[0]['count'] > 0 else 0
                
                class_labels = {'A': 'Non-Hazardous', 'B': 'Hazardous', 'C': 'Immediately Hazardous'}
                logger.info(f"  Class {vclass} ({class_labels.get(vclass, '')}): {trend_dir} {pct_change:+.1f}%")
                
                all_trends.append({
                    'category': f'Class {vclass} ({class_labels.get(vclass, "")})',
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                    'slope_per_year': slope,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'start_year': int(class_data['year'].min()),
                    'end_year': int(class_data['year'].max()),
                    'start_value': int(class_data.iloc[0]['count']),
                    'end_value': int(class_data.iloc[-1]['count']),
                    'pct_change': pct_change,
                })
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    trends_df = pd.DataFrame(all_trends)
    output_file = TABLES_DIR / "temporal_trends.csv"
    write_csv(trends_df, output_file)
    logger.info(f"\nSaved trend results to {output_file}")
    
    # Also save the yearly summary
    yearly_output = TABLES_DIR / "annual_violations_summary.csv"
    write_csv(yearly, yearly_output)
    logger.info(f"Saved annual summary to {yearly_output}")
    
    # Summary
    log_step_complete(logger, "Temporal Trend Analysis")
    
    logger.info("\n" + "="*60)
    logger.info("KEY TEMPORAL FINDINGS")
    logger.info("="*60)
    
    sig_trends = [t for t in all_trends if t.get('significant', False)]
    increasing = [t for t in all_trends if t.get('trend_direction') == 'increasing']
    decreasing = [t for t in all_trends if t.get('trend_direction') == 'decreasing']
    
    logger.info(f"\nTotal trends analyzed: {len(all_trends)}")
    logger.info(f"Statistically significant: {len(sig_trends)}")
    logger.info(f"Increasing trends: {len(increasing)}")
    logger.info(f"Decreasing trends: {len(decreasing)}")
    
    # Overall assessment
    citywide = next((t for t in all_trends if t['category'] == 'Citywide Total'), None)
    if citywide:
        logger.info(f"\n📈 Overall: Violations {citywide['trend_direction']} by {citywide['pct_change']:.1f}%")
    
    return trends_df


if __name__ == "__main__":
    results = temporal_trend_analysis()
    
    print("\n" + "="*70)
    print("✅ TEMPORAL TREND ANALYSIS COMPLETE")
    print("="*70)
    
    if results is not None:
        sig = results[results['significant'] == True]
        print(f"\nSignificant trends: {len(sig)} / {len(results)}")
        
        print("\nKey trends:")
        for _, row in results[results['category'].str.contains('Citywide')].iterrows():
            print(f"  • {row['category']}: {row['trend_direction']} {row['pct_change']:+.1f}%")

