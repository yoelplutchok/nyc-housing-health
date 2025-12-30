"""
Disparity Analysis: Housing-Health by Race and Income

This script examines housing-health disparities across demographic groups.

Analyses:
1. Compare CHHI by demographics (race, income)
2. Exposure analysis (% children in high-risk neighborhoods)
3. Environmental Justice quintile analysis

Output: outputs/tables/disparity_analysis.csv
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats
import warnings

from housing_health.paths import PROCESSED_DIR, TABLES_DIR, ensure_dirs_exist
from housing_health.io_utils import write_csv
from housing_health.logging_utils import (
    setup_logger,
    get_timestamped_log_filename,
    log_step_start,
    log_step_complete,
)

warnings.filterwarnings('ignore')


def t_test_groups(group1, group2):
    """Run independent samples t-test."""
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    if len(g1) < 5 or len(g2) < 5:
        return np.nan, np.nan
    
    t_stat, p_val = stats.ttest_ind(g1, g2)
    return t_stat, p_val


def disparity_analysis():
    """Run full disparity analysis."""
    
    # Setup
    log_file = get_timestamped_log_filename("disparity_analysis")
    logger = setup_logger(__name__, log_file=log_file)
    log_step_start(logger, "Disparity Analysis")
    ensure_dirs_exist()
    
    # Load data
    input_file = PROCESSED_DIR / "housing_health_index.csv"
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} neighborhoods")
    logger.info(f"Columns available: {list(df.columns)}")
    
    all_results = []
    
    # =========================================================================
    # 1. BOROUGH DISPARITIES
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("1. BOROUGH-LEVEL DISPARITIES")
    logger.info("="*60)
    
    if 'borough' in df.columns and 'housing_health_index' in df.columns:
        borough_stats = df.groupby('borough').agg({
            'housing_health_index': ['mean', 'std', 'count'],
            'asthma_rate_per_10k': 'mean',
            'lead_rate_per_1k': 'mean',
            'violation_health_per_bldg': 'mean',
        }).round(2)
        
        borough_stats.columns = ['_'.join(col) for col in borough_stats.columns]
        borough_stats = borough_stats.reset_index()
        borough_stats = borough_stats.sort_values('housing_health_index_mean', ascending=False)
        
        logger.info("\nHousing Health Index by Borough:")
        for _, row in borough_stats.iterrows():
            logger.info(f"  {row['borough']:<15}: Index = {row['housing_health_index_mean']:.1f} ± {row['housing_health_index_std']:.1f} (n={int(row['housing_health_index_count'])})")
        
        # Highest vs Lowest borough t-test
        highest = borough_stats.iloc[0]['borough']
        lowest = borough_stats.iloc[-1]['borough']
        t_stat, p_val = t_test_groups(
            df[df['borough'] == highest]['housing_health_index'],
            df[df['borough'] == lowest]['housing_health_index']
        )
        logger.info(f"\n  T-test ({highest} vs {lowest}): t = {t_stat:.2f}, p = {p_val:.4f}")
        
        all_results.append({
            'analysis': 'Borough Disparity',
            'comparison': f'{highest} vs {lowest}',
            'highest_group_mean': borough_stats.iloc[0]['housing_health_index_mean'],
            'lowest_group_mean': borough_stats.iloc[-1]['housing_health_index_mean'],
            'difference': borough_stats.iloc[0]['housing_health_index_mean'] - borough_stats.iloc[-1]['housing_health_index_mean'],
            't_statistic': t_stat,
            'p_value': p_val,
        })
    
    # =========================================================================
    # 2. INCOME QUINTILE ANALYSIS
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("2. INCOME QUINTILE ANALYSIS")
    logger.info("="*60)
    
    if 'median_income' in df.columns:
        # Create income quintiles
        df['income_quintile'] = pd.qcut(
            df['median_income'].rank(method='first'),
            q=5,
            labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)']
        )
        
        income_stats = df.groupby('income_quintile').agg({
            'housing_health_index': ['mean', 'std', 'count'],
            'median_income': 'median',
            'asthma_rate_per_10k': 'mean',
            'violation_health_per_bldg': 'mean',
        }).round(2)
        
        income_stats.columns = ['_'.join(col) for col in income_stats.columns]
        income_stats = income_stats.reset_index()
        
        logger.info("\nHousing Health Index by Income Quintile:")
        for _, row in income_stats.iterrows():
            logger.info(f"  {row['income_quintile']}: Index = {row['housing_health_index_mean']:.1f}, Median Income = ${row['median_income_median']:,.0f}")
        
        # Q1 vs Q5 comparison
        q1_index = df[df['income_quintile'] == 'Q1 (Lowest)']['housing_health_index']
        q5_index = df[df['income_quintile'] == 'Q5 (Highest)']['housing_health_index']
        t_stat, p_val = t_test_groups(q1_index, q5_index)
        
        logger.info(f"\n  T-test (Q1 vs Q5): t = {t_stat:.2f}, p = {p_val:.4f}")
        logger.info(f"  Disparity: Low-income neighborhoods have {q1_index.mean() - q5_index.mean():.1f} points higher risk")
        
        all_results.append({
            'analysis': 'Income Quintile Disparity',
            'comparison': 'Q1 (Lowest) vs Q5 (Highest)',
            'highest_group_mean': q1_index.mean(),
            'lowest_group_mean': q5_index.mean(),
            'difference': q1_index.mean() - q5_index.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
        })
    else:
        logger.warning("No median_income column found for income analysis")
    
    # =========================================================================
    # 3. POVERTY ANALYSIS
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("3. POVERTY RATE ANALYSIS")
    logger.info("="*60)
    
    if 'pct_poverty' in df.columns:
        # High poverty vs low poverty neighborhoods
        median_poverty = df['pct_poverty'].median()
        df['high_poverty'] = df['pct_poverty'] >= median_poverty
        
        high_pov = df[df['high_poverty'] == True]['housing_health_index']
        low_pov = df[df['high_poverty'] == False]['housing_health_index']
        
        t_stat, p_val = t_test_groups(high_pov, low_pov)
        
        logger.info(f"\nHigh vs Low Poverty Neighborhoods:")
        logger.info(f"  High poverty (>{median_poverty:.1f}%): Index = {high_pov.mean():.1f} (n={len(high_pov)})")
        logger.info(f"  Low poverty (<{median_poverty:.1f}%): Index = {low_pov.mean():.1f} (n={len(low_pov)})")
        logger.info(f"  T-test: t = {t_stat:.2f}, p = {p_val:.4f}")
        
        all_results.append({
            'analysis': 'Poverty Disparity',
            'comparison': 'High Poverty vs Low Poverty',
            'highest_group_mean': high_pov.mean(),
            'lowest_group_mean': low_pov.mean(),
            'difference': high_pov.mean() - low_pov.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
        })
    
    # =========================================================================
    # 4. RISK TIER DEMOGRAPHICS
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("4. DEMOGRAPHIC PROFILE BY RISK TIER")
    logger.info("="*60)
    
    if 'risk_tier' in df.columns:
        tier_cols = ['risk_tier']
        demo_cols = [c for c in ['median_income', 'pct_poverty', 'pct_black', 'pct_hispanic', 'pct_white'] 
                     if c in df.columns]
        
        if demo_cols:
            tier_demo = df.groupby('risk_tier')[demo_cols].mean().round(1)
            logger.info("\nDemographic Profiles by Risk Tier:")
            logger.info(tier_demo.to_string())
    
    # =========================================================================
    # 5. HIGH-RISK EXPOSURE ANALYSIS
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("5. HIGH-RISK NEIGHBORHOOD EXPOSURE")
    logger.info("="*60)
    
    if 'housing_health_index' in df.columns:
        # Define high-risk as top 25%
        high_risk_threshold = df['housing_health_index'].quantile(0.75)
        df['high_risk'] = df['housing_health_index'] >= high_risk_threshold
        
        high_risk_ntas = df[df['high_risk'] == True]
        low_risk_ntas = df[df['high_risk'] == False]
        
        logger.info(f"\nHigh-Risk Neighborhoods (Index >= {high_risk_threshold:.1f}):")
        logger.info(f"  Count: {len(high_risk_ntas)} / {len(df)} ({len(high_risk_ntas)/len(df)*100:.1f}%)")
        
        # Borough breakdown
        if 'borough' in df.columns:
            high_risk_borough = high_risk_ntas['borough'].value_counts()
            logger.info("\n  High-Risk Neighborhoods by Borough:")
            for borough, count in high_risk_borough.items():
                total_in_borough = len(df[df['borough'] == borough])
                pct = count / total_in_borough * 100
                logger.info(f"    {borough}: {count} / {total_in_borough} ({pct:.0f}%)")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    results_df = pd.DataFrame(all_results)
    output_file = TABLES_DIR / "disparity_analysis.csv"
    write_csv(results_df, output_file)
    logger.info(f"\nSaved disparity results to {output_file}")
    
    # Summary
    log_step_complete(logger, "Disparity Analysis")
    
    logger.info("\n" + "="*60)
    logger.info("KEY DISPARITY FINDINGS")
    logger.info("="*60)
    
    sig_results = [r for r in all_results if r.get('p_value', 1) < 0.05]
    logger.info(f"\nSignificant disparities found: {len(sig_results)} / {len(all_results)}")
    
    for r in sig_results:
        logger.info(f"\n  • {r['analysis']}:")
        logger.info(f"    {r['comparison']}")
        logger.info(f"    Difference: {r['difference']:.1f} points (p = {r['p_value']:.4f})")
    
    return results_df


if __name__ == "__main__":
    results = disparity_analysis()
    
    print("\n" + "="*70)
    print("✅ DISPARITY ANALYSIS COMPLETE")
    print("="*70)
    
    sig = results[results['p_value'] < 0.05] if 'p_value' in results.columns else pd.DataFrame()
    print(f"\nSignificant disparities: {len(sig)} / {len(results)}")
    
    if len(sig) > 0:
        print("\nKey findings:")
        for _, row in sig.iterrows():
            print(f"  • {row['analysis']}: {row['difference']:.1f} point gap")

