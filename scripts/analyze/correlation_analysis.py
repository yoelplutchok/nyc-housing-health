"""
Correlation Analysis: Housing Conditions vs Health Outcomes

This script tests the statistical relationship between housing conditions
and pediatric health outcomes.

Analyses:
1. Pearson Correlation Matrix
2. Regression Analysis (OLS)
3. Key statistical findings

Output: outputs/tables/correlation_analysis.csv
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
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


def calculate_correlation(x, y, method='pearson'):
    """Calculate correlation with p-value, handling missing data."""
    # Remove NaN pairs
    mask = ~(pd.isna(x) | pd.isna(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 10:
        return np.nan, np.nan, 0
    
    if method == 'pearson':
        r, p = pearsonr(x_clean, y_clean)
    else:
        r, p = spearmanr(x_clean, y_clean)
    
    return r, p, len(x_clean)


def run_regression(df, y_col, x_cols, logger):
    """Run OLS regression and return summary."""
    try:
        import statsmodels.api as sm
        
        # Prepare data
        data = df[[y_col] + x_cols].dropna()
        if len(data) < 20:
            return None
        
        X = data[x_cols]
        X = sm.add_constant(X)
        y = data[y_col]
        
        model = sm.OLS(y, X).fit()
        
        return {
            'dependent': y_col,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'n_obs': int(model.nobs),
            'coefficients': model.params.to_dict(),
            'pvalues': model.pvalues.to_dict(),
        }
    except ImportError:
        logger.warning("statsmodels not installed, skipping regression")
        return None
    except Exception as e:
        logger.warning(f"Regression failed: {e}")
        return None


def correlation_analysis():
    """Run full correlation analysis."""
    
    # Setup
    log_file = get_timestamped_log_filename("correlation_analysis")
    logger = setup_logger(__name__, log_file=log_file)
    log_step_start(logger, "Correlation Analysis")
    ensure_dirs_exist()
    
    # Load data
    input_file = PROCESSED_DIR / "housing_health_index.csv"
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} neighborhoods")
    
    # Define variable pairs for correlation
    correlation_pairs = [
        # Housing vs Asthma
        ('violation_health_per_bldg', 'asthma_rate_per_10k', 'Health Violations vs Asthma'),
        ('violation_per_bldg', 'asthma_rate_per_10k', 'All Violations vs Asthma'),
        ('complaint_health_per_bldg', 'asthma_rate_per_10k', 'Health Complaints vs Asthma'),
        
        # Specific violations vs Asthma
        ('violation_mold', 'asthma_rate_per_10k', 'Mold Violations vs Asthma'),
        ('violation_pests', 'asthma_rate_per_10k', 'Pest Violations vs Asthma'),
        ('violation_heat', 'asthma_rate_per_10k', 'Heat Violations vs Asthma'),
        
        # Housing vs Lead
        ('violation_lead', 'lead_rate_per_1k', 'Lead Violations vs Lead Poisoning'),
        ('pct_pre_1978', 'lead_rate_per_1k', 'Pre-1978 Buildings vs Lead Poisoning'),
        
        # Building age vs Health
        ('pct_pre_1978', 'asthma_rate_per_10k', 'Pre-1978 Buildings vs Asthma'),
        
        # Demographics vs Health (if available)
        ('median_income', 'asthma_rate_per_10k', 'Income vs Asthma'),
        ('median_income', 'housing_health_index', 'Income vs Housing Health Index'),
        
        # Composite index vs health
        ('housing_health_index', 'asthma_rate_per_10k', 'Housing Index vs Asthma'),
    ]
    
    # Calculate correlations
    logger.info("\n" + "="*60)
    logger.info("PEARSON CORRELATION ANALYSIS")
    logger.info("="*60)
    
    results = []
    for x_col, y_col, description in correlation_pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        
        r, p, n = calculate_correlation(df[x_col], df[y_col])
        
        if not pd.isna(r):
            # Interpret strength
            if abs(r) >= 0.7:
                strength = "Strong"
            elif abs(r) >= 0.4:
                strength = "Moderate"
            elif abs(r) >= 0.2:
                strength = "Weak"
            else:
                strength = "Negligible"
            
            direction = "positive" if r > 0 else "negative"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            
            results.append({
                'description': description,
                'x_variable': x_col,
                'y_variable': y_col,
                'correlation_r': round(r, 4),
                'p_value': p,
                'n_observations': n,
                'strength': strength,
                'direction': direction,
                'significant': p < 0.05,
            })
            
            logger.info(f"\n{description}:")
            logger.info(f"  r = {r:.3f}{sig}, p = {p:.4f}, n = {n}")
            logger.info(f"  Interpretation: {strength} {direction} correlation")
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(results)
    
    # Save correlation results
    corr_output = TABLES_DIR / "correlation_analysis.csv"
    write_csv(corr_df, corr_output)
    logger.info(f"\nSaved correlation results to {corr_output}")
    
    # Run regression analyses
    logger.info("\n" + "="*60)
    logger.info("REGRESSION ANALYSIS")
    logger.info("="*60)
    
    regression_results = []
    
    # Model 1: Asthma ~ housing factors
    asthma_predictors = ['violation_health_per_bldg', 'pct_pre_1978']
    asthma_available = [c for c in asthma_predictors if c in df.columns]
    
    if 'asthma_rate_per_10k' in df.columns and len(asthma_available) > 0:
        result = run_regression(df, 'asthma_rate_per_10k', asthma_available, logger)
        if result:
            regression_results.append(result)
            logger.info(f"\nAsthma Model:")
            logger.info(f"  R² = {result['r_squared']:.3f}")
            logger.info(f"  Adjusted R² = {result['adj_r_squared']:.3f}")
            logger.info(f"  F-statistic = {result['f_statistic']:.2f}, p = {result['f_pvalue']:.4f}")
            logger.info(f"  Observations: {result['n_obs']}")
            for var, coef in result['coefficients'].items():
                if var != 'const':
                    pval = result['pvalues'].get(var, 1)
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    logger.info(f"  {var}: β = {coef:.3f}{sig}")
    
    # Model 2: Lead ~ housing factors
    lead_predictors = ['violation_lead', 'pct_pre_1978']
    lead_available = [c for c in lead_predictors if c in df.columns]
    
    if 'lead_rate_per_1k' in df.columns and len(lead_available) > 0:
        result = run_regression(df, 'lead_rate_per_1k', lead_available, logger)
        if result:
            regression_results.append(result)
            logger.info(f"\nLead Poisoning Model:")
            logger.info(f"  R² = {result['r_squared']:.3f}")
    
    # Summary statistics
    log_step_complete(logger, "Correlation Analysis")
    
    logger.info("\n" + "="*60)
    logger.info("KEY FINDINGS SUMMARY")
    logger.info("="*60)
    
    # Find strongest correlations
    if len(corr_df) > 0:
        sig_corrs = corr_df[corr_df['significant'] == True].sort_values('correlation_r', key=abs, ascending=False)
        
        logger.info(f"\nSignificant correlations found: {len(sig_corrs)} / {len(corr_df)}")
        
        if len(sig_corrs) > 0:
            logger.info("\nTop 5 strongest significant correlations:")
            for _, row in sig_corrs.head(5).iterrows():
                logger.info(f"  • {row['description']}: r = {row['correlation_r']:.3f} ({row['strength']} {row['direction']})")
    
    return corr_df, regression_results


if __name__ == "__main__":
    corr_df, reg_results = correlation_analysis()
    
    print("\n" + "="*70)
    print("✅ CORRELATION ANALYSIS COMPLETE")
    print("="*70)
    
    sig = corr_df[corr_df['significant'] == True]
    print(f"\nSignificant correlations: {len(sig)} / {len(corr_df)}")
    
    if len(sig) > 0:
        print("\nTop findings:")
        for _, row in sig.sort_values('correlation_r', key=abs, ascending=False).head(5).iterrows():
            print(f"  • {row['description']}: r = {row['correlation_r']:.3f}")

