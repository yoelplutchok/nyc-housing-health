#!/usr/bin/env python3
"""
Create static publication-quality figures for the NYC Housing Health project.

This script generates:
1. Correlation scatter plots (violations vs asthma, violations vs lead)
2. Disparity bar charts (by income quintile, by borough)
3. Top 10 best/worst neighborhoods table
4. Violation type breakdown
5. NYC choropleth map (static PNG version)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from housing_health.logging_utils import setup_logger, get_timestamped_log_filename

# Paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette
BOROUGH_COLORS = {
    'Bronx': '#e74c3c',      # Red
    'Brooklyn': '#3498db',    # Blue
    'Manhattan': '#9b59b6',   # Purple
    'Queens': '#2ecc71',      # Green
    'Staten Island': '#f39c12' # Orange
}

RISK_COLORS = {
    'Very High Risk': '#7f0000',
    'High Risk': '#d32f2f',
    'Elevated Risk': '#ff9800',
    'Moderate Risk': '#fdd835',
    'Low Risk': '#4caf50',
}


def load_data(logger):
    """Load the housing health index data."""
    index_file = PROCESSED_DIR / "housing_health_index.csv"
    
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    df = pd.read_csv(index_file)
    logger.info(f"Loaded {len(df)} neighborhoods")
    
    return df


def create_correlation_scatter(df, logger):
    """Create correlation scatter plots."""
    logger.info("Creating correlation scatter plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Violations vs Asthma
    ax1 = axes[0]
    
    # Filter out NaN values
    mask1 = df['violation_per_bldg'].notna() & df['asthma_rate_per_10k'].notna()
    x1 = df.loc[mask1, 'violation_per_bldg']
    y1 = df.loc[mask1, 'asthma_rate_per_10k']
    boroughs1 = df.loc[mask1, 'borough']
    
    # Scatter plot colored by borough
    for borough in BOROUGH_COLORS:
        mask = boroughs1 == borough
        ax1.scatter(x1[mask], y1[mask], c=BOROUGH_COLORS[borough], 
                   label=borough, alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
    
    # Add regression line
    slope1, intercept1, r1, p1, se1 = stats.linregress(x1, y1)
    x_line = np.linspace(x1.min(), x1.max(), 100)
    ax1.plot(x_line, intercept1 + slope1 * x_line, 'k--', lw=2, alpha=0.8)
    
    # Add correlation annotation
    ax1.annotate(f'r = {r1:.3f}\np < 0.001', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Violations per Building', fontsize=12)
    ax1.set_ylabel('Child Asthma ED Visits per 10,000', fontsize=12)
    ax1.set_title('Housing Violations Predict Childhood Asthma', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    
    # Plot 2: Violations vs Lead
    ax2 = axes[1]
    
    mask2 = df['violation_per_bldg'].notna() & df['lead_rate_per_1k'].notna()
    x2 = df.loc[mask2, 'violation_per_bldg']
    y2 = df.loc[mask2, 'lead_rate_per_1k']
    boroughs2 = df.loc[mask2, 'borough']
    
    for borough in BOROUGH_COLORS:
        mask = boroughs2 == borough
        ax2.scatter(x2[mask], y2[mask], c=BOROUGH_COLORS[borough], 
                   label=borough, alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
    
    # Add regression line
    slope2, intercept2, r2, p2, se2 = stats.linregress(x2, y2)
    x_line2 = np.linspace(x2.min(), x2.max(), 100)
    ax2.plot(x_line2, intercept2 + slope2 * x_line2, 'k--', lw=2, alpha=0.8)
    
    ax2.annotate(f'r = {r2:.3f}\np < 0.001' if p2 < 0.001 else f'r = {r2:.3f}\np = {p2:.3f}', 
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Violations per Building', fontsize=12)
    ax2.set_ylabel('Child Lead Poisoning per 1,000', fontsize=12)
    ax2.set_title('Housing Violations Predict Lead Poisoning', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / 'correlation_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'correlation_scatter.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  Saved: {output_file}")
    
    return r1, r2


def create_disparity_charts(df, logger):
    """Create disparity analysis charts."""
    logger.info("Creating disparity charts...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: CHHI by Borough
    ax1 = axes[0]
    
    borough_stats = df.groupby('borough')['housing_health_index'].agg(['mean', 'std', 'count'])
    borough_stats = borough_stats.sort_values('mean', ascending=False)
    
    bars = ax1.bar(range(len(borough_stats)), borough_stats['mean'], 
                   color=[BOROUGH_COLORS.get(b, '#888') for b in borough_stats.index],
                   edgecolor='white', linewidth=1)
    
    # Add error bars (standard error)
    ax1.errorbar(range(len(borough_stats)), borough_stats['mean'],
                yerr=borough_stats['std'] / np.sqrt(borough_stats['count']),
                fmt='none', color='black', capsize=4)
    
    ax1.set_xticks(range(len(borough_stats)))
    ax1.set_xticklabels(borough_stats.index, fontsize=11)
    ax1.set_ylabel('Child Health Housing Index (Mean)', fontsize=12)
    ax1.set_title('Housing-Health Risk by Borough', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for i, (idx, row) in enumerate(borough_stats.iterrows()):
        ax1.text(i, row['mean'] + 3, f'{row["mean"]:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: CHHI by Income Quintile
    ax2 = axes[1]
    
    # Create income quintiles using rank-based approach
    df_valid = df[df['median_income'].notna()].copy()
    # Use rank to handle ties
    df_valid['income_rank'] = df_valid['median_income'].rank(method='first')
    df_valid['income_quintile'] = pd.cut(df_valid['income_rank'], 
                                         bins=5, 
                                         labels=['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)'])
    
    quintile_stats = df_valid.groupby('income_quintile')['housing_health_index'].agg(['mean', 'std', 'count'])
    
    # Color gradient from red (low income) to green (high income)
    income_colors = ['#d32f2f', '#ff9800', '#fdd835', '#8bc34a', '#4caf50']
    
    bars = ax2.bar(range(len(quintile_stats)), quintile_stats['mean'], 
                   color=income_colors, edgecolor='white', linewidth=1)
    
    ax2.errorbar(range(len(quintile_stats)), quintile_stats['mean'],
                yerr=quintile_stats['std'] / np.sqrt(quintile_stats['count']),
                fmt='none', color='black', capsize=4)
    
    ax2.set_xticks(range(len(quintile_stats)))
    ax2.set_xticklabels(quintile_stats.index, fontsize=10)
    ax2.set_xlabel('Income Quintile', fontsize=12)
    ax2.set_ylabel('Child Health Housing Index (Mean)', fontsize=12)
    ax2.set_title('Housing-Health Risk by Neighborhood Income', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for i, (idx, row) in enumerate(quintile_stats.iterrows()):
        ax2.text(i, row['mean'] + 3, f'{row["mean"]:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Add disparity annotation
    gap = quintile_stats.loc['Q1\n(Lowest)', 'mean'] - quintile_stats.loc['Q5\n(Highest)', 'mean']
    ax2.annotate(f'Gap: {gap:.1f} points', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / 'disparity_charts.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'disparity_charts.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  Saved: {output_file}")


def create_top_10_table(df, logger):
    """Create table showing top 10 best and worst neighborhoods."""
    logger.info("Creating top 10 neighborhoods table...")
    
    # Sort by index
    df_sorted = df.sort_values('housing_health_index', ascending=False)
    
    # Get top 10 worst and best
    worst_10 = df_sorted.head(10)[['nta', 'borough', 'housing_health_index', 'risk_tier']].copy()
    best_10 = df_sorted.tail(10)[['nta', 'borough', 'housing_health_index', 'risk_tier']].copy()
    best_10 = best_10.iloc[::-1]  # Reverse order (best at top)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Hide axes
    for ax in axes:
        ax.axis('off')
    
    # Worst 10 table
    ax1 = axes[0]
    ax1.set_title('Top 10 Highest Risk Neighborhoods', fontsize=14, fontweight='bold', 
                  color='#d32f2f', pad=20)
    
    cell_text = [[row['nta'][:30], row['borough'], f"{row['housing_health_index']:.1f}", row['risk_tier']] 
                 for _, row in worst_10.iterrows()]
    
    table1 = ax1.table(cellText=cell_text,
                       colLabels=['Neighborhood', 'Borough', 'CHHI', 'Risk Tier'],
                       loc='center',
                       cellLoc='left',
                       colWidths=[0.45, 0.2, 0.15, 0.2])
    
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.8)
    
    # Color header
    for i in range(4):
        table1[(0, i)].set_facecolor('#d32f2f')
        table1[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color rows by risk tier
    for row_idx in range(1, 11):
        tier = worst_10.iloc[row_idx-1]['risk_tier']
        color = RISK_COLORS.get(tier, '#fff')
        for col_idx in range(4):
            table1[(row_idx, col_idx)].set_facecolor(color)
            if tier in ['Very High Risk', 'High Risk']:
                table1[(row_idx, col_idx)].set_text_props(color='white')
    
    # Best 10 table
    ax2 = axes[1]
    ax2.set_title('Top 10 Lowest Risk Neighborhoods', fontsize=14, fontweight='bold', 
                  color='#4caf50', pad=20)
    
    cell_text = [[row['nta'][:30], row['borough'], f"{row['housing_health_index']:.1f}", row['risk_tier']] 
                 for _, row in best_10.iterrows()]
    
    table2 = ax2.table(cellText=cell_text,
                       colLabels=['Neighborhood', 'Borough', 'CHHI', 'Risk Tier'],
                       loc='center',
                       cellLoc='left',
                       colWidths=[0.45, 0.2, 0.15, 0.2])
    
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.8)
    
    # Color header
    for i in range(4):
        table2[(0, i)].set_facecolor('#4caf50')
        table2[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color rows
    for row_idx in range(1, 11):
        tier = best_10.iloc[row_idx-1]['risk_tier']
        color = RISK_COLORS.get(tier, '#fff')
        for col_idx in range(4):
            table2[(row_idx, col_idx)].set_facecolor(color)
    
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / 'top_10_neighborhoods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'top_10_neighborhoods.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  Saved: {output_file}")


def create_violation_breakdown(df, logger):
    """Create violation type breakdown chart."""
    logger.info("Creating violation breakdown chart...")
    
    # Sum violation types across all neighborhoods
    violation_types = {
        'Lead Paint': df['violation_lead'].sum(),
        'Mold/Moisture': df['violation_mold'].sum(),
        'Pests/Vermin': df['violation_pests'].sum(),
        'Heat/Hot Water': df['violation_heat'].sum(),
    }
    
    # Calculate "other" (total - health violations)
    total_health = sum(violation_types.values())
    total_all = df['violation_count'].sum()
    violation_types['Other'] = total_all - total_health
    
    # Colors
    colors = ['#d32f2f', '#795548', '#4caf50', '#ff9800', '#9e9e9e']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(
        violation_types.values(), 
        labels=violation_types.keys(),
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.02] * len(violation_types),
        textprops={'fontsize': 10}
    )
    
    ax1.set_title('Violation Types (Citywide)', fontsize=14, fontweight='bold')
    
    # Bar chart showing counts
    ax2 = axes[1]
    
    bars = ax2.barh(list(violation_types.keys()), list(violation_types.values()), 
                    color=colors, edgecolor='white', linewidth=1)
    
    ax2.set_xlabel('Number of Violations', fontsize=12)
    ax2.set_title('Violation Counts by Type', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, violation_types.values()):
        ax2.text(val + total_all * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:,.0f}', va='center', fontsize=10)
    
    # Format x-axis with commas
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / 'violation_breakdown.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'violation_breakdown.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  Saved: {output_file}")


def create_risk_tier_map(df, logger):
    """Create a simple risk tier distribution map (bar chart by borough)."""
    logger.info("Creating risk tier distribution chart...")
    
    # Create crosstab of borough vs risk tier
    risk_order = ['Very High Risk', 'High Risk', 'Elevated Risk', 'Moderate Risk', 'Low Risk']
    
    crosstab = pd.crosstab(df['borough'], df['risk_tier'])
    crosstab = crosstab.reindex(columns=[c for c in risk_order if c in crosstab.columns])
    
    # Sort boroughs by total high risk
    if 'High Risk' in crosstab.columns and 'Very High Risk' in crosstab.columns:
        crosstab['_sort'] = crosstab.get('Very High Risk', 0) + crosstab.get('High Risk', 0)
    elif 'High Risk' in crosstab.columns:
        crosstab['_sort'] = crosstab['High Risk']
    else:
        crosstab['_sort'] = 0
    crosstab = crosstab.sort_values('_sort', ascending=False)
    crosstab = crosstab.drop('_sort', axis=1)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = [RISK_COLORS.get(tier, '#888') for tier in crosstab.columns]
    
    crosstab.plot(kind='bar', stacked=True, ax=ax, color=colors, 
                  edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Borough', fontsize=12)
    ax.set_ylabel('Number of Neighborhoods', fontsize=12)
    ax.set_title('Risk Tier Distribution by Borough', fontsize=14, fontweight='bold')
    ax.legend(title='Risk Tier', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / 'risk_tier_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'risk_tier_distribution.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  Saved: {output_file}")


def main():
    """Main function."""
    log_file = get_timestamped_log_filename("create_static_figures")
    logger = setup_logger(__name__, log_file)
    
    logger.info("=" * 60)
    logger.info("STARTING: Create Static Publication Figures")
    logger.info("=" * 60)
    
    try:
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load data
        df = load_data(logger)
        
        # Create figures
        r1, r2 = create_correlation_scatter(df, logger)
        logger.info(f"  Violations-Asthma correlation: r={r1:.3f}")
        logger.info(f"  Violations-Lead correlation: r={r2:.3f}")
        
        create_disparity_charts(df, logger)
        create_top_10_table(df, logger)
        create_violation_breakdown(df, logger)
        create_risk_tier_map(df, logger)
        
        logger.info("=" * 60)
        logger.info("COMPLETED: All figures created successfully")
        logger.info("=" * 60)
        
        print(f"\n✅ Figures saved to: {OUTPUT_DIR}")
        print("   - correlation_scatter.png/pdf")
        print("   - disparity_charts.png/pdf")
        print("   - top_10_neighborhoods.png/pdf")
        print("   - violation_breakdown.png/pdf")
        print("   - risk_tier_distribution.png/pdf")
        
    except Exception as e:
        logger.error(f"Error creating figures: {e}")
        raise


if __name__ == "__main__":
    main()

