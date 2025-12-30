# NYC Housing Violations Lookup Tool

A building lookup tool that allows users to search any NYC address and see how its housing code violations compare to other buildings citywide and within its neighborhood.

## What This Tool Does

Enter any NYC address to see:
- **Violation counts and percentiles** - How this building ranks against all NYC buildings
- **Neighborhood comparison** - How the building compares to others in its immediate area
- **Adjusted vs raw scores** - Both per-unit normalized and absolute violation counts
- **Violation breakdown** - Categories including lead, mold, pests, and heat issues
- **Complaint history** - 311 complaints filed against the building
- **Trend analysis** - Whether violations are increasing or decreasing over time
- **Nearby buildings** - Comparison with buildings within 200 meters

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will open at http://localhost:8501

## Methodology

### Data Sources

| Data | Source | Description |
|------|--------|-------------|
| Housing Violations | NYC HPD via Open Data | Housing Maintenance Code violations from inspections |
| Housing Complaints | NYC HPD via Open Data | 311 complaints about housing conditions |
| Building Info | NYC PLUTO | Year built, units, floors, location |
| NTA Boundaries | NYC Open Data | Neighborhood Tabulation Areas for geographic grouping |

### Two Comparison Metrics

**1. Adjusted Score (Per-Unit Normalized)**

For fair comparison across buildings of different sizes, we calculate a severity-weighted score normalized by the number of units:

```
Adjusted Score = (Class A × 1 + Class B × 2 + Class C × 3) / Number of Units
```

- Class A: Non-hazardous (weight: 1)
- Class B: Hazardous (weight: 2)
- Class C: Immediately hazardous (weight: 3)

This accounts for the fact that a 100-unit building will naturally have more violations than a 5-unit building.

**2. Raw Violation Count**

The total number of violations regardless of building size. Useful for understanding absolute conditions but not for fair cross-building comparison.

### Percentile Calculations

For each metric, buildings are ranked against:
- **All NYC buildings** - Citywide percentile (e.g., "worse than 75% of NYC")
- **Neighborhood buildings** - Local percentile within the same NTA

A building at the 75th percentile has more violations than 75% of comparison buildings.

### Violation Categories

Violations are categorized by keyword matching in the violation description:
- **Lead**: Lead paint hazards, lead-based paint violations
- **Mold**: Mold, mildew, moisture damage
- **Pests**: Roaches, mice, rats, bedbugs, vermin
- **Heat**: Heat, hot water, heating system issues

### Trend Analysis

Compares violations from the last 12 months against the prior 12 months to determine if conditions are improving or worsening.

## Limitations

### Data Coverage

- **Residential buildings only** - Only buildings with 3+ residential units are tracked by HPD
- **Single/two-family homes excluded** - These are not in the HPD violations database
- **Commercial buildings excluded** - Non-residential properties are not included

### Data Currency

- **Historical records** - Violation data reflects when inspections occurred, not necessarily current conditions
- **Open vs closed** - Some violations may be resolved but still appear in historical counts
- **Inspection frequency varies** - Buildings are not inspected on a regular schedule; data reflects complaint-driven inspections

### Methodological Limitations

- **Keyword matching** - Violation categorization (lead, mold, etc.) relies on text matching and may miss some relevant violations or include false positives
- **Building size normalization** - Adjusted score assumes violations scale linearly with units, which may not always hold
- **Neighborhood boundaries** - NTA boundaries are administrative, not necessarily reflective of actual neighborhood conditions

### Reporting Bias

- **311 complaint-driven** - Most inspections result from tenant complaints; buildings with fewer complaints may have fewer documented violations regardless of actual conditions
- **Underreporting in some areas** - Tenants in certain neighborhoods may be less likely to file complaints due to fear of retaliation, language barriers, or other factors

### What This Tool Does NOT Show

- **Health outcomes** - This tool shows housing violations, not health data. Correlation between housing conditions and health outcomes exists but is not displayed here
- **Current building conditions** - Violations are historical records; current conditions may differ
- **Causation** - High violation counts indicate documented problems but don't establish causation for any health or quality-of-life outcomes

## Data Sources

- [HPD Housing Maintenance Code Violations](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Violations/wvxf-dwi5)
- [HPD Housing Maintenance Code Complaints](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Complaints/uwyv-629c)
- [NYC PLUTO](https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks)
- [NTA Boundaries](https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd)

## License

MIT License
