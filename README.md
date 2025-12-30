# NYC Child Health Housing Index

**Mapping the Connection Between Housing Conditions and Child Health in NYC**

This project builds an interactive dashboard and address lookup tool that quantifies the relationship between housing code violations (lead, mold, pests, heat) and pediatric health outcomes (asthma, lead poisoning) across NYC neighborhoods.

## 🎯 Project Goals

1. **Neighborhood Explorer** - Interactive choropleth map showing a composite "Child Health Housing Index" by neighborhood
2. **Address Lookup Tool** - Enter any NYC address → get a health risk score based on building violations + neighborhood context
3. **Correlation Analysis** - Statistical evidence showing the housing-health connection

## 🆕 Recent Updates (December 2024)

### Data Accuracy Fixes
- **Fixed percentile calculation bug** - Total Violations now correctly uses `violation_count_pct` instead of `violations_open_pct`
- **Improved NTA coverage** - Spatial join now assigns neighborhoods to 99.9% of buildings (up from 23.5%)
- **Enhanced health violation detection** - Expanded keyword matching for lead, mold, pest, and heat violations

### New Features
- **Time Period Filtering** - Filter violations by All Time, Last 2 Years, or Last Year
- **Violations Per Unit** - Fair comparison metric normalized by building size
- **Class A/B/C Breakdown** - Separate percentile rankings by violation severity class
- **Trend Indicators** - Compare last 12 months vs prior year with directional arrows
- **Building Comparison** - Add multiple buildings to compare side-by-side
- **Fuzzy Address Matching** - Recognizes street name variants (St/Street, Ave/Avenue, etc.)

### UX Improvements
- **Zero-violation messaging** - Shows "Better than X% of NYC buildings" instead of confusing 0th percentile
- **Improved tooltips** - Neighborhood map shows detailed stats on hover
- **Data freshness indicator** - Shows when data was last updated

### Technical Improvements
- **Database connection context manager** - Prevents connection leaks
- **Error handling in map generation** - Graceful fallback for invalid data
- **Configuration constants** - Magic numbers extracted to config section

## 📊 Key Metrics

| Metric | Definition | Source |
|--------|------------|--------|
| Housing Violations Score | Weighted violations per 1,000 housing units | HPD |
| 311 Complaints Score | Housing-health complaints per 1,000 residents (bias-adjusted) | 311 |
| Child Asthma Rate | Pediatric asthma ED visits per 10,000 children | NYC DOHMH |
| Lead Poisoning Rate | % children with elevated blood lead levels | NYC DOHMH |
| Building Age Score | % housing stock pre-1978 (lead paint era) | PLUTO |
| **Child Health Housing Index (CHHI)** | Composite score 0-100 | Calculated |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip
- geopandas (for spatial NTA assignment)
- Node.js 18+ (for optional web application)

```bash
# Install geopandas (requires proj library)
brew install proj  # macOS
pip install geopandas
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/nyc-housing-health.git
cd nyc-housing-health

# Create conda environment
conda env create -f environment.yml
conda activate housing-health

# Install package
pip install -e .

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys (optional but recommended)
```

### Run the Pipeline

```bash
# Run everything
make all

# Or run individual steps:
make collect   # Download data from NYC Open Data
make process   # Clean and aggregate data
make analyze   # Calculate indices and run analyses
make visualize # Generate maps and charts
```

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open at http://localhost:8501 with:
- Address lookup with building health scores
- Neighborhood comparison maps
- Building comparison tool

### Web Application (Optional)

```bash
cd web
npm install
npm run dev
```

## 📁 Project Structure

```
nyc-housing-health/
├── data/
│   ├── raw/              # Original downloaded files
│   ├── processed/        # Cleaned and transformed data
│   ├── geo/              # Geographic boundaries
│   └── health/           # Health outcome data
├── scripts/
│   ├── collect/          # Data collection scripts
│   ├── process/          # Data cleaning and transformation
│   ├── analyze/          # Statistical analysis
│   └── visualize/        # Chart generation
├── src/housing_health/   # Core Python utilities
├── web/                  # Next.js web application
├── outputs/
│   ├── figures/          # Static charts and maps
│   ├── interactive/      # HTML maps
│   └── tables/           # Summary statistics
├── docs/                 # Documentation
├── configs/              # Configuration files
└── tests/                # Test files
```

## 📚 Data Sources

- **HPD Violations**: [NYC Open Data](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Violations/wvxf-dwi5)
- **HPD Complaints**: [NYC Open Data](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Complaints/uwyv-629c)
- **311 Requests**: [NYC Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)
- **PLUTO**: [NYC Open Data](https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks)
- **Health Data**: [NYC Environment & Health Data Portal](https://a816-dohbesp.nyc.gov/IndicatorPublic/data-explorer/)
- **NTA Boundaries**: [NYC Open Data](https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd)

## 🔧 Configuration

All configurable parameters are in `configs/params.yml`:
- Data collection date ranges
- Violation keywords for categorization
- Index component weights
- Color scales and thresholds

## 📖 Methodology

See [docs/methodology.md](docs/methodology.md) for detailed documentation of:
- Data processing and cleaning
- Index calculation methodology
- 311 bias adjustment approach
- Statistical analysis methods

## ⚠️ Limitations

1. **311 reporting bias** - Reporting rates vary by neighborhood income/demographics
2. **Health data granularity** - Health data is aggregated to UHF42 areas (42 neighborhoods), not NTA level
3. **Correlation ≠ causation** - Housing conditions correlate with but don't necessarily cause health outcomes
4. **Violation detection** - Keyword-based categorization may miss some health-related violations
5. **Data currency** - Violation data reflects historical records; current building conditions may differ

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.

## 📧 Contact

For questions about this project, please open an issue on GitHub.

