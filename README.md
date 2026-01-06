# NYC Building Health & Safety Index

A comprehensive tool to compare any NYC building to 767,000+ residential properties across New York City. Analyze housing violations, 311 complaints, rodent inspections, and building safety data.

## 🏢 Features

- **Comprehensive Building Scores**: Combines HPD violations, rodent failures, bedbug infestations, and DOB violations
- **Neighborhood Comparison**: Compare buildings against citywide and local neighborhood benchmarks
- **Health Category Analysis**: Track lead, mold, pests, heat/hot water, and other health-related violations
- **Violation Timeline**: See violation trends over time
- **Building Comparison**: Compare multiple buildings side-by-side

## 📊 Data Sources

All data comes from [NYC Open Data](https://opendata.cityofnewyork.us/):

- [HPD Housing Maintenance Code Violations](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Violations/wvxf-dwi5) - 5.5M violations
- [HPD Housing Maintenance Code Complaints](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Complaints-Problems/a2h7-g4k6) - 2.2M complaints
- [DOHMH Rodent Inspections](https://data.cityofnewyork.us/Health/Rodent-Inspection/p937-wjvj) - Active rodent activity tracking
- [DOB Bedbug Reporting](https://data.cityofnewyork.us/Housing-Development/Bedbug-Reporting/wz6d-d3jb) - Bedbug infestations
- [PLUTO](https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks) - Building characteristics

## 🚀 Live Demo

[**Launch App →**](https://huggingface.co/spaces/YOUR_USERNAME/nyc-housing-health)

## 🛠️ Local Development

### Prerequisites
- Python 3.9+
- 600MB+ free disk space (for database)

### Setup

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/nyc-housing-health
cd nyc-housing-health

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📈 Scoring Methodology

The **Comprehensive Score** combines multiple data sources into a single health/safety index:

1. **HPD Violations**: Weighted by severity class (A=1, B=2, C=3, I=4)
2. **Rodent Failures**: Active rat/mouse activity (+15 points per failure)
3. **Bedbug Infestations**: (+10 points per infestation)
4. **DOB Violations**: (+5 points per violation)

The score is then **adjusted per unit** for multi-unit buildings (10+ units) to fairly compare large apartment buildings to smaller properties.

**Percentile Rankings**: A building in the 90th percentile has more health/safety issues than 90% of NYC buildings.

## 📁 Project Structure

```
nyc-housing-health/
├── app.py                    # Main Streamlit application
├── data/
│   ├── processed/
│   │   └── building_lookup.db   # SQLite database (543MB)
│   └── geo/
│       └── nta_boundaries.geojson
├── requirements.txt
└── README.md
```

## 📋 License

This project uses publicly available data from NYC Open Data. Data is provided under the [NYC Open Data Terms of Use](https://opendata.cityofnewyork.us/overview/#termsofuse).

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.
