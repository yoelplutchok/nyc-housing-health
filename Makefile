# NYC Housing Health Index - Makefile
# ====================================
# Run `make help` to see available targets

.PHONY: setup collect process analyze visualize test clean all help web-dev web-build

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# SETUP
# ============================================================================

setup: ## Set up the development environment
	@echo "Creating conda environment..."
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Installing package in development mode..."
	pip install -e .
	@echo "Setting up web dependencies..."
	cd web && npm install
	@echo "Setup complete! Activate with: conda activate housing-health"

setup-python: ## Set up Python environment only (no Node.js)
	conda env create -f environment.yml || conda env update -f environment.yml
	pip install -e .

# ============================================================================
# DATA COLLECTION
# ============================================================================

collect: ## Download all data from NYC Open Data
	@echo "Downloading HPD Violations..."
	python scripts/collect/download_hpd_violations.py
	@echo "Downloading HPD Complaints..."
	python scripts/collect/download_hpd_complaints.py
	@echo "Downloading 311 Requests..."
	python scripts/collect/download_311_requests.py
	@echo "Downloading Health Data..."
	python scripts/collect/download_health_data.py
	@echo "Downloading Geographic Boundaries..."
	python scripts/collect/download_geo.py
	@echo "Downloading Census Demographics..."
	python scripts/collect/download_census.py
	@echo "Data collection complete!"

collect-violations: ## Download HPD violations only
	python scripts/collect/download_hpd_violations.py

collect-complaints: ## Download HPD complaints only
	python scripts/collect/download_hpd_complaints.py

collect-311: ## Download 311 requests only
	python scripts/collect/download_311_requests.py

collect-geo: ## Download geographic boundaries only
	python scripts/collect/download_geo.py

# ============================================================================
# DATA PROCESSING
# ============================================================================

process: ## Process and clean all downloaded data
	@echo "Cleaning violations data..."
	python scripts/process/clean_violations.py
	@echo "Cleaning 311 data..."
	python scripts/process/clean_311.py
	@echo "Aggregating to neighborhoods..."
	python scripts/process/aggregate_to_neighborhoods.py
	@echo "Joining health data..."
	python scripts/process/join_health_data.py
	@echo "Building address lookup database..."
	python scripts/process/build_address_lookup.py
	@echo "Creating master dataset..."
	python scripts/process/create_master_dataset.py
	@echo "Data processing complete!"

# ============================================================================
# ANALYSIS
# ============================================================================

analyze: ## Run all statistical analyses
	@echo "Calculating Child Health Housing Index..."
	python scripts/analyze/calculate_index.py
	@echo "Running correlation analysis..."
	python scripts/analyze/correlation_analysis.py
	@echo "Running disparity analysis..."
	python scripts/analyze/disparity_analysis.py
	@echo "Running temporal trends analysis..."
	python scripts/analyze/temporal_trends.py
	@echo "Analysis complete!"

# ============================================================================
# VISUALIZATION
# ============================================================================

visualize: ## Generate all visualizations
	@echo "Creating neighborhood map..."
	python scripts/visualize/create_neighborhood_map.py
	@echo "Creating static figures..."
	python scripts/visualize/create_static_figures.py
	@echo "Visualizations complete!"

# ============================================================================
# WEB APPLICATION
# ============================================================================

web-dev: ## Run web application in development mode
	cd web && npm run dev

web-build: ## Build web application for production
	cd web && npm run build

web-start: ## Start production web server
	cd web && npm start

# ============================================================================
# TESTING & VALIDATION
# ============================================================================

test: ## Run validation tests
	python scripts/validate/run_validation.py

lint: ## Run linting checks
	ruff check src/ scripts/
	black --check src/ scripts/

format: ## Auto-format code
	black src/ scripts/
	ruff check --fix src/ scripts/

# ============================================================================
# UTILITIES
# ============================================================================

clean: ## Remove processed data and outputs (keeps raw data)
	rm -rf data/processed/*
	rm -rf outputs/figures/*
	rm -rf outputs/interactive/*
	rm -rf outputs/tables/*
	@echo "Cleaned processed data and outputs."

clean-all: ## Remove all data including raw downloads
	rm -rf data/raw/hpd_violations/*
	rm -rf data/raw/hpd_complaints/*
	rm -rf data/raw/311_requests/*
	rm -rf data/processed/*
	rm -rf data/geo/*
	rm -rf data/health/*
	rm -rf outputs/*
	@echo "Cleaned all data."

# ============================================================================
# FULL PIPELINE
# ============================================================================

all: collect process analyze visualize test ## Run the complete pipeline
	@echo "=============================================="
	@echo "Full pipeline complete!"
	@echo "=============================================="

# ============================================================================
# HELP
# ============================================================================

help: ## Show this help message
	@echo "NYC Housing Health Index - Available Commands"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

