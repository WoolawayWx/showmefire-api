#!/bin/bash

# Show Me Fire - Automated ML Model Training Pipeline
# This script automates the complete process of training a new fuel moisture prediction model

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default options
SKIP_INGEST=false
SKIP_INDEX=false
SKIP_SNAPSHOTS=false
SKIP_EXTRACT=false
FULL_RETRAIN=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-ingest)
            SKIP_INGEST=true
            shift
            ;;
        --skip-index)
            SKIP_INDEX=true
            shift
            ;;
        --skip-snapshots)
            SKIP_SNAPSHOTS=true
            shift
            ;;
        --skip-extract)
            SKIP_EXTRACT=true
            shift
            ;;
        --full-retrain)
            FULL_RETRAIN=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Show Me Fire - Automated ML Model Training Pipeline"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-ingest     Skip data ingestion (observations)"
            echo "  --skip-index      Skip station indexing"
            echo "  --skip-snapshots  Skip snapshot creation"
            echo "  --skip-extract    Skip HRRR feature extraction"
            echo "  --full-retrain    Reset all snapshots for complete reprocessing"
            echo "  --dry-run         Show what would be executed without running"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Full pipeline run"
            echo "  $0 --skip-ingest           # Skip ingestion, start from indexing"
            echo "  $0 --full-retrain          # Complete reprocessing of all data"
            echo "  $0 --dry-run               # Preview the pipeline steps"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python environment is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        error "Python3 is not available. Please install Python 3.7+"
        exit 1
    fi

    # Skip package check in dry-run mode
    if [[ "$DRY_RUN" != "true" ]]; then
        check_python_packages
    fi
}

# Check if required packages are installed
check_python_packages() {
    local missing_packages=""

    if ! python3 -c "import pandas" 2>/dev/null; then
        missing_packages="$missing_packages pandas"
    fi
    if ! python3 -c "import xgboost" 2>/dev/null; then
        missing_packages="$missing_packages xgboost"
    fi
    if ! python3 -c "import sklearn" 2>/dev/null; then
        missing_packages="$missing_packages scikit-learn"
    fi

    if [[ -n "$missing_packages" ]]; then
        error "Required Python packages not found: $missing_packages"
        echo "Please run: pip install$missing_packages"
        exit 1
    fi
}

# Check if required directories exist
check_directories() {
    local dirs=("data" "models" "plots" "cache/hrrr")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            warning "Directory '$dir' does not exist. Creating..."
            if [[ "$DRY_RUN" != "true" ]]; then
                mkdir -p "$dir"
            fi
        fi
    done
}

# Run a Python script with error handling
run_python_script() {
    local script="$1"
    local description="$2"

    log "Running: $description"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  python3 pipelines/$script"
        return 0
    fi

    if [[ -f "pipelines/$script" ]]; then
        if python3 "pipelines/$script"; then
            success "$description completed"
        else
            error "$description failed"
            exit 1
        fi
    else
        error "Script not found: pipelines/$script"
        exit 1
    fi
}

# Main pipeline execution
main() {
    echo "🔥 Show Me Fire - ML Model Training Pipeline"
    echo "=========================================="

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "🔍 DRY RUN MODE - Showing pipeline steps"
        echo ""
    fi

    # Pre-flight checks
    check_python
    check_directories

    # Phase 1: Data Ingestion (One-time setup)
    if [[ "$SKIP_INGEST" != "true" ]]; then
        log "Phase 1: Data Ingestion"
        run_python_script "ingest_obs.py" "Ingest RAWS observations into database"
    else
        log "Skipping data ingestion (--skip-ingest)"
    fi

    # Phase 2: Station Indexing (Performance optimization)
    if [[ "$SKIP_INDEX" != "true" ]]; then
        log "Phase 2: Station Indexing"
        run_python_script "index_stations.py" "Index stations for HRRR grid lookup optimization"
    else
        log "Skipping station indexing (--skip-index)"
    fi

    # Phase 3: Snapshot Management
    log "Phase 3: Snapshot Management"

    if [[ "$FULL_RETRAIN" == "true" ]]; then
        warning "Full retrain requested - resetting all snapshots"
        if [[ "$DRY_RUN" != "true" ]]; then
            python3 scripts/reset_snapshots.py
        else
            echo "  python3 scripts/reset_snapshots.py"
        fi
    fi

    if [[ "$SKIP_SNAPSHOTS" != "true" ]]; then
        run_python_script "../scripts/create_snapshots.py" "Create snapshots for new HRRR data"
    else
        log "Skipping snapshot creation (--skip-snapshots)"
    fi

    # Phase 4: Feature Extraction
    if [[ "$SKIP_EXTRACT" != "true" ]]; then
        log "Phase 4: HRRR Feature Extraction"
        run_python_script "extract_hrrr.py" "Extract weather features from HRRR data"
    else
        log "Skipping HRRR extraction (--skip-extract)"
    fi

    # Phase 5: Dataset Generation
    log "Phase 5: Dataset Generation"
    run_python_script "generate_training_set.py" "Generate training dataset from observations + weather features"

    # Phase 6: Feature Engineering
    log "Phase 6: Feature Engineering"
    run_python_script "prepare_features.py" "Prepare ML features with rolling averages and precipitation metrics"

    # Phase 7: Model Training
    log "Phase 7: Model Training"
    run_python_script "train_model.py" "Train XGBoost model with new data"

    # Phase 8: Validation
    log "Phase 8: Model Validation"
    if [[ "$DRY_RUN" != "true" ]]; then
        echo ""
        echo "📊 Training Data Summary:"
        python3 -c "
import pandas as pd
try:
    df = pd.read_csv('data/final_training_data.csv')
    print(f'   Total training samples: {len(df):,}')
    print(f'   Date range: {df[\"obs_time\"].min()} to {df[\"obs_time\"].max()}')
    print(f'   Stations: {df[\"station_id\"].nunique()}')
    print(f'   Features: {len([c for c in df.columns if not c.startswith(\"target\")])}')
except FileNotFoundError:
    print('   No training data found')
" 2>/dev/null || echo "   Could not read training data"
    fi

    # Success message
    echo ""
    success "ML Pipeline completed successfully!"
    echo ""
    echo "📁 Generated files:"
    echo "   • models/fuel_moisture_model.json (new model)"
    echo "   • data/final_training_data.csv (training dataset)"
    echo "   • plots/feature_importance.png (model insights)"
    echo "   • plots/station_preview.png (data quality check)"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Test the model: python3 pipelines/predict.py"
    echo "   2. Run forecast: python3 forecast/DailyForecast.py"
    echo "   3. Validate results: python3 scripts/compare_forecasts.py"
}

# Run main function
main "$@"