#!/bin/bash
# Comprehensive suicide detection pipeline runner
# This script orchestrates all enhanced experiments

set -e  # Exit on error

echo "=================================="
echo "Suicide Detection Comprehensive Pipeline"
echo "=================================="
echo ""

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/comprehensive_run_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}/logs"

echo "Output directory: ${OUTPUT_DIR}"
echo "Starting at: $(date)"
echo ""

# Function to run a command and log output
run_stage() {
    local stage_name=$1
    shift
    local cmd="$@"
    
    echo "----------------------------------------"
    echo "Stage: ${stage_name}"
    echo "Command: ${cmd}"
    echo "----------------------------------------"
    
    if eval "${cmd}" 2>&1 | tee -a "${LOG_FILE}"; then
        echo "✓ ${stage_name} completed successfully"
    else
        echo "✗ ${stage_name} failed"
        return 1
    fi
    echo ""
}

# Check dependencies
echo "Checking dependencies..."
python -c "import optuna" || pip install optuna
python -c "import rich" || pip install rich
python -c "import seaborn" || pip install seaborn matplotlib
echo ""

# Stage 1: Data preparation (if needed)
if [ ! -d "data/kaggle/splits" ] || [ ! -d "data/mendeley/splits" ]; then
    echo "⚠️  Missing data splits. Please run data preparation first:"
    echo "   python scripts/prepare_datasets.py"
    echo ""
fi

# Stage 2: Run master pipeline (limited version for quick test)
echo "Running comprehensive pipeline..."
run_stage "Master Pipeline" \
    "python scripts/master_pipeline.py \
        --output_dir ${OUTPUT_DIR} \
        --stages baseline_training cross_dataset"

# Stage 3: Generate summary report
echo "Generating summary report..."
if [ -f "${OUTPUT_DIR}/pipeline_manifest.json" ]; then
    echo "Pipeline manifest created successfully"
    
    # Create summary
    cat > "${OUTPUT_DIR}/summary.txt" << EOF
====================================
Pipeline Execution Summary
====================================
Timestamp: ${TIMESTAMP}
Output Directory: ${OUTPUT_DIR}

Stages Executed:
- Baseline Training
- Cross-Dataset Evaluation

Key Artifacts:
- Pipeline Manifest: ${OUTPUT_DIR}/pipeline_manifest.json
- Final Report: ${OUTPUT_DIR}/final_report.html
- Logs: ${OUTPUT_DIR}/logs/

To view results:
  open ${OUTPUT_DIR}/final_report.html

To run full pipeline with all stages:
  python scripts/master_pipeline.py --output_dir results/full_run

====================================
EOF
    
    cat "${OUTPUT_DIR}/summary.txt"
fi

echo ""
echo "=================================="
echo "Pipeline completed at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=================================="

# Open report if on macOS
if [ "$(uname)" == "Darwin" ] && [ -f "${OUTPUT_DIR}/final_report.html" ]; then
    echo "Opening report in browser..."
    open "${OUTPUT_DIR}/final_report.html"
fi
