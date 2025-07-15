#!/bin/bash
# Simple Self-Attention Benchmark Script
# Runs forward and backward tests and calculates a combined score

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}NKI Self-Attention Benchmark${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Create log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs/self_attention/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Default NKI FLOP ratio (will be overridden by calculated on NKIfrom tests)
# This represents the ratio of operations that are accelerated by NKI
# Higher values indicate better hardware utilization
NKI_FLOP_RATIO=0.85  # Default: 85% of operations executed on NKI

# Step 1: Run forward test
echo -e "\n${YELLOW}Running forward attention test...${NC}"
cd "${PROJECT_ROOT}/src/self-attention"
python -m pytest tests/test_flash_attn_fwd.py::TestAttention::test_flash_attn_fwd_perf -v -s > "${LOG_DIR}/forward_perf.log"
FWD_PERF_STATUS=$?

python -m pytest tests/test_flash_attn_fwd.py::TestAttention::test_flash_attn_fwd_numerical -v -s > "${LOG_DIR}/forward_numerical.log"
FWD_NUM_STATUS=$?

# Step 2: Run backward test
echo -e "\n${YELLOW}Running backward attention test...${NC}"
python -m pytest tests/test_flash_attn_bwd.py::TestAttention::test_flash_attn_bwd_perf -v -s > "${LOG_DIR}/backward_perf.log"
BWD_PERF_STATUS=$?

python -m pytest tests/test_flash_attn_bwd.py::TestAttention::test_flash_attn_bwd_numerical -v -s > "${LOG_DIR}/backward_numerical.log"
BWD_NUM_STATUS=$?

# Step 3: Extract latency values from logs and config
echo -e "\n${YELLOW}Extracting metrics from test results...${NC}"

# Extract P50 latency from forward performance test
FWD_P50_LATENCY=$(grep -o "P50 Latency: [0-9,]* ns" "${LOG_DIR}/forward_perf.log" | grep -o "[0-9,]*" | tr -d ',')
if [[ -z "$FWD_P50_LATENCY" ]]; then
    echo -e "${RED}Could not extract forward P50 latency. Using baseline.${NC}"
    FWD_P50_LATENCY=$FWD_BASE_LATENCY
fi

# Extract P50 latency from backward performance test
BWD_P50_LATENCY=$(grep -o "P50 Latency: [0-9,]* ns" "${LOG_DIR}/backward_perf.log" | grep -o "[0-9,]*" | tr -d ',')
if [[ -z "$BWD_P50_LATENCY" ]]; then
    echo -e "${RED}Could not extract backward P50 latency. Using baseline.${NC}"
    BWD_P50_LATENCY=$BWD_BASE_LATENCY
fi

# Extract accumulated latency metrics from config file
CONFIG_FILE="${PROJECT_ROOT}/src/self-attention/config/performance_metrics.json"
if [[ -f "$CONFIG_FILE" ]]; then
    echo -e "${GREEN}Loading accumulated metrics from config file...${NC}"
    
    # Use jq if available, otherwise use grep and sed
    if command -v jq &> /dev/null; then
        FWD_LATENCY_TOTAL=$(jq -r '.FWD_LATENCY_TOTAL' "$CONFIG_FILE")
        FWD_BASE_LATENCY_TOTAL=$(jq -r '.FWD_BASE_LATENCY_TOTAL' "$CONFIG_FILE")
        FWD_TEST_COUNT=$(jq -r '.FWD_TEST_COUNT' "$CONFIG_FILE")
        BWD_LATENCY_TOTAL=$(jq -r '.BWD_LATENCY_TOTAL' "$CONFIG_FILE")
        BWD_BASE_LATENCY_TOTAL=$(jq -r '.BWD_BASE_LATENCY_TOTAL' "$CONFIG_FILE")
        BWD_TEST_COUNT=$(jq -r '.BWD_TEST_COUNT' "$CONFIG_FILE")
    else
        # Fallback to grep and sed if jq is not available
        FWD_LATENCY_TOTAL=$(grep -o '"FWD_LATENCY_TOTAL": [0-9.]*' "$CONFIG_FILE" | sed 's/.*: //')
        FWD_BASE_LATENCY_TOTAL=$(grep -o '"FWD_BASE_LATENCY_TOTAL": [0-9.]*' "$CONFIG_FILE" | sed 's/.*: //')
        FWD_TEST_COUNT=$(grep -o '"FWD_TEST_COUNT": [0-9.]*' "$CONFIG_FILE" | sed 's/.*: //')
        BWD_LATENCY_TOTAL=$(grep -o '"BWD_LATENCY_TOTAL": [0-9.]*' "$CONFIG_FILE" | sed 's/.*: //')
        BWD_BASE_LATENCY_TOTAL=$(grep -o '"BWD_BASE_LATENCY_TOTAL": [0-9.]*' "$CONFIG_FILE" | sed 's/.*: //')
        BWD_TEST_COUNT=$(grep -o '"BWD_TEST_COUNT": [0-9.]*' "$CONFIG_FILE" | sed 's/.*: //')
    fi
    
    # Use the accumulated metrics if available
    if [[ -n "$FWD_LATENCY_TOTAL" && -n "$FWD_BASE_LATENCY_TOTAL" && "$FWD_TEST_COUNT" -gt 0 ]]; then
        echo -e "${GREEN}Using accumulated forward metrics from ${FWD_TEST_COUNT} tests${NC}"
        FWD_AVG_LATENCY=$(echo "scale=2; $FWD_LATENCY_TOTAL / $FWD_TEST_COUNT" | bc)
        FWD_AVG_BASE_LATENCY=$(echo "scale=2; $FWD_BASE_LATENCY_TOTAL / $FWD_TEST_COUNT" | bc)
        echo -e "   Average achieved latency: ${FWD_AVG_LATENCY} ns"
        echo -e "   Average baseline latency: ${FWD_AVG_BASE_LATENCY} ns"
        FWD_P50_LATENCY=$FWD_AVG_LATENCY
        FWD_BASE_LATENCY=$FWD_AVG_BASE_LATENCY
    fi
    
    if [[ -n "$BWD_LATENCY_TOTAL" && -n "$BWD_BASE_LATENCY_TOTAL" && "$BWD_TEST_COUNT" -gt 0 ]]; then
        echo -e "${GREEN}Using accumulated backward metrics from ${BWD_TEST_COUNT} tests${NC}"
        BWD_AVG_LATENCY=$(echo "scale=2; $BWD_LATENCY_TOTAL / $BWD_TEST_COUNT" | bc)
        BWD_AVG_BASE_LATENCY=$(echo "scale=2; $BWD_BASE_LATENCY_TOTAL / $BWD_TEST_COUNT" | bc)
        echo -e "   Average achieved latency: ${BWD_AVG_LATENCY} ns"
        echo -e "   Average baseline latency: ${BWD_AVG_BASE_LATENCY} ns"
        BWD_P50_LATENCY=$BWD_AVG_LATENCY
        BWD_BASE_LATENCY=$BWD_AVG_BASE_LATENCY
    fi
else
    echo -e "${YELLOW}No accumulated metrics found. Using single test results.${NC}"
fi

# Step 4: Calculate scores using the Python script
echo -e "\n${YELLOW}Calculating scores...${NC}"

# Convert test status to pass/fail strings for the Python script
FWD_NUMERICAL_RESULT=$([ $FWD_NUM_STATUS -eq 0 ] && echo "pass" || echo "fail")
BWD_NUMERICAL_RESULT=$([ $BWD_NUM_STATUS -eq 0 ] && echo "pass" || echo "fail")

# Define weights
FWD_WEIGHT=0.4
BWD_WEIGHT=0.6

# Run the Python script to calculate scores
SCORE_OUTPUT_FILE="${LOG_DIR}/score_details.json"
echo -e "${BLUE}Running score calculation script...${NC}"

# Extract NKI_FLOP_RATIO from config if available
if [[ -f "$CONFIG_FILE" ]]; then
    if command -v jq &> /dev/null; then
        CONFIG_NKI_FLOP_RATIO=$(jq -r '.NKI_FLOP_RATIO' "$CONFIG_FILE")
        if [[ -n "$CONFIG_NKI_FLOP_RATIO" && "$CONFIG_NKI_FLOP_RATIO" != "null" ]]; then
            NKI_FLOP_RATIO=$CONFIG_NKI_FLOP_RATIO
            echo -e "${GREEN}Using calculated NKI_FLOP_RATIO from config: ${NKI_FLOP_RATIO}${NC}"
        fi
    else
        # Fallback to grep and sed if jq is not available
        CONFIG_NKI_FLOP_RATIO=$(grep -o '"NKI_FLOATIO": [0- "$CONFIG_FILE" | sed 's/.*: //'')
        if [[ -n "$CONFIG_NKI_FLOP_RATIO" ]]; then
            NKI_FLOP_RATIO=$CONFIG_P_RATIO
            echo -e "${GREEN}Using calculated NKI_FLOP_RATIO from config: ${NKI_FLOP_RATIO}${NC}"
        fi
    fi
else
    -e "${YELLOW}No config file found. Using default NKI_FLOP_RATIO: ${NKI_FLOP_RATIO}${NC}"
fi

python "${SCRIPT_DIR}/calculate_score.py" \
  --fwd-latency "$FWD_P50_LATENCY" \
  --fwd-base-latency "$FWD_BASE_LATENCY" \
  --fwd-numerical-accuracy "$FWD_NUMERICAL_RESULT" \
  --bwd-latency "$BWD_P50_LATENCY" \
  --bwd-base-latency "$BWD_BASE_LATENCY" \
  --bwd-numerical-accuracy "$BWD_NUMERICAL_RESULT" \
  --fwd-weight "$FWD_WEIGHT" \
  --bwd-weight "$BWD_WEIGHT" \
  --nki-flop-ratio "$NKI_FLOP_RATIO" \
  --output "$SCORE_OUTPUT_FILE"

# Extract values from the JSON file
if [ -f "$SCORE_OUTPUT_FILE" ]; then
    # Use jq if available, otherwise use grep and sed
    if command -v jq &> /dev/null; then
        FWD_LATENCY_IMPROVEMENT=$(jq -r '.forward.latency_improvement' "$SCORE_OUTPUT_FILE")
        BWD_LATENCY_IMPROVEMENT=$(jq -r '.backward.latency_improvement' "$SCORE_OUTPUT_FILE")
        FWD_THROUGHPUT_IMPROVEMENT=$(jq -r '.forward.throughput_improvement' "$SCORE_OUTPUT_FILE")
        BWD_THROUGHPUT_IMPROVEMENT=$(jq -r '.backward.throughput_improvement' "$SCORE_OUTPUT_FILE")
        FWD_SCORE=$(jq -r '.forward.score' "$SCORE_OUTPUT_FILE")
        BWD_SCORE=$(jq -r '.backward.score' "$SCORE_OUTPUT_FILE")
        RAW_SCORE=$(jq -r '.combined.raw_score' "$SCORE_OUTPUT_FILE")
        COMBINED_SCORE=$(jq -r '.combined.score' "$SCORE_OUTPUT_FILE")
    else
        # Fallback to grep and sed if jq is not available
        FWD_LATENCY_IMPROVEMENT=$(grep -o '"latency_improvement": [0-9.]*' "$SCORE_OUTPUT_FILE" | head -1 | sed 's/.*: //')
        BWD_LATENCY_IMPROVEMENT=$(grep -o '"latency_improvement": [0-9.]*' "$SCORE_OUTPUT_FILE" | tail -1 | sed 's/.*: //')
        FWD_THROUGHPUT_IMPROVEMENT=$(grep -o '"throughput_improvement": [0-9.]*' "$SCORE_OUTPUT_FILE" | head -1 | sed 's/.*: //')
        BWD_THROUGHPUT_IMPROVEMENT=$(grep -o '"throughput_improvement": [0-9.]*' "$SCORE_OUTPUT_FILE" | tail -1 | sed 's/.*: //')
        FWD_SCORE=$(grep -o '"score": [0-9.]*' "$SCORE_OUTPUT_FILE" | head -1 | sed 's/.*: //')
        BWD_SCORE=$(grep -o '"score": [0-9.]*' "$SCORE_OUTPUT_FILE" | head -2 | tail -1 | sed 's/.*: //')
        RAW_SCORE=$(grep -o '"raw_score": [0-9.]*' "$SCORE_OUTPUT_FILE" | sed 's/.*: //')
        COMBINED_SCORE=$(grep -o '"score": [0-9.]*' "$SCORE_OUTPUT_FILE" | tail -1 | sed 's/.*: //')
    fi
else
    echo -e "${RED}Score calculation failed. Using manual calculation.${NC}"
    
    # Convert test status to numerical accuracy (1.0 for pass, 0.0 for fail)
    FWD_NUMERICAL_ACCURACY=$([ $FWD_NUM_STATUS -eq 0 ] && echo 1.0 || echo 0.0)
    BWD_NUMERICAL_ACCURACY=$([ $BWD_NUM_STATUS -eq 0 ] && echo 1.0 || echo 0.0)
    
    # Calculate latency improvements
    FWD_LATENCY_IMPROVEMENT=$(echo "scale=2; $FWD_BASE_LATENCY / $FWD_P50_LATENCY" | bc)
    BWD_LATENCY_IMPROVEMENT=$(echo "scale=2; $BWD_BASE_LATENCY / $BWD_P50_LATENCY" | bc)
    FWD_THROUGHPUT_IMPROVEMENT=$FWD_LATENCY_IMPROVEMENT
    BWD_THROUGHPUT_IMPROVEMENT=$BWD_LATENCY_IMPROVEMENT
    
    # Calculate individual scores
    FWD_SCORE=$(echo "scale=2; $FWD_NUMERICAL_ACCURACY * $FWD_LATENCY_IMPROVEMENT * $FWD_THROUGHPUT_IMPROVEMENT" | bc)
    BWD_SCORE=$(echo "scale=2; $BWD_NUMERICAL_ACCURACY * $BWD_LATENCY_IMPROVEMENT * $BWD_THROUGHPUT_IMPROVEMENT" | bc)
    
    # Calculate combined score
    COMBINED_NUMERICAL_ACCURACY=$(echo "scale=2; $FWD_NUMERICAL_ACCURACY * $BWD_NUMERICAL_ACCURACY" | bc)
    if (( $(echo "$COMBINED_NUMERICAL_ACCURACY < 1.0" | bc -l) )); then
        RAW_SCORE=0.0
        COMBINED_SCORE=0.0
    else
        RAW_SCORE=$(echo "scale=2; ($FWD_WEIGHT * $FWD_SCORE) + ($BWD_WEIGHT * $BWD_SCORE)" | bc)
        COMBINED_SCORE=$(echo "scale=2; $RAW_SCORE * (1.0 + $NKI_FLOP_RATIO)" | bc)
    fi
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "\nLog files saved to: ${LOG_DIR}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Save results to JSON file if the Python script didn't already create one
if [ ! -f "$SCORE_OUTPUT_FILE" ]; then
    cat > "${LOG_DIR}/results.json" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "forward": {
        "latency": ${FWD_P50_LATENCY},
        "base_latency": ${FWD_BASE_LATENCY},
        "latency_improvement": ${FWD_LATENCY_IMPROVEMENT},
        "throughput_improvement": ${FWD_THROUGHPUT_IMPROVEMENT},
        "numerical_accuracy": $([ "$FWD_NUMERICAL_RESULT" = "pass" ] && echo 1.0 || echo 0.0),
        "score": ${FWD_SCORE}
    },
    "backward": {
        "latency": ${BWD_P50_LATENCY},
        "base_latency": ${BWD_BASE_LATENCY},
        "latency_improvement": ${BWD_LATENCY_IMPROVEMENT},
        "throughput_improvement": ${BWD_THROUGHPUT_IMPROVEMENT},
        "numerical_accuracy": $([ "$BWD_NUMERICAL_RESULT" = "pass" ] && echo 1.0 || echo 0.0),
        "score": ${BWD_SCORE}
    },
    "combined": {
        "forward_weight": ${FWD_WEIGHT},
        "backward_weight": ${BWD_WEIGHT},
        "nki_flop_ratio": ${NKI_FLOP_RATIO},
        "raw_score": ${RAW_SCORE},
        "score": ${COMBINED_SCORE}
    }
}
EOF
else
    # Copy the score details file to the standard results.json location
    cp "$SCORE_OUTPUT_FILE" "${LOG_DIR}/results.json"
fi

exit 0