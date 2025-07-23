#!/bin/bash
# download-model.sh - Download model from Hugging Face

set -euo pipefail

ENV_FILE=".env"
KEY_1="NEURON_RT_NUM_CORES"
KEY_2="TENSOR_PARALLEL_SIZE"

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../nki-llama.config"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Downloading model from Hugging Face...${NC}"

# Determine the instance type
source "${SCRIPT_DIR}/../../../src/inference/scripts/instance_type.sh"

# Check if MODEL_NAME is provided as an argument or environment variable
if [ -n "${1:-}" ]; then
    MODEL_NAME="$1"
elif [ -z "${MODEL_NAME:-}" ]; then
    echo -e "${RED}Error: MODEL_NAME not specified!${NC}"
    echo "Usage: $0 <MODEL_NAME>"
    echo "   or: export MODEL_NAME=<model_name> && $0"
    echo ""
    
    # Provide recommendations based on instance type
    if [ "$EC2_INSTANCE_TYPE" == "trn1.2xlarge" ]; then
        echo "Recommended model for $EC2_INSTANCE_TYPE:"
        echo "  - llama-3-2_1b"
        echo ""
        echo "Example: $0 llama-3-2_1b"
    elif [ "$EC2_INSTANCE_TYPE" == "trn1.32xlarge" ]; then
        echo "Recommended model for $EC2_INSTANCE_TYPE:"
        echo "  - llama-3-1_8b"
        echo ""
        echo "Example: $0 llama-3-1_8b"
    else
        echo "Unsupported instance type: $EC2_INSTANCE_TYPE"
        echo "This script requires either trn1.2xlarge or trn1.32xlarge"
    fi
    exit 1
fi

# Check if MODEL_ID is set
if [ -z "${MODEL_ID:-}" ]; then
    echo -e "${RED}Error: MODEL_ID environment variable is not set!${NC}"
    echo "Please set MODEL_ID to the Hugging Face model identifier"
    echo ""
    echo "Examples:"
    echo "  For llama-3-2_1b: export MODEL_ID=meta-llama/Llama-3.2-1B"
    echo "  For llama-3-1_8b: export MODEL_ID=meta-llama/Meta-Llama-3-8B"
    exit 1
fi

# Check HF token
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo -e "${YELLOW}HF_TOKEN not set${NC}"
    echo "Get a token at: https://huggingface.co/settings/tokens"
    read -p "Enter your Hugging Face token: " HF_TOKEN
    if [[ -z "$HF_TOKEN" ]]; then
        echo -e "${RED}Error: HF_TOKEN is required${NC}"
        exit 1
    fi
fi

# Ensure huggingface-cli is installed
pip install -q huggingface_hub[cli]

# Ensure transformers < 4.50 (needed by Neuron hf_adapter)
python - <<'PY'
import subprocess, pkg_resources, sys
req = "4.50.0"
try:
    ver = pkg_resources.get_distribution("transformers").version
except pkg_resources.DistributionNotFound:
    ver = ""
if not ver or pkg_resources.parse_version(ver) >= pkg_resources.parse_version(req):
    print("Installing transformers<%s …" % req)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"transformers<{req}"])
PY

# Configure NeuronCore settings based on instance type
if [ "$EC2_INSTANCE_TYPE" == "trn1.2xlarge" ]; then
    CORE_VALUE="2"
    echo "🚀 Configuring for small instance (2 NeuronCores)..."
    
    # Optional warning if using large model on small instance
    if [[ "$MODEL_NAME" == *"8b"* ]] || [[ "$MODEL_NAME" == *"8B"* ]]; then
        echo -e "${YELLOW}⚠️  Warning: Using a large model (8B) on a small instance (trn1.2xlarge) may cause performance issues${NC}"
    fi
elif [ "$EC2_INSTANCE_TYPE" == "trn1.32xlarge" ]; then
    CORE_VALUE="8"
    echo "🚀 Configuring for large instance (8 NeuronCores)..."
    
    # Optional note if using small model on large instance
    if [[ "$MODEL_NAME" == *"1b"* ]] || [[ "$MODEL_NAME" == *"1B"* ]]; then
        echo -e "${YELLOW}ℹ️  Note: Using a small model (1B) on a large instance (trn1.32xlarge)${NC}"
        echo -e "${YELLOW}   Consider using a larger model for better resource utilization${NC}"
    fi
else
    echo -e "${RED}Error: Unsupported instance type: $EC2_INSTANCE_TYPE${NC}"
    echo "This script requires either trn1.2xlarge or trn1.32xlarge"
    exit 1
fi

# Update NEURON_RT_NUM_CORES in .env file
if grep -q "^${KEY_1}=" "$ENV_FILE" 2>/dev/null; then
    sed -i "s/^${KEY_1}=.*/${KEY_1}=${CORE_VALUE}/" "$ENV_FILE"
else
    echo "${KEY_1}=${CORE_VALUE}" >> "$ENV_FILE"
fi

# Update TENSOR_PARALLEL_SIZE in .env file
if grep -q "^${KEY_2}=" "$ENV_FILE" 2>/dev/null; then
    sed -i "s/^${KEY_2}=.*/${KEY_2}=${CORE_VALUE}/" "$ENV_FILE"
else
    echo "${KEY_2}=${CORE_VALUE}" >> "$ENV_FILE"
fi

echo "Updated .env file with:"
echo "  ${KEY_1}=${CORE_VALUE}"
echo "  ${KEY_2}=${CORE_VALUE}"

# Display configuration summary
echo ""
echo "Configuration Summary:"
echo "  Instance Type: $EC2_INSTANCE_TYPE"
echo "  Model Name: $MODEL_NAME"
echo "  Model ID: $MODEL_ID"
echo "  NeuronCores: $CORE_VALUE"
echo ""

# Create models directory
mkdir -p "$NKI_MODELS"

# Check if model already exists
if [ -d "${NKI_MODELS}/${MODEL_NAME}" ] && [ -n "$(ls -A ${NKI_MODELS}/${MODEL_NAME} 2>/dev/null)" ]; then
    echo -e "${YELLOW}Model already exists at ${NKI_MODELS}/${MODEL_NAME}${NC}"
    read -p "Do you want to re-download it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download. Using existing model."
    else
        echo "Re-downloading model..."
        rm -rf "${NKI_MODELS}/${MODEL_NAME}"
    fi
fi

# Download model if needed
if [ ! -d "${NKI_MODELS}/${MODEL_NAME}" ] || [ -z "$(ls -A ${NKI_MODELS}/${MODEL_NAME} 2>/dev/null)" ]; then
    echo "Downloading ${MODEL_ID} to ${NKI_MODELS}/${MODEL_NAME}"
    huggingface-cli download \
        --token "$HF_TOKEN" \
        "$MODEL_ID" \
        --local-dir "${NKI_MODELS}/${MODEL_NAME}"
fi

# Export variables to environment for other scripts to use
echo ""
echo "Creating model environment file..."

# Create a file to store these variables for other scripts
cat > "${SCRIPT_DIR}/model_env.sh" << EOF
#!/bin/bash
# Auto-generated by download-model.sh
# Contains model environment variables for other scripts

export MODEL_NAME="${MODEL_NAME}"
export MODEL_ID="${MODEL_ID}"
export NEURON_RT_NUM_CORES="${CORE_VALUE}"
export TENSOR_PARALLEL_SIZE="${CORE_VALUE}"
EOF

chmod +x "${SCRIPT_DIR}/model_env.sh"
echo "✅ Created model environment file at: ${SCRIPT_DIR}/model_env.sh"

echo -e "${GREEN}✓ Model downloaded successfully${NC}"
echo "Location: ${NKI_MODELS}/${MODEL_NAME}"

# Save configuration hint
if [[ -z "${HF_TOKEN:-}" ]] && [[ -n "${HF_TOKEN}" ]]; then
    echo ""
    echo "To save your token, add to .env file:"
    echo "HF_TOKEN=$HF_TOKEN"
fi

# Provide next steps
echo ""
echo "Next steps:"
echo "1. Source the model environment: source ${SCRIPT_DIR}/model_env.sh"
echo "2. Run your inference or fine-tuning scripts"