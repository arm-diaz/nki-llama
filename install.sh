#!/bin/bash
# install.sh - NKI-LLAMA installation script

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Banner
echo -e "${CYAN}"
cat << 'EOF'
    _   __ __ __ ____       __    __       ___    __  ___    ___ 
   / | / // //_//  _/      / /   / /      /   |  /  |/  /   /   |
  /  |/ // ,<   / /______ / /   / /      / /| | / /|_/ /   / /| |
 / /|  // /| |_/ /_______/ /___/ /___   / ___ |/ /  / /   / ___ |
/_/ |_//_/ |_/___/       /_____/_____/  /_/  |_/_/  /_/   /_/  |_|
                                                             
EOF
echo -e "${NC}"

echo -e "${BOLD}NKI-LLAMA Installation${NC}"
echo -e "===================="
echo

# Get installation directory
INSTALL_DIR="${1:-$(pwd)}"
echo -e "${BLUE}Installing to: ${INSTALL_DIR}${NC}"

# Create directories
echo -e "\n${YELLOW}Creating directory structure...${NC}"
mkdir -p "${INSTALL_DIR}/logs/benchmarks"
mkdir -p "${INSTALL_DIR}/src/inference/scripts"
mkdir -p "${INSTALL_DIR}/src/fine-tune/scripts"

# Make scripts executable
echo -e "${YELLOW}Setting up scripts...${NC}"
chmod +x "${INSTALL_DIR}/nki-llama.sh" 2>/dev/null || true
chmod +x "${INSTALL_DIR}/src/inference/scripts/"*.sh 2>/dev/null || true
chmod +x "${INSTALL_DIR}/src/fine-tune/scripts/"*.sh 2>/dev/null || true

# Create symlink for easier access
if [[ -f "${INSTALL_DIR}/nki-llama.sh" ]] && [[ ! -f "${INSTALL_DIR}/nki-llama" ]]; then
    ln -s "${INSTALL_DIR}/nki-llama.sh" "${INSTALL_DIR}/nki-llama"
    echo -e "${GREEN}✓ Created nki-llama symlink${NC}"
fi

# Copy example environment file
if [[ ! -f "${INSTALL_DIR}/.env" ]]; then
    if [[ -f "${INSTALL_DIR}/.env.example" ]]; then
        cp "${INSTALL_DIR}/.env.example" "${INSTALL_DIR}/.env"
        echo -e "${GREEN}✓ Created .env file from example${NC}"
        echo -e "${YELLOW}  Please edit .env and add your HF_TOKEN${NC}"
    else
        # Create a basic .env file if no example exists
        cat > "${INSTALL_DIR}/.env" << 'EOF'
# NKI-LLAMA Configuration
HF_TOKEN=
MODEL_ID_8B=meta-llama/Meta-Llama-3-8B
MODEL_NAME_8B=llama-3-8b
MODEL_ID_1B=meta-llama/Llama-3.2-1B
MODEL_NAME_1B=llama-3-2-1b
TENSOR_PARALLEL_SIZE=8
INFERENCE_PORT=8080
MAX_MODEL_LEN=2048
MAX_NUM_SEQS=4
DATASET_NAME=databricks/databricks-dolly-15k
EOF
        echo -e "${GREEN}✓ Created default .env file${NC}"
        echo -e "${YELLOW}  Please edit .env and add your HF_TOKEN${NC}"
    fi
fi

# Check for Neuron environments
echo -e "\n${BOLD}Checking Neuron environments...${NC}"
MISSING_ENV=false

if [[ -d "/opt/aws_neuronx_venv_pytorch_2_6" ]]; then
    echo -e "${GREEN}✓ Fine-tuning environment found${NC}"
else
    echo -e "${RED}✗ Fine-tuning environment not found${NC}"
    echo -e "  ${YELLOW}Expected at: /opt/aws_neuronx_venv_pytorch_2_6${NC}"
    MISSING_ENV=true
fi

if [[ -d "/opt/aws_neuronx_venv_pytorch_2_6_nxd_inference" ]]; then
    echo -e "${GREEN}✓ Inference environment found${NC}"
else
    echo -e "${RED}✗ Inference environment not found${NC}"
    echo -e "  ${YELLOW}Expected at: /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference${NC}"
    MISSING_ENV=true
fi

# Check for tmux
echo -e "\n${BOLD}Checking system dependencies...${NC}"
if command -v tmux &> /dev/null; then
    TMUX_VERSION=$(tmux -V | cut -d' ' -f2)
    echo -e "${GREEN}✓ tmux ${TMUX_VERSION} found${NC}"
else
    echo -e "${YELLOW}⚠ tmux not found - recommended for long-running operations${NC}"
    echo -e "  Install with: ${CYAN}sudo apt-get install tmux${NC}"
fi

# Check for neuron-ls
if command -v neuron-ls &> /dev/null; then
    echo -e "${GREEN}✓ Neuron SDK tools found${NC}"
else
    echo -e "${YELLOW}⚠ neuron-ls not found - Neuron SDK may not be installed${NC}"
fi

# Verify configuration file
if [[ -f "${INSTALL_DIR}/nki-llama.config" ]]; then
    echo -e "${GREEN}✓ Configuration file found${NC}"
else
    echo -e "${RED}✗ nki-llama.config not found!${NC}"
    echo -e "  This file is required for operation"
fi

# Check if running on correct instance
if [[ -f /sys/devices/virtual/dmi/id/product_name ]]; then
    INSTANCE_TYPE=$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo "unknown")
    if [[ "$INSTANCE_TYPE" == *"trn1"* ]]; then
        echo -e "${GREEN}✓ Running on Trainium instance (${INSTANCE_TYPE})${NC}"
    else
        echo -e "${YELLOW}⚠ Not running on Trainium instance${NC}"
        echo -e "  Current: ${INSTANCE_TYPE}"
        echo -e "  Recommended: trn1.32xlarge"
    fi
fi

# Installation summary
echo -e "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [[ "$MISSING_ENV" == "true" ]]; then
    echo -e "${YELLOW}⚠️  Installation completed with warnings${NC}"
    echo -e "\nSome Neuron environments are missing. This is expected if you're not on"
    echo -e "a Neuron-enabled instance or haven't installed the Neuron SDK yet."
else
    echo -e "${GREEN}✅ Installation complete!${NC}"
fi
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Next steps
echo -e "\n${BOLD}Next steps:${NC}"
echo -e "1. ${YELLOW}Configure:${NC} Edit ${CYAN}${INSTALL_DIR}/.env${NC} with your settings"
echo -e "   • Add your Hugging Face token (HF_TOKEN)"
echo -e "   • Adjust model and training parameters as needed"
echo

if [[ "$MISSING_ENV" == "true" ]]; then
    echo -e "2. ${YELLOW}Install Neuron SDK:${NC} Follow AWS documentation to install Neuron SDK"
    echo -e "   This will create the required virtual environments"
    echo
    echo -e "3. ${YELLOW}Activate environment:${NC}"
else
    echo -e "2. ${YELLOW}Activate environment:${NC}"
fi
echo -e "   Fine-tuning: ${CYAN}source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate${NC}"
echo -e "   Inference:   ${CYAN}source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate${NC}"
echo

if [[ "$MISSING_ENV" == "true" ]]; then
    echo -e "4. ${YELLOW}Get started:${NC}"
else
    echo -e "3. ${YELLOW}Get started:${NC}"
fi
echo -e "   • Run ${CYAN}./nki-llama setup${NC} for interactive setup"
echo -e "   • Use ${CYAN}./nki-llama help${NC} to see all commands"
echo -e "   • Check ${CYAN}./nki-llama status${NC} to verify installation"
echo

echo -e "${BOLD}Workflow overview:${NC}"
echo -e "   Fine-tuning → NKI Compilation → vLLM Inference → Agent Development"
echo

if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}💡 Tip:${NC} Install tmux for better experience with long-running tasks:"
    echo -e "   ${CYAN}sudo apt-get update && sudo apt-get install tmux${NC}"
    echo
fi

echo -e "${BLUE}Documentation:${NC} See README.md for detailed usage instructions"
echo