# Flash Self-Attention Kernel Optimizations Guide for NKI-LLAMA Hackathon

## 🎯 Overview

This guide focuses on working with the self-attention kernels provided and optimizing them further using the Neuron Kernel Interface (NKI) compilation on AWS Inferentia/Trainium. This is a perfect starting place for teams who want to learn more about NKI and how kernel optimizations can be applied without having to train or inference components.

### Instance Requirements
- **Instance Type**: trn1.2xlarge (minimum) or trn1.32xlarge
- **AMI**: Deep Learning AMI Neuron (Ubuntu 22.04) 20250520
  - **us-east-1**: `ami-0e65a95c79775d1b6`
  - **us-west-2**: `ami-0d0a2d26f80b645c2`
- **Storage**: 256GB+ recommended (800GB default in CloudFormation)
- **Neuron SDK**: 2.23.0

### Environment Setup
```bash
# Activate the self-attention environment
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
```

## 🚀 Deployment

Deploy the NKI-LLAMA training environment using AWS CloudFormation with one click:

| AWS Region | Launch CloudFormation Stack |
|:-----------|:----------------------------|
| us-east-1 (N. Virginia) |<a href="https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=hackathon&templateURL=" target="_blank">Launch stack</a> |
| us-west-2 (Oregon) |<a href="https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=hackathon&templateURL=" target="_blank">Launch stack</a> |

**Note:** Only us-east-1 and us-west-2 regions support Trainium (trn1) instances with the required Neuron AMIs.

### Deployment Steps

1. **Download the CloudFormation template**: 
   - Click here to download: [deployment.yaml](../deployment/deployment.yaml)

2. **Click** on one of the CloudFormation Console links above for your preferred region.

3. **Upload the template**:
   - Choose **Upload a template file**
   - Click **Choose file** and select the downloaded `deployment.yaml`
   - Click **Next**

4. **Configure the stack:**
   - **Stack name**: Keep default or customize (e.g., `nki-llama-attention`)
   - **KeyPairOption**: Choose `use-existing` (recommended - create key in EC2 console first)
   - **ExistingKeyPairName**: Select your key from dropdown (see note below)
   - **Ec2InstanceType**: Default: `trn1.32xlarge` - can be changed to use `trn1.2xlarge`
   - Click **Next**
   
   **Note**: For easy key download, first create a key pair in EC2 → Key Pairs → Create key pair, download it, then return here and select it from the dropdown.

5. **Configure stack options**: Leave all values as default and click **Next**

6. **Review and create:**
   - Check the box: "I acknowledge that AWS CloudFormation might create IAM resources"
   - Click **Create stack**
   - Stack creation takes ~5-10 minutes

7. **Access your instance:**
   - Go to CloudFormation → Select your stack → **Outputs** tab
   - Copy the **SSHCommand** value
   - If you created a new key, download it from EC2 → Key Pairs
   - Connect: `ssh -i <your-key.pem> ubuntu@<instance-ip>`

### Post-Deployment Setup

Once connected to your instance:

```bash
# Repository is pre-cloned
cd ~/nki-llama

# Install dependencies
chmod +x install.sh
./install.sh
```

## 🎮 Using the NKI-LLAMA CLI

The repository includes a unified command-line interface that simplifies all operations. You can use either the CLI commands or run the scripts directly.

### Option 1: Using NKI-LLAMA CLI (Recommended)

```bash
# Once connected to your instance
cd ~/nki-llama

# View all self-attention commands
./nki-llama help

# Run interactive setup wizard
./nki-llama setup
```

**Self-Attention CLI Commands:**
- `./nki-llama self-attention benchmark` - Run comprehensive benchmarks
- `./nki-llama self-attention test` - Run all tests
- `./nki-llama self-attention test forward` - Run forward pass tests only
- `./nki-llama self-attention test backward` - Run backward pass tests only
- `./nki-llama self-attention run <script>` - Run a specific script

**Key Benefits of Using CLI:**
- Automatic environment detection and activation guidance
- Built-in tmux recommendations for long operations
- Integrated logging to `logs/` directory
- Consistent error handling and reporting

### Option 2: Direct Script Execution

If you prefer to run scripts directly:

```bash
# Navigate to scripts directory
cd ~/nki-llama/src/self-attention/scripts

# Run the comprehensive benchmark script
./self-attention_benchmark.sh 

# Run specific test suite
pytest ../tests/test_flash_attn_fwd.py -v -s
pytest ../tests/test_flash_attn_bwd.py -v -s
```

## 📁 File Overview

### Core Test Files

| File | Description | Purpose |
|------|-------------|---------|
| `test_flash_attn_fwd.py` | Forward pass tests | Performance + numerical validation for forward attention |
| `test_flash_attn_bwd.py` | Backward pass tests | Performance + numerical validation for backward attention |

### Kernel Implementation Files
| File | Description | Key Functions |
|------|-------------|---------|
| `attention.py` | Core NKI Kernel implementation | `flash_fwd, flash_attn_bwd, fused_self_attn_for_SD_small_head_size` |
| `FlashConfig` | Configuration dataclass | Performance tuning parameters |

### Kernel Functions Overview

**`flash_fwd` - Flash Attention Forward Pass**

- **Purpose:** Optimized forward attention computation with tiling and memory efficiency
- **Features:** Causal masking, mixed precision, dropout, GQA/MQA support, logit bias
- **Optimizations:** Memory tiling, recomputation, SBUF management
- **Usage:** `flash_fwd[batch_size, kv_heads](q, k, v, seed, config=FlashConfig(...))`

**`flash_attn_bwd` - Flash Attention Backward Pass**

- **Purpose:** Backward pass gradient computation for attention
- **Features:** Efficient gradient calculation for Q, K, V with recomputation
- **Optimizations:** Tiled computation, memory-efficient recomputation
- **Usage:** `flash_attn_bwd[batch_size, heads](q, k, v, o, dy, lse, seed)`

**`fused_self_attn_for_SD_small_head_size` - Stable Diffusion Specialized**

- **Purpose:** Optimized attention for small head sizes (≤128) in Stable Diffusion
- **Features:** Specialized for SD workloads, different tensor layouts
- **Usage:** `fused_self_attn_for_SD_small_head_size[batch_size](q, k, v)`

## 🚀 Quick Start

### Step 1 (OPTIONAL): Clone and Setup

**Please skip this step when deploying the infrastructure with cloudformation**

```bash
# Clone the repository
git clone https://github.com/aws-neuron/nki-llama.git
cd nki-llama

# Install dependencies
chmod +x install.sh
./install.sh
```

### Step 2: Modify and Optimize the Kernel Implementations

Refer to the `attention.py` file for details on the kernel implementation. This is the main file where contestants would want to edit to implement their optimization before testing the kernels.

### Step 3: Run the Flash Self-Attention Kernel Unit Tests

#### Using NKI-LLAMA CLI (Recommended)

```bash
# The CLI will check your environment and guide you if needed
cd ~/nki-llama

# Use tmux for benchmarking (recommended)
tmux new -s self-attention

# Activate python environment
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Run comprehensive benchmarks
./nki-llama self-attention benchmark

# Run all tests
./nki-llama self-attention test

# Run specific test types
./nki-llama self-attention test forward   # Forward pass only
./nki-llama self-attention test backward  # Backward pass only

# Detach from tmux with Ctrl+B, D
# Reattach with: tmux attach -t self-attention
```

**Environment Handling:**
The CLI will automatically:
- Detect if you're in the correct virtual environment
- Provide the exact activation command if needed
- Suggest tmux for long operations
- Log all output to `logs/nki-llama_<timestamp>.log`

Example CLI output when environment is not active:
```
❌ No virtual environment active
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  Self-attention environment required
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please activate the environment first:
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
```

#### Using Direct Script Execution

```bash
# Activate the environment manually
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Run the unit tests
cd ~/nki-llama/src/self-attention/scripts

# Use tmux for long operations
tmux new -s benchmark

# Run the comprehensive benchmark script
./self-attention_benchmark.sh 

# Run specific test suite
pytest ../tests/test_flash_attn_fwd.py -v -s
pytest ../tests/test_flash_attn_bwd.py -v -s
```

### Step 4: Understand the Scoring Mechanism

The benchmark calculates a combined score based on the following formula:

```
final_score = accuracy * latency_improvement * throughput_improvement * (1.0 + nki_flop_ratio)
```

Where:
- `accuracy`: Binary value (1.0 or 0.0) indicating if numerical tests pass
- `latency_improvement`: Ratio of baseline latency to measured latency
- `throughput_improvement`: Inversely proportional to latency (higher is better)
- `nki_flop_ratio`: Ratio of operations executed on NKI hardware (hardware utilization)

The NKI FLOP ratio is automatically calculated based on the kernel characteristics, considering:
- Matrix multiplication operations (highly accelerated on NKI)
- Softmax operations (partially accelerated)
- Batch size, sequence length, and head dimension effects on hardware utilization

This scoring mechanism rewards both correctness and performance improvements, with a bonus for efficient hardware utilization.

## 🧪 Test Categories

### Performance Tests (`test_*_perf`)

**Purpose**: Validate that Flash Attention kernels meet latency requirements under various configurations.

**What they test:**
- Execution latency across different percentiles (P50, P90, P95, P99)
- Memory usage efficiency
- Performance scaling with sequence length and batch size

**Example output:**
```
📈 PERFORMANCE METRICS:
   P50 Latency: 12,500,000 ns (0.013s) ✅ PASS
   P90 Latency: 15,200,000 ns (0.015s) ✅ PASS
   P95 Latency: 16,800,000 ns (0.017s) ✅ PASS
   Expected:   15,100,000,000 ns (15.100s)

💾 MEMORY USAGE ESTIMATES:
   Q tensor:     3072.00 MB
   K tensor:     3072.00 MB
   V tensor:     3072.00 MB
   Total Input:  9216.00 MB
   Est. Peak:    18432.00 MB (2x for intermediate)
   
🔢 NKI FLOP RATIO: 0.8734
   Calculated NKI FLOP ratio represents the percentage of operations
   that can be accelerated by the NKI hardware.
```

**Performance Metrics JSON:**
The benchmark generates a detailed JSON file with accumulated metrics:
```json
{
  "timestamp": "2025-07-14T18:45:23Z",
  "forward": {
    "latency": 12500000,
    "base_latency": 15100000000,
    "latency_improvement": 1208.00,
    "throughput_improvement": 1208.00,
    "numerical_accuracy": 1.0,
    "score": 1459264.00
  },
  "backward": {
    "latency": 41482,
    "base_latency": 117000,
    "latency_improvement": 2.82,
    "throughput_improvement": 2.82,
    "numerical_accuracy": 1.0,
    "score": 7.95
  },
  "combined": {
    "forward_weight": 0.4,
    "backward_weight": 0.6,
    "nki_flop_ratio": 0.87,
    "raw_score": 583710.37,
    "score": 1091538.39
  }
}
```

### Numerical Accuracy Tests (`test_*_numerical`)

**Purpose**: Ensure computational accuracy by comparing Flash Attention outputs against reference CPU implementations.

**What they test:**
- Numerical correctness within tolerance (1e-2)
- Forward pass: Output tensors and LSE (Log-Sum-Exp) values
- Backward pass: Gradient tensors (dQ, dK, dV)
- Cross-validation between hardware and simulation modes

**Example output:**
```
📊 dQ Gradient Comparison:
   Max absolute difference:  0.000847
   Mean absolute difference: 0.000234
   Mean relative error:      0.001245
   Tolerance:                0.01
   Result: ✅ PASS

🔬 NUMERICAL VERIFICATION:
   Flash Output vs Reference CPU:
   ✅ Output tensor: PASS (max_diff: 0.00234)
   ✅ LSE tensor: PASS (max_diff: 0.00156)
   🎉 All numerical checks passed!
```

## 🛠️ Advanced Usage

### Custom Test Execution with NKI-LLAMA CLI

```bash
# Run specific scripts with the CLI
./nki-llama self-attention run custom_benchmark.sh

# Pass additional pytest arguments through CLI
./nki-llama self-attention test forward -k "test_flash_attn_fwd_perf" --tb=short

# Run with verbose output
./nki-llama self-attention test all -v -s --durations=10
```

### Direct pytest Execution
```bash
# Run with maximum verbosity and detailed tracebacks
pytest test_flash_attn_fwd.py -v -s --tb=long

# Run specific parameter combinations
pytest test_flash_attn_fwd.py::TestAttention::test_flash_attn_fwd_perf[1-6-32768-32768-96-bfloat16-True-True-True-2048-3-False-87000000000] -v -s

# Stop on first failure for debugging
pytest test_flash_attn_fwd.py -v -s -x

# Run with timing information
pytest test_flash_attn_fwd.py -v -s --durations=10

# Capture output to file
pytest test_flash_attn_fwd.py -v -s > test_results.log 2>&1
```

## 🔧 Troubleshooting

### Using NKI-LLAMA CLI for Diagnostics

```bash
# Check overall status
./nki-llama status

# Check self-attention specific status
./nki-llama self-attention status

# View logs
ls -la logs/nki-llama_*.log
tail -f logs/nki-llama_*.log
```

### Common Issues

#### Environment Not Active:
The CLI will detect this and show:
```
❌ No virtual environment active
Please activate the environment first:
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
```

#### Performance Test Failures:
- Check hardware availability and configuration
- Verify expected latency thresholds are appropriate for your hardware
- Review memory usage estimates for resource constraints

#### Numerical Test Failures:
- Increase tolerance if needed for specific hardware characteristics
- Check tensor shapes and data types match expectations
- Verify reference implementation correctness

#### Simulation Mode Issues:
- Ensure simulation environment is properly configured
- Check that all required kernels are available in simulation

### Expected Test Outcomes

**Performance Tests:**
- ✅ Pass: Latency within expected bounds
- ❌ Fail: Latency exceeds thresholds (check hardware load, configuration)
- ⚠️ xfail: Known issues (marked with ticket numbers)

**Numerical Tests:**
- ✅ Pass: All gradients/outputs within tolerance
- ❌ Fail: Numerical differences exceed tolerance (check implementation)

**Test Status:**
- 🎉 PASSED: All metrics within acceptable ranges
- 💥 FAILED: One or more metrics exceeded thresholds
- ⚠️ xfail: Expected failure due to known issues
- ❓ Cannot Determine: Missing metric data (API issues)

## 📊 Complete Workflow Example

### Using NKI-LLAMA CLI (Recommended)
```bash
# Start a new tmux session
tmux new -s hackathon-attention

# Navigate to repository
cd ~/nki-llama

# If environment is not active, the CLI will tell you to:
# source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Modify kernel implementation
nano src/self-attention/attention.py

# Run benchmarks
./nki-llama self-attention benchmark

# Run specific tests
./nki-llama self-attention test forward
./nki-llama self-attention test backward

# Check logs
tail -f logs/nki-llama_*.log

# Detach from tmux: Ctrl+B, D
# Reattach later: tmux attach -t hackathon-attention
```

### Using Direct Scripts
```bash
# Start tmux
tmux new -s benchmark

# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Navigate to scripts
cd ~/nki-llama/src/self-attention/scripts

# Run benchmark
./self-attention_benchmark.sh

# Run individual tests
cd ../tests
pytest test_flash_attn_fwd.py -v -s
pytest test_flash_attn_bwd.py -v -s
```

## 📚 Resources

- [NKI Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
- [NKI Samples Repository](https://github.com/aws-neuron/nki-samples)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)

---

**Pro Tips:**
- Always use the NKI-LLAMA CLI for better environment management
- Run benchmarks in tmux to avoid disconnection issues
- Check `./nki-llama status` regularly to monitor system health
- The CLI logs everything to `logs/` for later analysis