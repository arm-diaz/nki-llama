# Flash Self-Attention Kernel Optimizations Guide for NKI-LLAMA Hackathon

## 🎯 Overview

This guide focuses on working with the sefl-attention kernels provided and optimizing them further using the Neuron Kernel Interface (NKI) compilation on AWS Inferentia/Trainium. This is a perfect starting place for teams who want to learn more about NKI and how kernel optimizations can be applied without having to train or inference components. 

### Instance Requirements
- **Instance Type**: trn1.2xlarge (minimum) or trn1.32xlarge
- **AMI**: Deep Learning AMI Neuron (Ubuntu 22.04) 20250520
  - **us-east-1**: `ami-0e65a95c79775d1b6`
  - **us-west-2**: `ami-0d0a2d26f80b645c2`
- **Storage**: 256GB+ recommended (800GB default in CloudFormation)
- **Neuron SDK**: 2.23.0

### Environment Setup
```bash
# Activate the inference environment
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
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
Usage: flash_fwd[batch_size, kv_heads](q, k, v, seed, config=FlashConfig(...))

**`flash_attn_bwd` - Flash Attention Backward Pass**

- **Purpose:** Backward pass gradient computation for attention
- **Features:** Efficient gradient calculation for Q, K, V with recomputation
- **Optimizations:** Tiled computation, memory-efficient recomputation
- **Usage:** flash_attn_bwd[batch_size, heads](q, k, v, o, dy, lse, seed)

**`fused_self_attn_for_SD_small_head_size` - Stable Diffusion Specialized**

- **Purpose:** Optimized attention for small head sizes (≤128) in Stable Diffusion
- **Features:** Specialized for SD workloads, different tensor layouts
- **Usage:** fused_self_attn_for_SD_small_head_size[batch_size](q, k, v)

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

### Step 2: Modify and Optimiza the Kernel Implementations

Refer to the `attention.py` file for details on the kernel implementation. This is the main file where contestants would want to edit to implement their optimization before testing the kernels. 

### Step 3: Run the Flash Self-Attention Kernel Unit Tests
```bash
# Run the unit tests
cd ~/nki-llama/src/self-attention/tests

# Run all forward and backward tests with full verbosity
pytest test_flash_attn_*.py -v -s

# Run specific test suite
pytest test_flash_attn_fwd.py -v -s
pytest test_flash_attn_bwd.py -v -s

# Performance tests only
pytest -k "perf" -v -s
# Numerical accuracy tests only  
pytest -k "numerical" -v -s
# Simulation tests only
pytest -m simulation -v -s
```

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

### Custom Test Execution
```
# Run with maximum verbosity and detailed tracebacks
pytest test_flash_attn_fwd_verbose.py -v -s --tb=long

# Run specific parameter combinations
pytest test_flash_attn_fwd_verbose.py::TestAttention::test_flash_attn_fwd_perf[1-6-32768-32768-96-bfloat16-True-True-True-2048-3-False-87000000000] -v -s

# Stop on first failure for debugging
pytest test_flash_attn_fwd_verbose.py -v -s -x

# Run with timing information
pytest test_flash_attn_fwd_verbose.py -v -s --durations=10

# Capture output to file
pytest test_flash_attn_fwd_verbose.py -v -s > test_results.log 2>&1
```

## 🔧 Troubleshooting

### Common Issues

#### Performance Test Failures:
- Check hardware availability and configuration
- Verify expected latency thresholds are appropriate for your hardware
- Review memory usage estimates for resource constraint

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
