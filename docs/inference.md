# Inference with NKI Compilation Guide for NKI-LLAMA Hackathon

## 🎯 Overview

This guide focuses on optimizing inference performance using Neuron Kernel Interface (NKI) compilation on AWS Inferentia/Trainium. Perfect for teams wanting to achieve maximum inference throughput and minimal latency without the training component.

## 📋 Prerequisites

### Instance Requirements
- **Instance Type**: trn1.32xlarge (recommended) or trn1.2xlarge (minimum)
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

Deploy the NKI-LLAMA inference environment using AWS CloudFormation with one click:

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
   - **Stack name**: Keep default or customize (e.g., `nki-llama-inference`)
   - **KeyPairOption**: Choose `use-existing` (recommended - create key in EC2 console first)
   - **ExistingKeyPairName**: Select your key from dropdown (see note below)
   - **Ec2InstanceType**: Keep default `trn1.32xlarge`
   - **VpcOption**: Choose `create-new` or select existing VPC
   - Click **Next**
   
   **Note**: For easy key download, first create a key pair in EC2 → Key Pairs → Create key pair, download it, then return here and select it from the dropdown. Alternatively, choose `none` to use SSM Session Manager without keys.

5. **Configure stack options**: Leave all values as default and click **Next**

6. **Review and create:**
   - Check the box: "I acknowledge that AWS CloudFormation might create IAM resources"
   - Click **Create stack**
   - Stack creation takes ~5-10 minutes

7. **Access your instance:**
   - Go to CloudFormation → Select your stack → **Outputs** tab
   - Use **SSHCommand** for SSH access or **EC2InstanceId** for SSM
   - For SSM: `aws ssm start-session --target <instance-id>`

### Post-Deployment Setup

Once connected to your instance:

```bash
# Repository is pre-cloned
cd ~/nki-llama

# Install dependencies
chmod +x install.sh
./install.sh

# Configure environment
nano .env  # Add your HF_TOKEN and inference settings

# Activate inference environment
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
```

### 🎮 Using the NKI-LLAMA CLI

The repository includes a unified command-line interface that simplifies all operations:

```bash
# Once connected to your instance
cd ~/nki-llama

# View all available commands
./nki-llama help

# Run interactive setup wizard
./nki-llama setup
```

**Key Commands:**
- `./nki-llama setup` - Interactive setup wizard with environment guidance
- `./nki-llama status` - Check system health and compilation cache
- `./nki-llama clean` - Clean artifacts and cache if needed

**Pro Tips:**
- Always run the setup wizard first: `./nki-llama setup`
- Use `tmux` for long operations (the CLI will remind you)
- Check `./nki-llama status` if you encounter issues
- The CLI automatically guides you to the correct virtual environment

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

# Configure environment
nano .env  # Add your HF_TOKEN and inference settings
```

### Step 2: Download Model
```bash
# Download the model using the CLI
./nki-llama inference download

# Or manually download a specific model
cd ~/models
huggingface-cli download --token YOUR_TOKEN meta-llama/Meta-Llama-3-8B --local-dir /home/ubuntu/models/llama-3-8b
```

### Step 3: Run Benchmark with NKI Compilation
```bash
# Use tmux for long-running compilation
tmux new -s benchmark

# Run benchmark (includes NKI compilation on first run)
./nki-llama inference benchmark
```

## 🔧 NKI Kernel Implementation

### Understanding NKI Optimizations

NKI (Neuron Kernel Interface) allows you to write custom, highly optimized kernels for Neuron devices. Key targets for optimization:

1. **RMSNorm** - Layer normalization operations
2. **Attention mechanisms** - Multi-head attention computation
3. **Linear transformations** - Matrix multiplications
4. **Activation functions** - GELU, SiLU, etc.

### Example: Implementing NKI RMSNorm

```python
import neuron_kernel_interface as nki
import torch.nn as nn

@nki.jit
def nki_rmsnorm_kernel(input_tensor, weight, epsilon):
    """
    Optimized RMSNorm implementation using NKI
    """
    # Get tensor dimensions
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    hidden_size = input_tensor.shape[2]
    
    # Allocate output tensor
    output = nki.tensor(shape=input_tensor.shape, dtype=input_tensor.dtype)
    
    # Compute RMS normalization
    # Your NKI implementation here
    # ...
    
    return output

# Modify the model to use NKI kernel
class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, nki_enabled=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.nki_enabled = nki_enabled
    
    def forward(self, hidden_states):
        if self.nki_enabled:
            return nki_rmsnorm_kernel(hidden_states, self.weight, self.variance_epsilon)
        # Fallback to standard implementation
        return standard_rmsnorm(hidden_states, self.weight, self.variance_epsilon)
```

### Implementing Additional NKI Kernels

#### 1. Attention Kernel
```python
@nki.jit
def nki_attention_kernel(q, k, v, mask=None):
    """
    Optimized attention computation
    """
    # Implement scaled dot-product attention
    # with NKI optimizations
    pass
```

#### 2. Linear Layer Kernel
```python
@nki.jit
def nki_linear_kernel(input, weight, bias=None):
    """
    Optimized linear transformation
    """
    # Implement matrix multiplication
    # with optional bias addition
    pass
```

## 📊 Benchmarking Process

### Running Benchmarks

```bash
# Full benchmark with default settings
./nki-llama inference benchmark

# Benchmark with custom sequence length
./nki-llama inference benchmark --seq-len 2048

# Clear cache and re-benchmark
./nki-llama inference benchmark --clear-cache
```

### Direct Benchmark Execution
For more control over benchmarking parameters:

```bash
cd src/inference
python main.py \
    --mode evaluate_all \
    --seq-len 1024 \
    --batch-size 1 \
    --enable-nki \
    --num-prompts 25
```

## 🎯 Score Calculation (Inference Only)

After benchmarking completes, calculate your performance score:

```bash
python /home/ubuntu/nki-llama/src/handler.py \
    --inference-results /home/ubuntu/nki-llama/src/inference/benchmark_inference.json \
    --output inference_score.json \
    --inference-weight 1.0 \
    --hw-backend trn1 \
    --calculate-score \
    --detailed \
    --verbose
```

The inference score evaluates:
- **Latency reduction**: Time to First Token (TTFT) improvement
- **Throughput increase**: Tokens/second improvement
- **NKI coverage**: Percentage of FLOPs using NKI kernels

## 🔍 Profiling and Optimization

### Using Neuron Profiler

```bash
# Enable profiling during benchmark
export NEURON_PROFILE=1
export NEURON_PROFILE_CONFIG=profile.json

# Create profile configuration
cat > profile.json << EOF
{
    "capture": {
        "enabled": true,
        "output_dir": "./profiles",
        "duration_ms": 10000
    }
}
EOF

# Run benchmark with profiling
./nki-llama inference benchmark

# Analyze results
neuron-profile view ./profiles/profile_*.neff
```

### Key Optimization Targets

1. **Memory Access Patterns**
   - Optimize data layout for Neuron memory hierarchy
   - Minimize HBM bandwidth usage
   - Use efficient tiling strategies

2. **Compute Efficiency**
   - Maximize tensor core utilization
   - Fuse operations where possible
   - Eliminate redundant computations

3. **Pipeline Optimization**
   - Overlap compute and memory operations
   - Optimize kernel launch overhead
   - Efficient synchronization

## 🛠️ Advanced NKI Techniques

### 1. Kernel Fusion
Combine multiple operations into a single kernel:

```python
@nki.jit
def nki_fused_attention_norm(q, k, v, norm_weight, epsilon):
    """
    Fused attention + normalization kernel
    """
    # Compute attention
    attn_output = nki_attention_kernel(q, k, v)
    
    # Apply normalization in the same kernel
    normalized = nki_rmsnorm_kernel(attn_output, norm_weight, epsilon)
    
    return normalized
```

### 2. Tiling Strategies
Optimize for Neuron's memory hierarchy:

```python
@nki.jit
def nki_tiled_matmul(a, b, tile_size=128):
    """
    Tiled matrix multiplication for better cache usage
    """
    # Implement tiled algorithm
    # optimized for Neuron architecture
    pass
```

### 3. Asynchronous Execution
Leverage Neuron's async capabilities:

```python
# Enable async execution in your kernels
@nki.jit(async_launch=True)
def nki_async_kernel(...):
    pass
```

## 📈 Performance Monitoring

### Real-time Monitoring
```bash
# Monitor device utilization
neuron-top

# Watch compilation progress
tail -f logs/nki-llama_*.log

# Check benchmark results
cat src/inference/benchmark_inference.json | jq
```

### Key Metrics
- **TTFT (Time to First Token)**: Target <100ms
- **Throughput**: Target >1000 tokens/sec
- **Device Utilization**: Target >90%
- **Memory Bandwidth**: Monitor for bottlenecks

## 🐛 Troubleshooting

### Common Issues

#### Compilation Cache Errors
```bash
# Clear the cache
./nki-llama clean
# or
rm -rf ~/neuron_cache/*
```

#### Out of Memory During Compilation
```bash
# Reduce parallelism
export NEURON_COMPILE_THREADS=4
```

#### Kernel Launch Failures
- Check tensor dimensions match kernel expectations
- Verify data types are supported
- Enable debug mode: `export NEURON_DEBUG=1`

## 🏆 Optimization Strategies

### 1. Target Hot Spots
Focus on operations that consume most time:
- Attention computation (usually 30-40% of time)
- Linear layers (20-30%)
- Normalization (10-15%)

### 2. Incremental Optimization
- Start with one kernel (e.g., RMSNorm)
- Validate correctness
- Measure improvement
- Move to next kernel

## 📊 Benchmark Configuration

### Custom Prompt Testing
Create your own prompts for testing:

```bash
# Edit prompts.txt
nano ./data/prompts.json
```

### Batch Processing
Test different batch sizes:

```bash
for batch in 1 2 4 8; do
    ./nki-llama inference benchmark --batch-size $batch
done
```

## 🎯 Next Steps

After mastering inference optimization:
1. Document your NKI kernel implementations
2. Create performance comparison charts
3. Consider adding fine-tuning (see [complete-pipeline.md](./complete-pipeline.md))
4. Prepare reasoning benchmarks for additional scoring

## 📚 Example Inference Session

```bash
# Complete workflow example
tmux new -s hackathon-inference

# Setup
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
cd ~/nki-llama

# Download model
./nki-llama inference download

# Run initial benchmark
./nki-llama inference benchmark

# Implement NKI optimizations
nano src/llama.py  # Add your NKI kernels

# Re-benchmark with optimizations
./nki-llama inference benchmark --clear-cache

# Calculate score
python src/handler.py --inference-results benchmark_inference.json --calculate-score

# Start serving (optional)
./nki-llama inference server
```

## 📚 Resources

- [NKI Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
- [NKI Samples Repository](https://github.com/aws-neuron/nki-samples)
- [NKI Autotune Tool](https://github.com/awslabs/nki-autotune)
- [Neuron Profiler Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html)

---

Remember: Focus on implementing high-performance NKI kernels for critical operations. The key to success is identifying and optimizing the bottlenecks in your model's inference pipeline!