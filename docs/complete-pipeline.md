# Complete Pipeline Guide: Fine-tuning + Inference with NKI

## 🎯 Overview

This guide covers the entire NKI-LLAMA pipeline, combining fine-tuning on AWS Trainium with NKI-optimized inference. This approach maximizes your hackathon score by optimizing both training and inference performance, plus optional reasoning evaluation.

## 📋 Prerequisites

### Instance Requirements
- **Instance Type**: trn1.32xlarge (strongly recommended)
- **AMI**: Deep Learning AMI Neuron (Ubuntu 22.04) 20250520
  - **us-east-1**: `ami-0e65a95c79775d1b6`
  - **us-west-2**: `ami-0d0a2d26f80b645c2`
- **Storage**: 512GB+ (800GB default in CloudFormation for models and datasets)
- **Neuron SDK**: 2.23.0

### Environment Management
Two virtual environments are used:
```bash
# For fine-tuning
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# For inference and benchmarking
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
```

## 🚀 Deployment

Deploy the complete NKI-LLAMA environment using AWS CloudFormation with one click:

| AWS Region | Launch CloudFormation Stack |
|:-----------|:----------------------------|
| us-east-1 (N. Virginia) |<a href="https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=hackathon&templateURL=" target="_blank">Launch stack</a> |
| us-west-2 (Oregon) |<a href="https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=hackathon&templateURL=" target="_blank">Launch stack</a> |

**Note:** Only us-east-1 and us-west-2 regions support Trainium (trn1) instances with the required Neuron AMIs.

### Deployment Steps

1. **Click** on one of the "Launch stack" links above for your preferred region.

2. **Configure the stack:**
   - **Stack name**: Keep default or customize (e.g., `nki-llama-complete`)
   - **KeyPairOption**: Choose `use-existing` (recommended - create key in EC2 console first)
   - **ExistingKeyPairName**: Select your key from dropdown (see note below)
   - **Ec2InstanceType**: Keep default `trn1.32xlarge`
   - **VpcOption**: Keep default `create-new`
   - Click **Next**
   
   **Note**: For easy key download, first create a key pair in EC2 → Key Pairs → Create key pair, download it, then return here and select it from the dropdown.

3. **Configure stack options**: Leave all values as default and click **Next**

4. **Review and create:**
   - Check the box: "I acknowledge that AWS CloudFormation might create IAM resources"
   - Click **Create stack**
   - Stack creation takes ~5-10 minutes

5. **Access your instance:**
   - Go to CloudFormation → Select your stack → **Outputs** tab
   - Note the **EC2InstanceId** and **EC2PublicIP**
   - Connect using your pre-downloaded key or SSM

### Quick Access Commands

```bash
# SSH access (with your pre-created key)
ssh -i ~/Downloads/your-key-name.pem ubuntu@<EC2PublicIP>

# SSM access (no key needed)
aws ssm start-session --target <EC2InstanceId>
```

### Post-Deployment Setup

Once connected:

```bash
# Repository is pre-cloned
cd ~/nki-llama

# Install dependencies
chmod +x install.sh
./install.sh

# Configure environment
nano .env  # Add your HF_TOKEN

# Verify setup
neuron-ls  # Check Neuron devices
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

## 🏃 Complete Workflow

### Step 1: Initial Setup
```bash
# Clone repository
git clone https://github.com/aws-neuron/nki-llama.git
cd nki-llama

# Install and configure
chmod +x install.sh
./install.sh

# Setup environment
nano .env  # Add HF_TOKEN and configure settings
```

### Step 2: Fine-tuning Phase
```bash
# Start tmux session for training
tmux new -s training

# Activate training environment
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Run complete fine-tuning pipeline
./nki-llama finetune all

# IMPORTANT: Note the compile directory from output
# Example: /home/ubuntu/neuron_cache/neuronxcc-2.18.121.0+9e31e41a/MODULE_15329989265349737271+a65e371e
```

### Step 3: Inference Optimization
```bash
# Start new tmux session for inference
tmux new -s inference

# Switch to inference environment
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Download model if not already done
./nki-llama inference download

# Run benchmark with NKI compilation
./nki-llama inference benchmark
```

### Step 4: Reasoning Evaluation (Optional)
```bash
# Start new tmux session for reasoning
tmux new -s reasoning

# Ensure inference environment is active
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

# Run reasoning benchmarks
./nki-llama/src/inference/scripts/reasoning-bench-lm-eval.sh
```

### Step 5: Calculate Combined Score
```bash
# After all components complete
python /home/ubuntu/nki-llama/src/handler.py \
    --config /home/ubuntu/nki-llama/src/fine-tune/neuronx-distributed-training/examples/conf/hf_llama3_8B_SFT_config.yaml \
    --model-config /home/ubuntu/nki-llama/src/fine-tune/configs/model-config/8B_config_llama3-1/config.json \
    --log-file /home/ubuntu/nki-llama/logs/nki-llama_[YOUR_TRAINING_LOG].log \
    --compile-dir [YOUR_COMPILE_DIR_FROM_TRAINING] \
    --inference-results /home/ubuntu/nki-llama/src/inference/benchmark_inference.json \
    --reasoning-results \
    --throughput 2.1 \
    --output complete_benchmark_results.json \
    --training-weight 0.33 \
    --inference-weight 0.33 \
    --reasoning-weight 0.34 \
    --hw-backend trn1 \
    --per-file-scores \
    --calculate-score \
    --detailed \
    --verbose
```

## 🔧 Integrated NKI Optimization Strategy

### Phase 1: Training Optimizations

#### Custom Training Kernels
```python
# Example: NKI-optimized gradient computation
@nki.jit
def nki_gradient_accumulation(gradients, accumulated_grads, scale_factor):
    """
    Optimized gradient accumulation for distributed training
    """
    # Efficient gradient scaling and accumulation
    pass

# Example: NKI-optimized optimizer step
@nki.jit
def nki_adam_step(params, grads, m, v, lr, beta1, beta2, eps):
    """
    Fused Adam optimizer step
    """
    # Implement fused parameter update
    pass
```

#### Training-specific Optimizations
1. **Gradient All-Reduce**: Optimize collective operations
2. **Loss Computation**: Fused loss calculation
3. **Activation Checkpointing**: Memory-efficient training
4. **Mixed Precision**: FP16/BF16 optimizations

### Phase 2: Inference Optimizations

#### Shared Kernel Optimizations
Many kernels can be shared between training and inference:

```python
# Shared RMSNorm implementation
@nki.jit
def nki_rmsnorm_kernel(input_tensor, weight, epsilon, training=False):
    """
    RMSNorm optimized for both training and inference
    """
    # Common normalization logic
    normalized = compute_rmsnorm(input_tensor, weight, epsilon)
    
    if training:
        # Store intermediate values for backward pass
        save_for_backward(input_tensor, normalized)
    
    return normalized

# Shared attention mechanism
@nki.jit
def nki_attention_kernel(q, k, v, mask=None, training=False):
    """
    Multi-head attention for training and inference
    """
    # Implement scaled dot-product attention
    # with different optimizations for each mode
    pass
```

#### Inference-specific Optimizations
1. **KV Cache Management**: Optimize cache operations
2. **Continuous Batching**: Dynamic batch processing
3. **Speculative Decoding**: Parallel token generation
4. **Quantization**: INT8/INT4 inference

## 📊 Performance Monitoring Dashboard

### Unified Monitoring Script
Create a monitoring script to track both phases:

```bash
#!/bin/bash
# monitor.sh

echo "=== NKI-LLAMA Performance Monitor ==="

# Training metrics
if pgrep -f "finetune" > /dev/null; then
    echo "📊 Training Status:"
    tail -n 20 logs/nki-llama_*.log | grep -E "(loss|throughput|mfu)"
fi

# Inference metrics
if pgrep -f "inference" > /dev/null; then
    echo "📊 Inference Status:"
    tail -n 10 src/inference/benchmark_inference.json
fi

# Device utilization
echo "📊 Device Utilization:"
neuron-top -n 1

# Memory usage
echo "📊 Memory Status:"
free -h
```

## 🏗️ Architecture Best Practices

### 1. Kernel Reusability
Design kernels that work for both training and inference:

```python
class NKIOptimizedLayer(nn.Module):
    def __init__(self, config, training_mode=True):
        super().__init__()
        self.training_mode = training_mode
        self.config = config
        
    def forward(self, x):
        if self.config.use_nki:
            return nki_kernel(x, training=self.training_mode)
        return standard_implementation(x)
```

### 2. Configuration Management
Unified configuration for both phases:

```yaml
# config.yaml
model:
  name: llama-3-8b
  use_nki: true
  
training:
  batch_size: 8
  learning_rate: 5e-5
  nki_kernels:
    - rmsnorm
    - attention
    - linear
    
inference:
  batch_size: 1
  max_length: 2048
  nki_kernels:
    - rmsnorm
    - attention
    - linear
    - kv_cache
```

### 3. Progressive Optimization
Start simple and add complexity:

1. **Baseline**: Get everything working without NKI
2. **Single Kernel**: Add one NKI kernel (e.g., RMSNorm)
3. **Core Kernels**: Add attention and linear layers
4. **Advanced**: Implement fusion and specialized kernels

## 🎯 Scoring Optimization Strategy

### Weight Distribution
For maximum score with all three components:

```python
# Recommended weight distribution
WEIGHTS = {
    "training": 0.33,
    "inference": 0.33,
    "reasoning": 0.34
}
```

### Focus Areas by Score Impact

#### High Impact (>20% score improvement)
1. **Attention Optimization**: Both training and inference
2. **Linear Layer Fusion**: Combine with activation functions
3. **Memory Access Patterns**: Optimize for Neuron architecture

#### Medium Impact (10-20% improvement)
1. **Normalization Layers**: RMSNorm, LayerNorm
2. **Gradient Operations**: Training-specific
3. **KV Cache**: Inference-specific

#### Low Impact (<10% improvement)
1. **Activation Functions**: Unless fused with other ops
2. **Element-wise Operations**: Minor gains
3. **Data Loading**: Already optimized in framework

## 🛠️ Development Workflow

### Iterative Development Cycle
```bash
# 1. Implement kernel
nano src/kernels/my_nki_kernel.py

# 2. Test in isolation
python test_kernel.py

# 3. Integrate into model
nano src/llama.py

# 4. Benchmark improvement
./nki-llama inference benchmark --seq-len 512

# 5. Profile and optimize
neuron-profile view profiles/
```

### Continuous Integration Testing
```python
# test_suite.py
import unittest

class NKIKernelTests(unittest.TestCase):
    def test_rmsnorm_accuracy(self):
        # Compare NKI vs PyTorch implementation
        pass
        
    def test_attention_performance(self):
        # Verify speedup
        pass
        
    def test_training_convergence(self):
        # Ensure training still converges
        pass
```

## 📈 Results Analysis

### Performance Tracking
Track improvements across iterations:

```python
# track_performance.py
import json
import matplotlib.pyplot as plt

def plot_improvements(baseline, optimized):
    metrics = ['training_mfu', 'inference_throughput', 'reasoning_accuracy']
    improvements = [(optimized[m] - baseline[m]) / baseline[m] * 100 
                   for m in metrics]
    
    plt.bar(metrics, improvements)
    plt.ylabel('Improvement (%)')
    plt.title('NKI Optimization Impact')
    plt.savefig('optimization_impact.png')
```

### Score Breakdown Analysis
```bash
# Analyze score components
python src/handler.py \
    --inference-results benchmark_inference.json \
    --analyze-components \
    --output score_analysis.json
```

## 🐛 Common Integration Issues

### Environment Conflicts
```bash
# Issue: Package version mismatch between environments
# Solution: Use separate conda environments
conda create -n nki-training python=3.10
conda create -n nki-inference python=3.10
```

### Model Compatibility
```bash
# Issue: Model trained with one config, inference with another
# Solution: Always save and load full configuration
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'nki_kernels': enabled_kernels
}, 'checkpoint.pt')
```

### Cache Conflicts
```bash
# Issue: Stale compiled kernels
# Solution: Clear cache between major changes
rm -rf ~/neuron_cache/*
rm -rf ~/.cache/neuron
```

## 🏆 Competition Tips

### 1. Time Management
- **Week 1**: Get baseline working, understand the code
- **Week 2**: Implement core NKI kernels
- **Week 3**: Optimize and fine-tune
- **Final days**: Polish, document, prepare presentation

### 2. Collaboration Strategy
- **Frontend**: One member on training optimizations
- **Backend**: One member on inference optimizations
- **Integration**: One member on testing and benchmarking

### 3. Documentation
Keep detailed logs of:
- Kernel implementations
- Performance improvements
- Failed attempts (for learning)
- Configuration changes

## 📊 Example Complete Run

```bash
#!/bin/bash
# complete_hackathon_run.sh

# Setup
echo "🚀 Starting complete NKI-LLAMA pipeline"

# Training phase
tmux new -d -s training
tmux -a -t training "source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate" Enter
tmux -a -t training "cd ~/nki-llama" Enter
tmux -a -t training "./nki-llama finetune all 2>&1 | tee training.log" Enter

# Wait for training to reach a checkpoint
sleep 3600  # Adjust based on your training time

# Inference phase
tmux new -d -s inference
tmux -a -t inference "source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate" Enter
tmux -a -t inference "cd ~/nki-llama" Enter
tmux -a -t inference "./nki-llama inference benchmark 2>&1 | tee inference.log" Enter

# Reasoning phase (optional)
tmux new -d -s reasoning
tmux -a -t reasoning "source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate" Enter
tmux -a -t reasoning "./nki-llama/src/inference/scripts/reasoning-bench-lm-eval.sh" Enter

# Monitor all sessions
tmux new -s monitor
watch -n 10 './monitor.sh'
```

## 📚 Resources

### Essential Documentation
- [Complete NKI Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
- [NeuronX Distributed Training](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html)
- [NeuronX Distributed Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed-inference/index.html)

### Example Repositories
- [NKI Samples](https://github.com/aws-neuron/nki-samples)
- [NKI Autotune](https://github.com/awslabs/nki-autotune)
- [AWS Neuron Samples](https://github.com/aws-neuron/aws-neuron-samples)

### Tools and Utilities
- [Neuron Profiler](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html)
- [Neuron Top](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-top-user-guide.html)
- [TensorBoard Integration](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/tutorials/training/tensorboard.html)

---

Remember: The key to maximizing your score is to optimize both training and inference with NKI kernels while maintaining model accuracy. Focus on the highest-impact optimizations first and ensure everything integrates smoothly. Good luck!