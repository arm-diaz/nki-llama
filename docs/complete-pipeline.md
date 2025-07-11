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

1. **Download the CloudFormation template**: 
   - Click here to download: [deployment.yaml](../deployment/deployment.yaml)

2. **Click** on one of the CloudFormation Console links above for your preferred region.

3. **Upload the template**:
   - Choose **Upload a template file**
   - Click **Choose file** and select the downloaded `deployment.yaml`
   - Click **Next**

4. **Configure the stack:**
   - **Stack name**: Keep default or customize (e.g., `nki-llama-complete`)
   - **KeyPairOption**: Choose `use-existing` (recommended - create key in EC2 console first)
   - **ExistingKeyPairName**: Select your key from dropdown (see note below)
   - **Ec2InstanceType**: Keep default `trn1.32xlarge`
   - **VpcOption**: Keep default `create-new`
   - Click **Next**
   
   **Note**: For easy key download, first create a key pair in EC2 → Key Pairs → Create key pair, download it, then return here and select it from the dropdown.

5. **Configure stack options**: Leave all values as default and click **Next**

6. **Review and create:**
   - Check the box: "I acknowledge that AWS CloudFormation might create IAM resources"
   - Click **Create stack**
   - Stack creation takes ~5-10 minutes

7. **Access your instance:**
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

## 🐛 Common Integration Issues

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
- **Final days**: Polish, document, prepare submission

### 2. Documentation
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

# Inference phase
tmux new -d -s inference
tmux -a -t inference "source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate" Enter
tmux -a -t inference "cd ~/nki-llama" Enter
tmux -a -t inference "./nki-llama inference benchmark 2>&1 | tee inference.log" Enter

# Reasoning phase (optional)
tmux new -d -s reasoning
tmux -a -t reasoning "source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate" Enter
tmux -a -t reasoning "./nki-llama/src/inference/scripts/reasoning-bench-lm-eval.sh" Enter
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