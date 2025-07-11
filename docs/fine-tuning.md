# Fine-tuning Guide for NKI-LLAMA Hackathon

## 🎯 Overview

This guide focuses exclusively on fine-tuning LLaMA models on AWS Trainium using NeuronX Distributed (NxD). Perfect for participants wanting to optimize training performance and achieve high Model FLOP Utilization (MFU).

## 📋 Prerequisites

### Instance Requirements
- **Instance Type**: trn1.32xlarge (recommended) or trn1.2xlarge (minimum)
- **AMI**: Deep Learning AMI Neuron (Ubuntu 22.04) 20250520
  - **us-east-1**: `ami-0e65a95c79775d1b6`
  - **us-west-2**: `ami-0d0a2d26f80b645c2`
- **Storage**: 512GB+ recommended (800GB default in CloudFormation)
- **Neuron SDK**: 2.23.0

### Environment Setup
```bash
# Activate the training environment
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
   - **Stack name**: Keep default or customize (e.g., `nki-llama-training`)
   - **KeyPairOption**: Choose `use-existing` (recommended - create key in EC2 console first)
   - **ExistingKeyPairName**: Select your key from dropdown (see note below)
   - **Ec2InstanceType**: Keep default `trn1.32xlarge`
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

# Configure environment
nano .env  # Add your HF_TOKEN
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
nano .env  # Add your HF_TOKEN
```

### Step 2: Run Complete Fine-tuning Pipeline
```bash
# Use tmux for long-running operations
tmux new -s training

# Activate training environment
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate

# Run the complete pipeline
./nki-llama finetune all
```

## 📊 Detailed Fine-tuning Workflow

### 1. Install Dependencies
```bash
./nki-llama finetune deps
```
This installs all required Python packages and NeuronX Distributed components.

### 2. Download Dataset
```bash
./nki-llama finetune data
```
Downloads and prepares the training dataset (default: dolly_15k).

### 3. Download Base Model
```bash
./nki-llama finetune model
```
Downloads the base LLaMA model from Hugging Face (requires HF_TOKEN).

### 4. Convert Model Format
```bash
./nki-llama finetune convert
```
Converts the model to NeuronX Distributed Training (NxDT) format.

### 5. Pre-compile Graphs
```bash
./nki-llama finetune compile
```
**Important**: Note the compile directory path from the output. You'll need this for score calculation.

Example output:
```
Pre-compile graphs: /home/ubuntu/neuron_cache/neuronxcc-2.18.121.0+9e31e41a/MODULE_15329989265349737271+a65e371e
```

### 6. Start Training
```bash
./nki-llama finetune train
```
Runs the actual fine-tuning process.

## 📈 Performance Metrics

During training, the system tracks:
- **MFU (Model FLOP Utilization)**: Target >40% for good performance
- **Throughput**: Tokens/second processed
- **Loss convergence**: Training and validation loss
- **Memory usage**: HBM utilization

## 🎯 Score Calculation (Training Only)

After training completes, calculate your performance score:

```bash
python /home/ubuntu/nki-llama/src/handler.py \
    --config /home/ubuntu/nki-llama/src/fine-tune/neuronx-distributed-training/examples/conf/hf_llama3_8B_SFT_config.yaml \
    --model-config /home/ubuntu/nki-llama/src/fine-tune/configs/model-config/8B_config_llama3-1/config.json \
    --log-file /home/ubuntu/nki-llama/logs/nki-llama_[YOUR_TIMESTAMP].log \
    --compile-dir [YOUR_COMPILE_DIR_FROM_STEP_5] \
    --throughput 2.1 \
    --output training_score.json \
    --training-weight 1.0 \
    --hw-backend trn1 \
    --calculate-score \
    --detailed \
    --verbose
```

The training score evaluates:
- **MFU improvement**: How well your optimizations improve hardware utilization
- **Throughput gains**: Training speed improvements
- **NKI optimization ratio**: Percentage of operations optimized with NKI

## 🔧 Configuration Options

### Training Configuration
Edit `src/fine-tune/neuronx-distributed-training/examples/conf/hf_llama3_8B_SFT_config.yaml`:

```yaml
# Model parameters
model:
  model_id: "meta-llama/Meta-Llama-3-8B"
  
# Training parameters
training:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 5e-5
  num_train_epochs: 1
  
# Hardware configuration
distributed:
  tensor_parallel_size: 8
  pipeline_parallel_size: 1
```

### Environment Variables (.env)
```bash
# Hugging Face token (required)
HF_TOKEN=your_token_here

# Model selection
MODEL_ID=meta-llama/Meta-Llama-3-8B
MODEL_NAME=llama-3-8b

# Hardware configuration
TENSOR_PARALLEL_SIZE=8
NEURON_RT_NUM_CORES=8
```

## 🛠️ Advanced Optimizations

### 1. Implement Custom NKI Kernels
Create optimized kernels for training operations:

```python
# Example: Optimized attention computation
@nki_jit
def nki_attention_kernel(q, k, v, mask=None):
    # Your NKI implementation here
    pass
```

### 2. Optimize Data Loading
- Use efficient data preprocessing
- Implement prefetching
- Optimize tokenization pipeline

### 3. Memory Optimization
- Gradient checkpointing
- Mixed precision training
- Efficient tensor layouts

## 📊 Monitoring Training

### Real-time Monitoring
```bash
# In a new terminal
neuron-top  # Monitor device utilization

# View training logs
tail -f logs/nki-llama_*.log
```

### Key Metrics to Watch
- **step_loss**: Should decrease over time
- **grad_norm**: Should remain stable
- **throughput**: Tokens/second
- **mfu**: Model FLOP Utilization

## 🐛 Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Reduce batch size or model parallelism
export TENSOR_PARALLEL_SIZE=4  # Instead of 8
```

#### Compilation Timeout
```bash
# Increase timeout
export NEURON_COMPILE_TIMEOUT=3600  # 1 hour
```

#### Training Instability
- Check gradient norms
- Reduce learning rate
- Enable gradient clipping

## 📚 Best Practices

1. **Always use tmux** for long-running operations
2. **Save checkpoints frequently** to prevent data loss
3. **Monitor metrics** throughout training
4. **Document your optimizations** for the presentation
5. **Test incrementally** - verify each optimization works

## 🏆 Scoring Tips

To maximize your training-only score:

1. **Focus on MFU**: Implement NKI kernels for compute-intensive operations
2. **Optimize throughput**: Reduce data loading bottlenecks
3. **Increase NKI coverage**: Replace more PyTorch ops with NKI kernels
4. **Profile extensively**: Use neuron-profile to identify bottlenecks

## 📄 Example Training Session

```bash
# Complete example workflow
tmux new -s hackathon-training

# Setup
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
cd ~/nki-llama

# Run training
./nki-llama finetune all

# Monitor progress (in another terminal)
tmux new -s monitoring
neuron-top

# After completion, calculate score
python src/handler.py --config [...] --calculate-score

# Detach from tmux: Ctrl+B, then D
```

## 🎯 Next Steps

After mastering fine-tuning:
1. Document your NKI kernel implementations
2. Prepare performance comparison charts
3. Consider exploring inference optimizations (see [inference.md](./inference.md))
4. Prepare your presentation highlighting training improvements

## 📚 Resources

- [NeuronX Distributed Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html)
- [NKI Training Examples](https://github.com/aws-neuron/nki-samples)
- [AWS Neuron SDK Guide](https://awsdocs-neuron.readthedocs-hosted.com/)

---

Remember: Focus on achieving high MFU through effective NKI kernel implementation. Good luck with your hackathon!