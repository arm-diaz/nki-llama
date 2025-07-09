# 🚀 NKI-LLAMA Hackathon: Getting Started Guide

Welcome to the **NKI-LLAMA Hackathon**! This guide will help you navigate the documentation and choose the best path for your hackathon journey.

## 🎯 Welcome Hackathon Participants!

You're about to embark on an exciting challenge to optimize LLaMA models using AWS Neuron's cutting-edge NKI (Neuron Kernel Interface) technology. Whether you're focusing on training, inference, or both, we've prepared guides to help you succeed.

## 📚 Choose Your Path

We've created three specialized guides based on your optimization focus:

### 1. 🏋️ [Fine-tuning Only Guide](./docs/fine-tuning.md)
**Perfect for teams focusing on training optimization**
- Optimize Model FLOP Utilization (MFU) during training
- Implement NKI kernels for training operations
- Achieve high throughput with NeuronX Distributed
- **Score Focus**: Training performance metrics

### 2. ⚡ [Inference with NKI Guide](./docs/inference.md)
**Ideal for teams targeting inference performance**
- Minimize latency with NKI-optimized kernels
- Maximize throughput for production serving
- Implement custom kernels for attention, normalization, and more
- **Score Focus**: Inference latency and throughput

### 3. 🎯 [Complete Pipeline Guide](./docs/complete-pipeline.md)
**For teams aiming for the highest overall score**
- Combine training and inference optimizations
- Implement shared NKI kernels across both phases
- Optional reasoning evaluation for bonus points
- **Score Focus**: Performance across all metrics

## 🏃 Quick Start (5 Minutes)

### 1. Deploy Your Environment

| AWS Region | Launch CloudFormation Stack |
|:-----------|:----------------------------|
| us-east-1 (N. Virginia) |<a href="https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=nki-llama&templateURL=https://nki-llama-cfn.s3.amazonaws.com/nki-llama-stack.yaml" target="_blank">Launch stack</a> |
| us-west-2 (Oregon) |<a href="https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=nki-llama&templateURL=https://nki-llama-cfn.s3.amazonaws.com/nki-llama-stack.yaml" target="_blank">Launch stack</a> |

**Note**: Create your SSH key pair first in EC2 → Key Pairs for easy download!

### 2. Connect to Your Instance

```bash
# SSH access (recommended)
ssh -i your-key.pem ubuntu@<instance-ip>

# Or use SSM (no key needed)
aws ssm start-session --target <instance-id>
```

### 3. Run Setup Wizard

```bash
cd ~/nki-llama
./nki-llama setup
```

## 🎮 Using the NKI-LLAMA CLI

The repository includes a unified command-line interface that simplifies all operations:

```bash
# View all available commands
./nki-llama help

# Check system status
./nki-llama status

# Start your chosen workflow
./nki-llama finetune all       # For training
./nki-llama inference benchmark # For inference
```

## 📊 Understanding the Scoring System

Your submission will be evaluated on:

1. **Accuracy** ✓ - Must maintain model quality
2. **Performance Improvements** 📈
   - Training: MFU and throughput gains
   - Inference: Latency reduction and throughput increase
3. **NKI Coverage** 🎯 - Percentage of operations using NKI kernels
4. **Reasoning (Bonus)** 🧠 - Optional evaluation on reasoning tasks

**Score Formula**: 
```
Score = Accuracy × Performance_Gains × (1 + NKI_Coverage)
```

## 🛠️ Essential Resources

### Documentation
- [AWS Neuron SDK Docs](https://awsdocs-neuron.readthedocs-hosted.com/)
- [NKI Programming Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
- [NKI Sample Kernels](https://github.com/aws-neuron/nki-samples)

### Instance Information
- **Instance Type**: trn1.32xlarge (32 Neuron cores)
- **AMI**: Deep Learning AMI Neuron (Ubuntu 22.04) 20250520
- **Pre-installed**: Neuron SDK 2.23.0, PyTorch, NeuronX

## 💡 Tips for Success

1. **Start Simple**: Get the baseline working before optimizing
2. **Use tmux**: All long operations should run in tmux sessions
3. **Profile First**: Use `neuron-profile` to identify bottlenecks
4. **Iterate Quickly**: Test kernels individually before integration
5. **Document Everything**: Keep notes on what works and what doesn't

## 🚦 Ready to Start?

1. **Choose your path** from the three guides above
2. **Deploy your environment** using CloudFormation
3. **Run the setup wizard**: `./nki-llama setup`
4. **Start optimizing** and show us what NKI can do!

## 📝 Submission Checklist

Before submitting, ensure you have:
- [ ] Implemented NKI kernels with measurable improvements
- [ ] Maintained model accuracy
- [ ] Documented your approach
- [ ] Prepared performance comparison data
- [ ] Submit your score

---

**Good luck, and may the best optimizations win!** 🎉

*Remember: The key to success is balancing performance gains with code quality and maintainability. Focus on high-impact optimizations first.*