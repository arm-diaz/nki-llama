# Self-Attention Module for NKI-LLAMA

This module implements optimized Flash Attention kernels using the Neuron Kernel Interface (NKI) for AWS Inferentia/Trainium hardware. The implementation focuses on high-performance, memory-efficient attention mechanisms for large language models.

## Overview

The self-attention module provides optimized implementations of attention mechanisms that are critical for transformer-based models like LLaMA. These implementations leverage NKI to achieve high performance on AWS Neuron hardware.

## Key Components

### Core Files

- **`attention.py`**: Main implementation of Flash Attention kernels using NKI
  - `flash_fwd`: Forward pass implementation of Flash Attention
  - `flash_attn_bwd`: Backward pass implementation for gradient computation
  - `fused_self_attn_for_SD_small_head_size`: Specialized attention for small head sizes

### Configuration

- **`FlashConfig`**: Configuration dataclass for tuning attention performance parameters
  - `seq_tile_size`: Size of sequence tiles for attention computation (default: 2048)
  - `attn_core_tile_size`: Size of attention core tiles (default: 256)
  - `training`: Flag to indicate training vs. inference mode (default: True)
  - `should_transpose_v`: Flag to control V tensor layout (default: False)
  - `lse_dtype`: Data type for log-sum-exp computation (default: "")

### Tests

- **`tests/test_flash_attn_fwd.py`**: Tests for forward pass performance and numerical accuracy
- **`tests/test_flash_attn_bwd.py`**: Tests for backward pass performance and numerical accuracy

## Features

- **Optimized Memory Usage**: Implements tiling strategies to efficiently use limited on-chip memory
- **Mixed Precision Support**: Configurable precision for computation vs. accumulation
- **Causal Masking**: Support for causal attention patterns used in decoder-only models
- **Dropout Support**: Configurable dropout for training stability
- **GQA/MQA Support**: Grouped Query Attention and Multi-Query Attention support
- **Performance Tuning**: Configurable parameters for different hardware configurations

## Usage

### Basic Usage

```python
from attention import flash_fwd, FlashConfig

# Configure the attention parameters
config = FlashConfig(
    seq_tile_size=2048,
    training=True,
    should_transpose_v=False
)

# Run the forward pass
# q: shape (bs, n_heads, d, seq_q)
# k: shape (bs, nk_heads, d, seq_k)
# v: shape (bs, nv_heads, d, seq_v) if config.should_transpose_v else (bs, nv_heads, seq_v, d)
output = flash_fwd[batch_size, kv_heads](
    q, k, v, seed, 
    use_causal_mask=True,
    mixed_precision=True,
    config=config
)
```

### Training Usage

```python
from attention import flash_fwd, flash_attn_bwd, FlashConfig

# Forward pass
output, lse = flash_fwd[batch_size, kv_heads](
    q, k, v, seed, 
    use_causal_mask=True,
    mixed_precision=True,
    config=FlashConfig(training=True)
)

# Backward pass
dq, dk, dv = flash_attn_bwd[batch_size, heads](
    q, k, v, output, dy, lse, seed,
    use_causal_mask=True,
    mixed_precision=True
)
```

## Performance Considerations

- **Sequence Length**: Performance scales with sequence length; use appropriate tiling
- **Head Dimensions**: Optimized for head dimensions ≤ 128
- **Batch Size**: Consider batch size impact on memory usage and parallelism
- **Tile Sizes**: Adjust `seq_tile_size` and `attn_core_tile_size` based on model size and hardware

## Testing

Run the tests to validate performance and numerical accuracy:

```bash
# Navigate to the tests directory
cd tests

# Run all tests
pytest test_flash_attn_*.py -v -s

# Run specific test suites
pytest test_flash_attn_fwd.py -v -s  # Forward pass tests
pytest test_flash_attn_bwd.py -v -s  # Backward pass tests

# Run only performance tests
pytest -k "perf" -v -s

# Run only numerical accuracy tests
pytest -k "numerical" -v -s

# Run simulation tests
pytest -m simulation -v -s
```

## Optimization Opportunities

Areas for potential optimization:

1. **Memory Tiling**: Improve tiling strategies for better memory locality
2. **Instruction Scheduling**: Optimize instruction ordering for better hardware utilization
3. **Precision Control**: Fine-tune mixed precision operations for specific model requirements
4. **Specialized Kernels**: Create specialized kernels for specific sequence lengths or head sizes
5. **Fused Operations**: Combine operations to reduce memory transfers

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [NKI Programming Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)