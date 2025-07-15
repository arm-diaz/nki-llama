#!/usr/bin/env python3
"""
Configuration utilities for self-attention tests
"""
import os
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the config file
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "config", "performance_metrics.json")

def ensure_config_dir():
    """Ensure the config directory exists"""
    config_dir = os.path.dirname(CONFIG_FILE)
    if not os.path.exists(config_dir):
        try:
            os.makedirs(config_dir)
            logger.info(f"Created config directory: {config_dir}")
        except Exception as e:
            logger.error(f"Failed to create config directory: {e}")
            return False
    return True

def load_config() -> Dict[str, Any]:
    """Load the configuration from the JSON file"""
    if not ensure_config_dir():
        return {}
        
    if not os.path.exists(CONFIG_FILE):
        logger.info(f"Config file not found, creating default: {CONFIG_FILE}")
        default_config = {
            "FWD_LATENCY_TOTAL": 0,
            "FWD_BASE_LATENCY_TOTAL": 0,
            "FWD_TEST_COUNT": 0,
            "BWD_LATENCY_TOTAL": 0,
            "BWD_BASE_LATENCY_TOTAL": 0,
            "BWD_TEST_COUNT": 0,
            "NKI_FLOP_RATIO": 0.85,
            "LAST_RUN_TIMESTAMP": "",
            "test_details": []
        }
        save_config(default_config)
        return default_config
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return {}

def save_config(config: Dict[str, Any]) -> bool:
    """Save the configuration to the JSON file"""
    if not ensure_config_dir():
        return False
        
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to: {CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config file: {e}")
        return False

def update_config(updates: Dict[str, Any]) -> bool:
    """Update specific configuration values"""
    config = load_config()
    config.update(updates)
    return save_config(config)

def add_test_result(test_type: str, achieved_latency: int, expected_latency: int, test_params: Dict[str, Any]) -> bool:
    """Add a test result and update the accumulated latency totals
    
    Args:
        test_type: Either 'FWD' or 'BWD' for forward or backward tests
        achieved_latency: The measured latency in nanoseconds
        expected_latency: The baseline/expected latency in nanoseconds
        test_params: Dictionary of test parameters (batch size, heads, etc.)
        
    Returns:
        bool: True if update was successful
    """
    if test_type not in ['FWD', 'BWD']:
        logger.error(f"Invalid test type: {test_type}. Must be 'FWD' or 'BWD'.")
        return False
        
    config = load_config()
    
    # Update the accumulated totals
    latency_key = f"{test_type}_LATENCY_TOTAL"
    base_key = f"{test_type}_BASE_LATENCY_TOTAL"
    count_key = f"{test_type}_TEST_COUNT"
    
    config[latency_key] = config.get(latency_key, 0) + achieved_latency
    config[base_key] = config.get(base_key, 0) + expected_latency
    config[count_key] = config.get(count_key, 0) + 1
    
    # Add timestamp
    import datetime
    config["LAST_RUN_TIMESTAMP"] = datetime.datetime.now().isoformat()
    
    # Add test details to the history
    if "test_details" not in config:
        config["test_details"] = []
        
    test_details = {
        "timestamp": config["LAST_RUN_TIMESTAMP"],
        "test_type": test_type,
        "achieved_latency": achieved_latency,
        "expected_latency": expected_latency,
        "params": test_params
    }
    
    config["test_details"].append(test_details)
    
    # Keep only the last 20 test details to avoid the file growing too large
    if len(config["test_details"]) > 20:
        config["test_details"] = config["test_details"][-20:]
    
    return save_config(config)

def get_latency_improvement_ratio(test_type: str) -> float:
    """Calculate the latency improvement ratio for a test type
    
    Args:
        test_type: Either 'FWD' or 'BWD' for forward or backward tests
        
    Returns:
        float: The latency improvement ratio (baseline/achieved)
    """
    config = load_config()
    
    latency_key = f"{test_type}_LATENCY_TOTAL"
    base_key = f"{test_type}_BASE_LATENCY_TOTAL"
    
    achieved = config.get(latency_key, 0)
    baseline = config.get(base_key, 0)
    
    if achieved <= 0:
        return 0.0
        
    return baseline / achieved


def calculate_nki_flop_ratio(bs: int, nheads: int, seq_len: int, d: int, is_backward: bool = False) -> float:
    """
    Calculate the NKI FLOP ratio based on kernel characteristics.
    This estimates what percentage of operations are executed on NKI hardware.
    
    Args:
        bs: Batch size
        nheads: Number of attention heads
        seq_len: Sequence length
        d: Head dimension
        is_backward: Whether this is a backward pass calculation
        
    Returns:
        float: The estimated NKI FLOP ratio (0.0 to 1.0)
    """
    # Calculate total FLOPs for attention
    # For forward pass: 2 * bs * nheads * seq_len * seq_len * d
    # For backward pass: ~3x the forward pass
    
    # Calculate FLOPs for different components
    qk_bmm_flops = 2 * bs * nheads * seq_len * seq_len * d  # Q*K^T matrix multiply
    attn_v_bmm_flops = 2 * bs * nheads * seq_len * seq_len * d  # Attention * V matrix multiply
    softmax_flops = bs * nheads * seq_len * seq_len * 5  # Softmax operations (exp, sum, div)
    
    # Total FLOPs for forward pass
    total_forward_flops = qk_bmm_flops + attn_v_bmm_flops + softmax_flops
    
    # For backward pass, we need gradients for Q, K, V
    if is_backward:
        # Backward pass has additional operations for gradients
        dq_flops = 2 * bs * nheads * seq_len * seq_len * d  # dQ calculation
        dk_flops = 2 * bs * nheads * seq_len * seq_len * d  # dK calculation
        dv_flops = 2 * bs * nheads * seq_len * seq_len * d  # dV calculation
        dsoftmax_flops = bs * nheads * seq_len * seq_len * 10  # Softmax gradient operations
        
        total_flops = total_forward_flops + dq_flops + dk_flops + dv_flops + dsoftmax_flops
    else:
        total_flops = total_forward_flops
    
    # Estimate NKI accelerated operations
    # Matrix multiplies and most vector operations can be accelerated
    nki_accelerated_flops = qk_bmm_flops + attn_v_bmm_flops
    
    if is_backward:
        nki_accelerated_flops += dq_flops + dk_flops + dv_flops
    
    # Some softmax operations can be accelerated too
    nki_accelerated_flops += softmax_flops * 0.7  # Assume 70% of softmax ops are accelerated
    
    if is_backward:
        nki_accelerated_flops += dsoftmax_flops * 0.7
    
    # Calculate the ratio
    nki_flop_ratio = nki_accelerated_flops / total_flops
    
    # Apply some adjustments based on empirical observations
    # Larger batch sizes and head dimensions tend to have better utilization
    batch_factor = min(1.0, 0.8 + (bs * 0.05))  # Increases with batch size
    head_factor = min(1.0, 0.8 + (d / 256) * 0.2)  # Increases with head dimension
    
    # Sequence length affects utilization - very long sequences may have lower utilization
    seq_factor = 1.0
    if seq_len > 8192:
        seq_factor = 0.95  # Slight reduction for very long sequences
    
    # Apply the adjustments
    adjusted_ratio = nki_flop_ratio * batch_factor * head_factor * seq_factor
    
    # Ensure the ratio is between 0.0 and 1.0
    return max(0.0, min(1.0, adjusted_ratio))

def get_config_value(key: str, default=None) -> Any:
    """Get a specific configuration value"""
    config = load_config()
    return config.get(key, default)