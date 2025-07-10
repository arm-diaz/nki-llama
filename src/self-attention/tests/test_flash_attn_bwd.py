"""
Copyright (c) 2023, Amazon.com. All Rights Reserved
"""
import pytest
import sys
import os
import logging
import time
from typing import Optional, Tuple
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention import flash_attn_bwd
from neuronxcc.nki import benchmark, baremetal, simulate_kernel
import neuronxcc.nki.language as nl

# Configure logging for verbose output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

xfail = pytest.mark.arch_specific_xfail
bench_func = benchmark(warmup=5, iters=10)(flash_attn_bwd)

def print_test_header(test_name: str, params: dict):
    """Print a formatted test header with parameters"""
    print("\n" + "="*80)
    print(f"🧪 RUNNING TEST: {test_name}")
    print("="*80)
    print("📋 Test Parameters:")
    for key, value in params.items():
        print(f"   {key:20}: {value}")
    print("="*80)

def print_tensor_info(name: str, tensor: np.ndarray):
    """Print detailed tensor information"""
    print(f"📊 {name} Info:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Size (elements): {tensor.size:,}")
    print(f"   Memory (MB): {tensor.nbytes / 1024 / 1024:.2f}")
    print(f"   Min/Max: {tensor.min():.6f} / {tensor.max():.6f}")
    print(f"   Mean/Std: {tensor.mean():.6f} / {tensor.std():.6f}")

def print_performance_metrics(latency_res, expected_latency: int, test_name: str):
    """Print detailed performance metrics with robust error handling"""
    print("\n📈 PERFORMANCE METRICS:")
    print("-" * 40)
    
    # Try to get available percentiles, fallback to common ones
    percentiles = [50, 90, 95, 99]
    available_percentiles = []
    
    for p in percentiles:
        try:
            latency = latency_res.get_latency_percentile(p)
            available_percentiles.append(p)
            status = "✅ PASS" if latency <= expected_latency else "❌ FAIL"
            print(f"   P{p:2d} Latency: {latency:,} ns ({latency/1e9:.3f}s) {status}")
        except (KeyError, AttributeError, Exception) as e:
            print(f"   P{p:2d} Latency: ❓ NOT AVAILABLE ({type(e).__name__})")
    
    # Try to get basic stats if percentiles fail
    if not available_percentiles:
        try:
            # Try alternative methods to get latency data
            if hasattr(latency_res, 'mean'):
                mean_latency = latency_res.mean
                print(f"   Mean Latency: {mean_latency:,} ns ({mean_latency/1e9:.3f}s)")
            if hasattr(latency_res, 'min'):
                min_latency = latency_res.min
                print(f"   Min Latency:  {min_latency:,} ns ({min_latency/1e9:.3f}s)")
            if hasattr(latency_res, 'max'):
                max_latency = latency_res.max
                print(f"   Max Latency:  {max_latency:,} ns ({max_latency/1e9:.3f}s)")
            
            print(f"   Available attributes: {[attr for attr in dir(latency_res) if not attr.startswith('_')]}")
        except Exception as e:
            print(f"   ⚠️  Could not extract latency metrics: {e}")
            print(f"   Latency result type: {type(latency_res)}")
            print(f"   Available methods: {[method for method in dir(latency_res) if not method.startswith('_')]}")
    
    print(f"   Expected:   {expected_latency:,} ns ({expected_latency/1e9:.3f}s)")
    
    # Use P50 if available, otherwise try other metrics
    try:
        p50_latency = latency_res.get_latency_percentile(50)
        test_passed = p50_latency <= expected_latency
        print(f"   Test Status: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        return p50_latency
    except:
        print(f"   Test Status: ❓ CANNOT DETERMINE (P50 not available)")
        return None

def print_memory_usage(bs: int, nheads: int, seqlen: int, d: int, dtype):
    """Calculate and print memory usage estimates for backward pass"""
    element_size = 2 if dtype == nl.bfloat16 else 4  # bytes
    
    # Input tensors
    q_size = bs * nheads * d * seqlen * element_size
    k_size = bs * nheads * d * seqlen * element_size
    v_size = bs * nheads * d * seqlen * element_size
    dy_size = bs * nheads * d * seqlen * element_size
    o_proj_size = bs * nheads * d * seqlen * element_size
    lse_size = bs * nheads * nl.tile_size.pmax * (seqlen // nl.tile_size.pmax) * 4  # float32
    
    # Output gradients
    dq_size = q_size
    dk_size = k_size
    dv_size = v_size
    
    total_input_size = q_size + k_size + v_size + dy_size + o_proj_size + lse_size
    total_output_size = dq_size + dk_size + dv_size
    total_size = total_input_size + total_output_size
    
    print("\n💾 MEMORY USAGE ESTIMATES:")
    print("-" * 40)
    print("   Input Tensors:")
    print(f"     Q tensor:       {q_size / 1024 / 1024:.2f} MB")
    print(f"     K tensor:       {k_size / 1024 / 1024:.2f} MB")
    print(f"     V tensor:       {v_size / 1024 / 1024:.2f} MB")
    print(f"     dY tensor:      {dy_size / 1024 / 1024:.2f} MB")
    print(f"     O_proj tensor:  {o_proj_size / 1024 / 1024:.2f} MB")
    print(f"     LSE tensor:     {lse_size / 1024 / 1024:.2f} MB")
    print("   Output Gradients:")
    print(f"     dQ tensor:      {dq_size / 1024 / 1024:.2f} MB")
    print(f"     dK tensor:      {dk_size / 1024 / 1024:.2f} MB")
    print(f"     dV tensor:      {dv_size / 1024 / 1024:.2f} MB")
    print("-" * 40)
    print(f"   Total Input:    {total_input_size / 1024 / 1024:.2f} MB")
    print(f"   Total Output:   {total_output_size / 1024 / 1024:.2f} MB")
    print(f"   Total Memory:   {total_size / 1024 / 1024:.2f} MB")
    print(f"   Est. Peak:      {total_size * 2 / 1024 / 1024:.2f} MB (2x for intermediate)")

def softmax(x: np.ndarray, dim: int, zero_max_mode=False,
            mixed_precision=False, return_max_reduce=False):
    """Softmax implementation with verbose logging"""
    logger.debug(f"Computing softmax on tensor shape {x.shape} along dim {dim}")
    
    max_value = np.amax(x, axis=dim, keepdims=True)
    max_value = np.maximum(0, max_value) if zero_max_mode else max_value
    exp = np.exp(x - max_value)
    
    if mixed_precision:
        reduce = np.add.reduce(exp.astype(np.float32), axis=dim, keepdims=True).astype(x.dtype)
    else:
        reduce = np.add.reduce(exp, axis=dim, keepdims=True)
    
    if return_max_reduce:
        return exp / reduce, -max_value, np.reciprocal(reduce)
    return exp / reduce

def softmax_dx(dy: np.ndarray, y: np.ndarray, dim: int, mixed_precision=False):
    """Softmax gradient computation with logging"""
    logger.debug(f"Computing softmax gradient on tensors shape {dy.shape}")
    
    # dx_i = (dy_i - sum(dy_k*y_k)) * y_i
    prod = dy * y
    if mixed_precision:
        reduce = np.add.reduce(prod.astype(np.float32), axis=dim, keepdims=True).astype(dy.dtype)
    else:
        reduce = np.add.reduce(prod, axis=dim, keepdims=True)
    subtract = dy - reduce
    return subtract * y

def cpu_attention_backward(q, k, v, dy, use_causal_mask=True, mixed_precision=True):
    """
    Compute the attention backward with the softmax recomputation
    """
    logger.info("🔄 Computing CPU reference attention backward pass...")
    start_time = time.time()
    
    def mixed_precision_matmul(a, b):
        input_dtype = a.dtype
        a, b = a.astype(np.float32), b.astype(np.float32)
        c = np.matmul(a, b)
        return c.astype(input_dtype)

    _, _, d, _ = q.shape
    logger.debug(f"Attention head dimension: {d}")
    
    # Compute golden output
    softmax_scale = 1.0 / (d ** 0.5)
    logger.debug(f"Softmax scale factor: {softmax_scale:.6f}")
    
    q_scaled = q * softmax_scale
    
    logger.debug("Computing attention scores...")
    raw_score = mixed_precision_matmul(q_scaled.transpose(0, 1, 3, 2), k)

    if use_causal_mask:
        logger.debug("Applying causal mask...")
        for i in range(raw_score.shape[0]):
            for j in range(raw_score.shape[1]):
                # -inf triggers invalid input error in softmax implementation, use a small negative instead
                # k=1 to exclude the diagonal, because each token can still attend to itself
                raw_score[i, j][np.triu_indices_from(raw_score[i, j], k=1)] = -9984.0

    logger.debug("Computing forward softmax...")
    norm_score, cached_negative_max, cached_sum_reciprocal = \
        softmax(raw_score, dim=-1, mixed_precision=mixed_precision, return_max_reduce=True)

    logger.debug("Computing backward pass gradients...")
    
    # Calculate softmax_dy = (dL/dy)^T @ V
    logger.debug("Computing softmax gradient input...")
    softmax_dy = mixed_precision_matmul(dy.transpose(0, 1, 3, 2), v)

    # Calculate dv = (dL/dy) @ softmax_y
    logger.debug("Computing dV gradient...")
    dv_golden = mixed_precision_matmul(dy, norm_score)

    # Calculate softmax_dx
    logger.debug("Computing softmax gradient...")
    softmax_dx_golden = softmax_dx(softmax_dy, norm_score, dim=-1, mixed_precision=mixed_precision)

    # Calculate dq
    logger.debug("Computing dQ gradient...")
    dq_golden = mixed_precision_matmul(k, softmax_dx_golden.transpose(0, 1, 3, 2)) * softmax_scale

    # Calculate dk
    logger.debug("Computing dK gradient...")
    dk_golden = mixed_precision_matmul(q_scaled, softmax_dx_golden)

    # Calculate output projection
    logger.debug("Computing output projection...")
    o_proj = np.matmul(norm_score, v.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)

    elapsed_time = time.time() - start_time
    logger.info(f"✅ CPU reference backward pass completed in {elapsed_time:.2f} seconds")

    return dq_golden, dk_golden, dv_golden, cached_negative_max, cached_sum_reciprocal, o_proj

def print_gradient_comparison(grad_name: str, computed_grad: np.ndarray, reference_grad: np.ndarray, tolerance: float = 1e-2):
    """Print detailed comparison of gradients"""
    max_diff = np.max(np.abs(computed_grad - reference_grad))
    mean_diff = np.mean(np.abs(computed_grad - reference_grad))
    relative_error = np.mean(np.abs(computed_grad - reference_grad) / (np.abs(reference_grad) + 1e-8))
    close = np.allclose(computed_grad, reference_grad, atol=tolerance)
    
    print(f"📊 {grad_name} Gradient Comparison:")
    print(f"   Max absolute difference:  {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.6f}")
    print(f"   Mean relative error:      {relative_error:.6f}")
    print(f"   Tolerance:                {tolerance}")
    print(f"   Result: {'✅ PASS' if close else '❌ FAIL'}")
    
    if not close:
        # Additional debugging info for failures
        print(f"   Computed - Min/Max: {computed_grad.min():.6f} / {computed_grad.max():.6f}")
        print(f"   Reference - Min/Max: {reference_grad.min():.6f} / {reference_grad.max():.6f}")
    
    return close

class TestAttention:

    @xfail # P167481231
    @pytest.mark.parametrize("bs, nheads, seqlen, d, dtype, latency", [
        [1, 4, 32*1024, 128, nl.bfloat16, 117000],
    ])
    def test_flash_attn_bwd_perf(self, bs, nheads, seqlen, d, dtype, latency):
        
        # Print test header with all parameters
        test_params = {
            'Batch Size': bs,
            'Num Heads': nheads,
            'Sequence Length': f"{seqlen:,}",
            'Head Dimension': d,
            'Data Type': str(dtype),
            'Expected Latency': f"{latency:,} ns"
        }
        
        print_test_header("Flash Attention Backward Performance Test", test_params)
        print_memory_usage(bs, nheads, seqlen, d, dtype)
        
        print("\n⚙️  SETUP PHASE:")
        print("-" * 40)
        
        # Generate test data
        print("🎲 Generating random test tensors...")
        q = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        k = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        v = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        dy = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        o_proj = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        lse = np.random.random_sample([bs, nheads, nl.tile_size.pmax, seqlen // nl.tile_size.pmax]).astype(np.float32)
        seed = None

        # Print tensor information
        print_tensor_info("Q", q)
        print_tensor_info("K", k)
        print_tensor_info("V", v)
        print_tensor_info("dY (output gradient)", dy)
        print_tensor_info("O_proj (forward output)", o_proj)
        print_tensor_info("LSE (log-sum-exp)", lse)
        
        # Cast to target dtype
        print(f"\n🔄 Converting tensors to {dtype}...")
        q = nl.static_cast(q, dtype)
        k = nl.static_cast(k, dtype)
        v = nl.static_cast(v, dtype)
        o_proj = nl.static_cast(o_proj, dtype)
        dy = nl.static_cast(dy, dtype)
        
        print("\n🚀 BENCHMARKING PHASE:")
        print("-" * 40)
        print("⏱️  Running benchmark with warmup=5, iters=10...")
        print("⚠️  Note: This test is marked as xfail due to P167481231")
        
        bench_func_ = bench_func[bs, nheads]
        
        # Run the benchmark
        start_time = time.time()
        bench_func_(q, k, v, o_proj, dy, lse, seed,
                    use_causal_mask=True, mixed_precision=True)
        benchmark_time = time.time() - start_time
        
        print(f"✅ Benchmark completed in {benchmark_time:.2f} seconds")
        
        # Get and display results
        latency_res = bench_func_.benchmark_result.nc_latency
        p50_latency = print_performance_metrics(latency_res, latency, "Flash Attention Backward")
        
        # Final assertion with better error handling
        if p50_latency is not None:
            try:
                assert p50_latency <= latency
                print(f"\n🎉 TEST PASSED! P50 latency ({p50_latency:,} ns) <= expected ({latency:,} ns)")
            except AssertionError:
                print(f"\n💥 TEST FAILED! P50 latency ({p50_latency:,} ns) > expected ({latency:,} ns)")
                raise
        else:
            # Fallback: try to find any available latency metric
            print(f"\n⚠️  WARNING: Could not determine P50 latency for comparison")
            print(f"   Benchmark result type: {type(bench_func_.benchmark_result)}")
            print(f"   NC latency type: {type(latency_res)}")
            
            # Try alternative assertion methods
            try:
                # Look for any latency value we can use
                if hasattr(latency_res, 'mean'):
                    mean_latency = latency_res.mean
                    assert mean_latency <= latency
                    print(f"✅ Using mean latency for comparison: {mean_latency:,} ns <= {latency:,} ns")
                else:
                    print("❌ No suitable latency metric found for assertion")
                    raise AssertionError("Cannot determine latency for comparison")
            except Exception as e:
                print(f"💥 Assertion failed: {e}")
                raise
        
        print("\n" + "="*80 + "\n")

    @pytest.mark.simulation
    @pytest.mark.parametrize("bs, nheads, seqlen, d, dtype", [
        [1, 4, 4096, 128, np.float32],
    ])
    def test_flash_attn_bwd_numerical(self, simulation_only, bs, nheads, seqlen, d, dtype):
        
        # Print test header
        test_params = {
            'Batch Size': bs,
            'Num Heads': nheads,
            'Sequence Length': f"{seqlen:,}",
            'Head Dimension': d,
            'Data Type': str(dtype),
            'Simulation Only': simulation_only
        }
        
        print_test_header("Flash Attention Backward Numerical Test", test_params)
        print_memory_usage(bs, nheads, seqlen, d, dtype)
        
        print("\n⚙️  SETUP PHASE:")
        print("-" * 40)
        
        # Generate test data
        print("🎲 Generating random test tensors...")
        q = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        k = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        v = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        dy = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
        
        # Print tensor information
        print_tensor_info("Q", q)
        print_tensor_info("K", k)
        print_tensor_info("V", v)
        print_tensor_info("dY (output gradient)", dy)
        
        # Cast to target dtype
        print(f"\n🔄 Converting tensors to {dtype}...")
        q = nl.static_cast(q, dtype)
        k = nl.static_cast(k, dtype)
        v = nl.static_cast(v, dtype)
        dy = nl.static_cast(dy, dtype)
        seed = None

        print("\n🔍 REFERENCE COMPUTATION:")
        print("-" * 40)
        
        # Compute reference (golden) output
        dq_golden, dk_golden, dv_golden, cached_negative_max, cached_sum_reciprocal, o_proj = \
          cpu_attention_backward(q, k, v, dy, use_causal_mask=True)
        
        # Reshape reference outputs to match expected format
        cached_negative_max = cached_negative_max.reshape(bs, nheads, seqlen // nl.tile_size.pmax,
                                                          nl.tile_size.pmax).transpose(0, 1, 3, 2)
        cached_sum_reciprocal = cached_sum_reciprocal.reshape(bs, nheads, seqlen // nl.tile_size.pmax,
                                                              nl.tile_size.pmax).transpose(0, 1, 3, 2)
        lse = -1.0 * (cached_negative_max + np.log(cached_sum_reciprocal))
        
        print_tensor_info("Reference dQ", dq_golden)
        print_tensor_info("Reference dK", dk_golden)
        print_tensor_info("Reference dV", dv_golden)
        print_tensor_info("Reference O_proj", o_proj)
        print_tensor_info("LSE (computed)", lse)

        print("\n🚀 FLASH ATTENTION BACKWARD COMPUTATION:")
        print("-" * 40)
        
        numeric_func = baremetal(flash_attn_bwd)
        
        if simulation_only:
            print("🔬 Running in simulation mode...")
            start_time = time.time()
            out_dq, out_dk, out_dv = simulate_kernel(numeric_func[bs, nheads], q, k, v, o_proj, dy, lse, seed,
                                                     use_causal_mask=True,
                                                     mixed_precision=True)
            compute_time = time.time() - start_time
            print(f"✅ Simulation completed in {compute_time:.2f} seconds")
        else:
            print("⚡ Running on hardware...")
            start_time = time.time()
            out_dq, out_dk, out_dv = numeric_func[bs, nheads](q, k, v, o_proj, dy, lse, seed,
                                                             use_causal_mask=True,
                                                             mixed_precision=True)
            compute_time = time.time() - start_time
            print(f"✅ Hardware execution completed in {compute_time:.2f} seconds")

        print("\n🔬 NUMERICAL VERIFICATION:")
        print("-" * 40)
        
        print_tensor_info("Flash dQ", out_dq)
        print_tensor_info("Flash dK", out_dk)
        print_tensor_info("Flash dV", out_dv)
        
        # Check all gradients
        dq_close = print_gradient_comparison("dQ", out_dq, dq_golden, tolerance=1e-2)
        dk_close = print_gradient_comparison("dK", out_dk, dk_golden, tolerance=1e-2)
        dv_close = print_gradient_comparison("dV", out_dv, dv_golden, tolerance=1e-2)
        
        # Final assertions
        try:
            assert dq_close, f"dQ gradient mismatch"
            assert dk_close, f"dK gradient mismatch"
            assert dv_close, f"dV gradient mismatch"
            print(f"\n🎉 TEST PASSED! All gradients match reference within tolerance")
        except AssertionError as e:
            print(f"\n💥 TEST FAILED! {str(e)}")
            raise
        
        print("\n" + "="*80 + "\n")