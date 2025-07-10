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
from attention import flash_fwd, FlashConfig
from neuronxcc.nki import benchmark, baremetal, simulate_kernel
import neuronxcc.nki.language as nl

# Configure logging for verbose output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

bench_func = benchmark(warmup=5, iters=10)(flash_fwd)

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
    """Print detailed performance metrics"""
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

def print_memory_usage(bs: int, nheads: int, seqlen_q: int, seqlen_k: int, d: int, dtype):
    """Calculate and print memory usage estimates"""
    element_size = 2 if dtype == nl.bfloat16 else 4  # bytes
    
    q_size = bs * nheads * d * seqlen_q * element_size
    k_size = bs * nheads * d * seqlen_k * element_size
    v_size = bs * nheads * seqlen_k * d * element_size
    total_size = q_size + k_size + v_size
    
    print("\n💾 MEMORY USAGE ESTIMATES:")
    print("-" * 40)
    print(f"   Q tensor:     {q_size / 1024 / 1024:.2f} MB")
    print(f"   K tensor:     {k_size / 1024 / 1024:.2f} MB")
    print(f"   V tensor:     {v_size / 1024 / 1024:.2f} MB")
    print(f"   Total Input:  {total_size / 1024 / 1024:.2f} MB")
    print(f"   Est. Peak:    {total_size * 2 / 1024 / 1024:.2f} MB (2x for intermediate)")

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

def cpu_attention_forward(q, k, v, use_causal_mask=True, mixed_precision=True):
    """CPU attention forward pass with verbose logging"""
    logger.info("🔄 Computing CPU reference attention forward pass...")
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
    nheads = q.shape[1]
    kv_heads = k.shape[1]
    
    if nheads > kv_heads:
        logger.info(f"📡 Expanding KV heads from {kv_heads} to {nheads} (GQA/MQA)")
        k = np.repeat(k, nheads//kv_heads, axis=1)
        v = np.repeat(v, nheads//kv_heads, axis=1)
    
    logger.debug("Computing attention scores...")
    raw_score = mixed_precision_matmul(q_scaled.transpose(0, 1, 3, 2), k)

    if use_causal_mask:
        logger.debug("Applying causal mask...")
        for i in range(raw_score.shape[0]):
            for j in range(raw_score.shape[1]):
                # -inf triggers invalid input error in softmax implementation, use a small negative instead
                # k=1 to exclude the diagonal, because each token can still attend to itself
                raw_score[i, j][np.triu_indices_from(raw_score[i, j], k=1)] = -9984.0

    logger.debug("Computing softmax...")
    norm_score, cached_negative_max, cached_sum_reciprocal = \
        softmax(raw_score, dim=-1, mixed_precision=mixed_precision, return_max_reduce=True)

    logger.debug("Computing final output...")
    # Transpose the result so it has the same layout as ours
    out_golden = mixed_precision_matmul(norm_score, v.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ CPU reference completed in {elapsed_time:.2f} seconds")

    return out_golden, cached_negative_max, cached_sum_reciprocal

class TestAttention:
    
    @pytest.mark.parametrize("bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask,\
                              mixed_precision, training, tile_size, kv_heads, should_transpose_v, latency", [
    [1, 6, 32*1024, 32*1024, 96, nl.bfloat16, True, True, True, 2048, 3, False, 87000000000],
    [1, 1, 32*1024, 32*1024, 96, nl.bfloat16, True, True, False, 2048, None, False, 15100000000],
    # Non-square
    [1, 3, 32*1024, 16*1024, 96, nl.bfloat16, True, True, False, 2048, None, False, 7550000000],
    [1, 3, 16*1024, 32*1024, 96, nl.bfloat16, True, True, False, 2048, None, False, 7550000000],
    ])
    def test_flash_attn_fwd_perf(self, bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask, 
                                 mixed_precision, training, tile_size, kv_heads, should_transpose_v, latency):
        
        # Print test header with all parameters
        test_params = {
            'Batch Size': bs,
            'Num Heads': nheads,
            'Q Sequence Length': f"{seqlen_q:,}",
            'K Sequence Length': f"{seqlen_k:,}",
            'Head Dimension': d,
            'Data Type': str(dtype),
            'Causal Mask': use_causal_mask,
            'Mixed Precision': mixed_precision,
            'Training Mode': training,
            'Tile Size': tile_size,
            'KV Heads': kv_heads or nheads,
            'Transpose V': should_transpose_v,
            'Expected Latency': f"{latency:,} ns"
        }
        
        print_test_header("Flash Attention Forward Performance Test", test_params)
        print_memory_usage(bs, nheads, seqlen_q, seqlen_k, d, dtype)
        
        print("\n⚙️  SETUP PHASE:")
        print("-" * 40)
        
        # Generate test data
        print("🎲 Generating random test tensors...")
        q = (np.random.random_sample([bs, nheads, d, seqlen_q]) - 0.5) * 2
        k = (np.random.random_sample([bs, nheads, d, seqlen_k]) - 0.5) * 2
        
        if should_transpose_v:
            v = (np.random.random_sample([bs, nheads, d, seqlen_k]) - 0.5) * 2
            print("   V tensor: Using transposed layout")
        else:
            v = (np.random.random_sample([bs, nheads, seqlen_k, d]) - 0.5) * 2
            print("   V tensor: Using standard layout")
        
        o_proj = np.zeros(shape=[bs, nheads, seqlen_q, d], dtype=dtype)
        out_lse = np.zeros(shape=[bs, nheads, int(nl.tile_size.pmax), seqlen_q // nl.tile_size.pmax], 
                                  dtype=nl.float32 if mixed_precision else dtype) if training else None
        seed = None
        
        # Print tensor information
        print_tensor_info("Q", q)
        print_tensor_info("K", k)
        print_tensor_info("V", v)
        
        # Cast to target dtype
        print(f"\n🔄 Converting tensors to {dtype}...")
        q = nl.static_cast(q, dtype)
        k = nl.static_cast(k, dtype)
        v = nl.static_cast(v, dtype)
        
        # Setup configuration
        config = FlashConfig(**{'seq_tile_size':tile_size, 'training':training, 'should_transpose_v':should_transpose_v})
        print(f"📝 Flash Config: {config.__dict__}")
        
        heads = nheads if kv_heads is None else kv_heads
        
        print("\n🚀 BENCHMARKING PHASE:")
        print("-" * 40)
        print("⏱️  Running benchmark with warmup=5, iters=10...")
        
        bench_func_ = bench_func[bs, heads]
        
        # Run the benchmark
        start_time = time.time()
        bench_func_(q, k, v, seed, use_causal_mask=use_causal_mask,
                    mixed_precision=mixed_precision, config=config)
        benchmark_time = time.time() - start_time
        
        print(f"✅ Benchmark completed in {benchmark_time:.2f} seconds")
        
        # Get and display results
        latency_res = bench_func_.benchmark_result.nc_latency
        p50_latency = print_performance_metrics(latency_res, latency, "Flash Attention Forward")
        
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
    @pytest.mark.parametrize("bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask,\
                              training, tile_size, kv_heads, should_transpose_v", [
    [1, 6, 4096, 4096, 128, np.float32, True, True, 2048, 3, False],
    [1, 1, 4096, 4096, 128, np.float32, True, False, 2048, None, False],
    [1, 1, 8192, 4096, 128, np.float32, True, False, 2048, None, False],
    [1, 1, 4096, 8192, 128, np.float32, True, False, 2048, None, False],
    ])
    def test_flash_attn_fwd_numerical(self, simulation_only, bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask, 
                                     training, tile_size, kv_heads, should_transpose_v):
        
        # Print test header
        test_params = {
            'Batch Size': bs,
            'Num Heads': nheads,
            'Q Sequence Length': f"{seqlen_q:,}",
            'K Sequence Length': f"{seqlen_k:,}",
            'Head Dimension': d,
            'Data Type': str(dtype),
            'Causal Mask': use_causal_mask,
            'Training Mode': training,
            'Tile Size': tile_size,
            'KV Heads': kv_heads or nheads,
            'Transpose V': should_transpose_v,
            'Simulation Only': simulation_only
        }
        
        print_test_header("Flash Attention Forward Numerical Test", test_params)
        print_memory_usage(bs, nheads, seqlen_q, seqlen_k, d, dtype)
        
        print("\n⚙️  SETUP PHASE:")
        print("-" * 40)
        
        # Generate test data
        print("🎲 Generating random test tensors...")
        q = (np.random.random_sample([bs, nheads, d, seqlen_q]) - 0.5) * 2
        k = (np.random.random_sample([bs, kv_heads or nheads, d, seqlen_k]) - 0.5) * 2
        
        if should_transpose_v:
            v = (np.random.random_sample([bs, nheads, d, seqlen_k]) - 0.5) * 2
            cpu_permute = (0, 1, 2, 3)
            print("   V tensor: Using transposed layout")
        else:
            v = (np.random.random_sample([bs, kv_heads or nheads, seqlen_k, d]) - 0.5) * 2
            cpu_permute = (0, 1, 3, 2)
            print("   V tensor: Using standard layout")

        # Print tensor information
        print_tensor_info("Q", q)
        print_tensor_info("K", k)
        print_tensor_info("V", v)
        
        # Cast to target dtype
        print(f"\n🔄 Converting tensors to {dtype}...")
        q = nl.static_cast(q, dtype)
        k = nl.static_cast(k, dtype)
        v = nl.static_cast(v, dtype)
        seed = None

        print("\n🔍 REFERENCE COMPUTATION:")
        print("-" * 40)
        
        # Compute reference (golden) output
        o_proj_golden, cached_negative_max, cached_sum_reciprocal = \
          cpu_attention_forward(q, k, v.transpose(cpu_permute), use_causal_mask=use_causal_mask, mixed_precision=True)
        
        # Reshape reference outputs to match expected format
        o_proj_golden = o_proj_golden.transpose(0,1,3,2) # (b,h, d, seq)
        cached_negative_max = cached_negative_max.reshape(bs, nheads, seqlen_q // nl.tile_size.pmax,
                                                          nl.tile_size.pmax).transpose(0, 1, 3, 2)
        cached_sum_reciprocal = cached_sum_reciprocal.reshape(bs, nheads, seqlen_q // nl.tile_size.pmax,
                                                              nl.tile_size.pmax).transpose(0, 1, 3, 2)
        lse_golden = -1.0 * (cached_negative_max + np.log(cached_sum_reciprocal)) if training else None
        
        print_tensor_info("Reference Output", o_proj_golden)
        if training:
            print_tensor_info("Reference LSE", lse_golden)
        
        # Setup configuration
        config = FlashConfig(**{'seq_tile_size':tile_size, 'training':training, 'should_transpose_v':should_transpose_v})
        print(f"📝 Flash Config: {config.__dict__}")
        
        heads = nheads if kv_heads is None else kv_heads

        print("\n🚀 FLASH ATTENTION COMPUTATION:")
        print("-" * 40)
        
        numeric_func = baremetal(flash_fwd)
        
        if simulation_only:
            print("🔬 Running in simulation mode...")
            start_time = time.time()
            results = simulate_kernel(numeric_func[bs, heads], q, k, v, seed,
                                          use_causal_mask=use_causal_mask,
                                          mixed_precision=True,
                                          config=config)
            compute_time = time.time() - start_time
            print(f"✅ Simulation completed in {compute_time:.2f} seconds")
        else:
            print("⚡ Running on hardware...")
            start_time = time.time()
            results = numeric_func[bs, heads](q, k, v, seed,
                                          use_causal_mask=use_causal_mask,
                                          mixed_precision=True,
                                          config=config)
            compute_time = time.time() - start_time
            print(f"✅ Hardware execution completed in {compute_time:.2f} seconds")

        print("\n🔬 NUMERICAL VERIFICATION:")
        print("-" * 40)
        
        if training:
            o_proj, out_lse = results
            print_tensor_info("Flash Output", o_proj)
            print_tensor_info("Flash LSE", out_lse)
            
            # Check output tensor
            output_close = np.allclose(o_proj, o_proj_golden, atol=1e-2)
            output_max_diff = np.max(np.abs(o_proj - o_proj_golden))
            output_mean_diff = np.mean(np.abs(o_proj - o_proj_golden))
            
            print(f"📊 Output Comparison:")
            print(f"   Max absolute difference: {output_max_diff:.6f}")
            print(f"   Mean absolute difference: {output_mean_diff:.6f}")
            print(f"   Tolerance: 1e-2")
            print(f"   Result: {'✅ PASS' if output_close else '❌ FAIL'}")
            
            # Check LSE tensor
            lse_close = np.allclose(out_lse, lse_golden, atol=1e-2)
            lse_max_diff = np.max(np.abs(out_lse - lse_golden))
            lse_mean_diff = np.mean(np.abs(out_lse - lse_golden))
            
            print(f"📊 LSE Comparison:")
            print(f"   Max absolute difference: {lse_max_diff:.6f}")
            print(f"   Mean absolute difference: {lse_mean_diff:.6f}")
            print(f"   Tolerance: 1e-2")
            print(f"   Result: {'✅ PASS' if lse_close else '❌ FAIL'}")
            
            # Final assertions
            try:
                assert output_close, f"Output mismatch: max_diff={output_max_diff:.6f} > 1e-2"
                assert lse_close, f"LSE mismatch: max_diff={lse_max_diff:.6f} > 1e-2"
                print(f"\n🎉 TEST PASSED! Both output and LSE match reference within tolerance")
            except AssertionError as e:
                print(f"\n💥 TEST FAILED! {str(e)}")
                raise
        else:
            o_proj = results
            print_tensor_info("Flash Output", o_proj)
            
            # Check output tensor
            output_close = np.allclose(o_proj, o_proj_golden, atol=1e-2)
            output_max_diff = np.max(np.abs(o_proj - o_proj_golden))
            output_mean_diff = np.mean(np.abs(o_proj - o_proj_golden))
            
            print(f"📊 Output Comparison:")
            print(f"   Max absolute difference: {output_max_diff:.6f}")
            print(f"   Mean absolute difference: {output_mean_diff:.6f}")
            print(f"   Tolerance: 1e-2")
            print(f"   Result: {'✅ PASS' if output_close else '❌ FAIL'}")
            
            # Final assertion
            try:
                assert output_close, f"Output mismatch: max_diff={output_max_diff:.6f} > 1e-2"
                print(f"\n🎉 TEST PASSED! Output matches reference within tolerance")
            except AssertionError as e:
                print(f"\n💥 TEST FAILED! {str(e)}")
                raise
        
        print("\n" + "="*80 + "\n")