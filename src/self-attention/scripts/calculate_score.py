#!/usr/bin/env python3
"""
Simple Self-Attention Score Calculator

This script calculates a combined score for self-attention based on forward and backward pass metrics.
It can be used directly after running the tests or with manually provided metrics.

The scoring formula is:
    final_score = accuracy * latency_improvement * throughput_improvement * (1.0 + nki_flop_ratio)

Where:
- accuracy: Binary value (1.0 or 0.0) indicating if numerical tests pass
- latency_improvement: Ratio of baseline latency to measured latency
- throughput_improvement: Inversely proportional to latency (higher is better)
- nki_flop_ratio: Ratio of operations executed on NKI hardware (hardware utilization)

The NKI FLOP ratio is calculated based on the kernel characteristics and represents
the percentage of operations that are accelerated by the NKI hardware.
"""
import argparse
import json
import os
import sys
from datetime import datetime

def calculate_score(fwd_latency, fwd_base_latency, fwd_numerical_accuracy,
                   bwd_latency, bwd_base_latency, bwd_numerical_accuracy,
                   fwd_weight=0.4, bwd_weight=0.6, nki_flop_ratio=0.0):
    """
    Calculate a combined score for self-attention.
    
    Parameters:
    - fwd_latency: Measured latency for forward pass (ns)
    - fwd_base_latency: Baseline latency for forward pass (ns)
    - fwd_numerical_accuracy: Boolean indicating if forward numerical tests passed
    - bwd_latency: Measured latency for backward pass (ns)
    - bwd_base_latency: Baseline latency for backward pass (ns)
    - bwd_numerical_accuracy: Boolean indicating if backward numerical tests passed
    - fwd_weight: Weight for forward pass in combined score (default: 0.4)
    - bwd_weight: Weight for backward pass in combined score (default: 0.6)
    - nki_flop_ratio: Ratio of NKI FLOPs to total FLOPs (default: 0.0)
    
    Returns:
    - Dictionary containing all score components and the final score
    """
    # Convert boolean accuracy to 1.0 or 0.0
    fwd_accuracy = 1.0 if fwd_numerical_accuracy else 0.0
    bwd_accuracy = 1.0 if bwd_numerical_accuracy else 0.0
    
    # Calculate latency improvements
    fwd_latency_improvement = fwd_base_latency / fwd_latency if fwd_latency > 0 else 0.0
    bwd_latency_improvement = bwd_base_latency / bwd_latency if bwd_latency > 0 else 0.0
    
    # Calculate throughput improvements (inversely proportional to latency)
    fwd_throughput_improvement = fwd_latency_improvement
    bwd_throughput_improvement = bwd_latency_improvement
    
    # Calculate individual scores
    # Score = accuracy * latency_improvement * throughput_improvement * (1.0 + nki_flop_ratio)
    fwd_score = fwd_accuracy * fwd_latency_improvement * fwd_throughput_improvement
    bwd_score = bwd_accuracy * bwd_latency_improvement * bwd_throughput_improvement
    
    # Calculate combined score with NKI FLOP ratio bonus
    # If either test fails numerically, the combined score is 0
    combined_numerical_accuracy = fwd_accuracy * bwd_accuracy
    if combined_numerical_accuracy < 1.0:
        raw_score = 0.0
        combined_score = 0.0
    else:
        # Apply the NKI FLOP ratio bonus to the weighted sum of forward and backward scores
        raw_score = ((fwd_weight * fwd_score) + (bwd_weight * bwd_score)) / 1000000000 # Dividing by ns
        combined_score = raw_score * (1.0 + nki_flop_ratio) # Dividing by nanoseconds
    
    # Return all components
    return {
        "forward": {
            "latency": fwd_latency,
            "base_latency": fwd_base_latency,
            "latency_improvement": fwd_latency_improvement,
            "throughput_improvement": fwd_throughput_improvement,
            "numerical_accuracy": fwd_accuracy,
            "score": fwd_score
        },
        "backward": {
            "latency": bwd_latency,
            "base_latency": bwd_base_latency,
            "latency_improvement": bwd_latency_improvement,
            "throughput_improvement": bwd_throughput_improvement,
            "numerical_accuracy": bwd_accuracy,
            "score": bwd_score
        },
        "combined": {
            "forward_weight": fwd_weight,
            "backward_weight": bwd_weight,
            "nki_flop_ratio": nki_flop_ratio,
            "raw_score": raw_score if combined_numerical_accuracy >= 1.0 else 0.0,
            "score": combined_score
        }
    }

def print_results(results):
    """Print formatted results to console"""
    print("\n" + "="*60)
    print("Self-Attention Benchmark Results")
    print("="*60)
    
    print("\nForward Pass:")
    print(f"   Latency:             {results['forward']['latency']:,} ns")
    print(f"   Base Latency:        {results['forward']['base_latency']:,} ns")
    print(f"   Latency Improvement: {results['forward']['latency_improvement']:.2f}x")
    print(f"   Throughput Improvement: {results['forward']['throughput_improvement']:.2f}x")
    print(f"   Numerical Accuracy:  {'✅ PASS' if results['forward']['numerical_accuracy'] == 1.0 else '❌ FAIL'}")
    print(f"   Forward Score:       {results['forward']['score']:.2f}")
    
    print("\nBackward Pass:")
    print(f"   Latency:             {results['backward']['latency']:,} ns")
    print(f"   Base Latency:        {results['backward']['base_latency']:,} ns")
    print(f"   Latency Improvement: {results['backward']['latency_improvement']:.2f}x")
    print(f"   Throughput Improvement: {results['backward']['throughput_improvement']:.2f}x")
    print(f"   Numerical Accuracy:  {'✅ PASS' if results['backward']['numerical_accuracy'] == 1.0 else '❌ FAIL'}")
    print(f"   Backward Score:      {results['backward']['score']:.2f}")
    
    print("\nCombined Metrics:")
    print(f"   Forward Weight:      {results['combined']['forward_weight']:.2f}")
    print(f"   Backward Weight:     {results['combined']['backward_weight']:.2f}")
    print(f"   NKI FLOP Ratio:      {results['combined']['nki_flop_ratio']:.2f}")
    print(f"   Raw Score:           {results['combined']['raw_score']:.2f}")
    print(f"   Final Score:         {results['combined']['score']:.2f} = Raw Score * (1 + NKI Flop Ratio)")
    
    # Print overall status
    if results['combined']['score'] > 0.0:
        print("\n🎉 OVERALL STATUS: PASS")
    else:
        print("\n❌ OVERALL STATUS: FAIL")
    
    print("\n" + "="*60)

def main():
    """Main function to parse arguments and calculate score"""
    parser = argparse.ArgumentParser(description="Calculate Self-Attention Score")
    
    # Required arguments
    parser.add_argument("--fwd-latency", type=float, required=True,
                        help="Measured latency for forward pass (ns)")
    parser.add_argument("--fwd-base-latency", type=float, required=True,
                        help="Baseline latency for forward pass (ns)")
    parser.add_argument("--fwd-numerical-accuracy", type=str, required=True, choices=["pass", "fail"],
                        help="Whether forward numerical tests passed")
    
    parser.add_argument("--bwd-latency", type=float, required=True,
                        help="Measured latency for backward pass (ns)")
    parser.add_argument("--bwd-base-latency", type=float, required=True,
                        help="Baseline latency for backward pass (ns)")
    parser.add_argument("--bwd-numerical-accuracy", type=str, required=True, choices=["pass", "fail"],
                        help="Whether backward numerical tests passed")
    
    # Optional arguments
    parser.add_argument("--fwd-weight", type=float, default=0.4,
                        help="Weight for forward pass in combined score (default: 0.4)")
    parser.add_argument("--bwd-weight", type=float, default=0.6,
                        help="Weight for backward pass in combined score (default: 0.6)")
    parser.add_argument("--nki-flop-ratio", type=float, default=0.0,
                        help="Ratio of NKI FLOPs to total FLOPs (default: 0.0)")
    parser.add_argument("--output", type=str,
                        help="Path to save results as JSON")
    
    args = parser.parse_args()
    
    # Convert string accuracy to boolean
    fwd_numerical_accuracy = args.fwd_numerical_accuracy.lower() == "pass"
    bwd_numerical_accuracy = args.bwd_numerical_accuracy.lower() == "pass"
    
    # Calculate score
    results = calculate_score(
        args.fwd_latency, args.fwd_base_latency, fwd_numerical_accuracy,
        args.bwd_latency, args.bwd_base_latency, bwd_numerical_accuracy,
        args.fwd_weight, args.bwd_weight, args.nki_flop_ratio
    )
    
    # Add timestamp
    results["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Print results
    print_results(results)
    
    # Save results if output path is provided
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Return success if combined score is positive
    return 0 if results["combined"]["score"] > 0.0 else 1

if __name__ == "__main__":
    sys.exit(main())