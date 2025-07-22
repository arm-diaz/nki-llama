#!/usr/bin/env python3
import os
import sys
import torch
from pathlib import Path

def get_model_paths():
    """
    Determine the input and output paths based on environment variables or defaults.
    Returns: (bin_dir, output_bin)
    """
    # Check for environment variables first
    bin_dir = os.environ.get('BIN_MODEL_DIR')
    output_dir = os.environ.get('CONSOLIDATED_BIN_MODEL_DIR')
    
    if bin_dir and output_dir:
        # Use environment variables if both are set
        output_bin = os.path.join(output_dir, "pytorch_model.bin")
        return bin_dir, output_bin
    
    # Try to construct paths from MODEL_NAME if available
    model_name = os.environ.get('MODEL_NAME')
    if model_name:
        base_path = os.path.expanduser("~/nki-llama/src/fine-tune/model_assets")
        hf_weight_name = f"{model_name}_hf_weights_bin"
        bin_dir = os.path.join(base_path, hf_weight_name)
        output_bin = os.path.join(base_path, "pckpt", "pytorch_model.bin")
        
        if os.path.exists(bin_dir):
            return bin_dir, output_bin
    
    # Fall back to default paths
    default_paths = [
        {
            'bin_dir': "/home/ubuntu/nki-llama/src/fine-tune/model_assets/llama3-8B_hf_weights_bin",
            'output_bin': "/home/ubuntu/nki-llama/src/fine-tune/model_assets/pckpt/pytorch_model.bin"
        },
        {
            'bin_dir': "/home/ubuntu/nki-llama/src/fine-tune/model_assets/llama-3-1_8b_hf_weights_bin",
            'output_bin': "/home/ubuntu/nki-llama/src/fine-tune/model_assets/pckpt/pytorch_model.bin"
        },
        {
            'bin_dir': "/home/ubuntu/nki-llama/src/fine-tune/model_assets/llama-3-2_1b_hf_weights_bin",
            'output_bin': "/home/ubuntu/nki-llama/src/fine-tune/model_assets/pckpt/pytorch_model.bin"
        }
    ]
    
    # Check which default path exists
    for paths in default_paths:
        if os.path.exists(paths['bin_dir']):
            print(f"Using default path: {paths['bin_dir']}")
            return paths['bin_dir'], paths['output_bin']
    
    # If nothing works, return None to trigger error
    return None, None

def print_error_help():
    """Print helpful error message with suggestions."""
    print("\nError: Could not find the binary model directory!")
    print("\nPossible solutions:")
    print("\n1. Set environment variables (recommended):")
    print("   export BIN_MODEL_DIR=/path/to/your/model_hf_weights_bin")
    print("   export CONSOLIDATED_BIN_MODEL_DIR=/path/to/output/pckpt")
    print("\n2. Set MODEL_NAME environment variable:")
    print("   export MODEL_NAME=llama-3-1_8b")
    print("   (This will look for ~/nki-llama/src/fine-tune/model_assets/llama-3-1_8b_hf_weights_bin)")
    print("\n3. Run from the download script which sets these variables automatically")
    print("\n4. Ensure one of these default directories exists:")
    print("   - /home/ubuntu/nki-llama/src/fine-tune/model_assets/llama3-8B_hf_weights_bin")
    print("   - /home/ubuntu/nki-llama/src/fine-tune/model_assets/llama-3-1_8b_hf_weights_bin")
    print("   - /home/ubuntu/nki-llama/src/fine-tune/model_assets/llama-3-2_1b_hf_weights_bin")
    print("\nCurrent environment variables:")
    print(f"   BIN_MODEL_DIR: {os.environ.get('BIN_MODEL_DIR', 'Not set')}")
    print(f"   CONSOLIDATED_BIN_MODEL_DIR: {os.environ.get('CONSOLIDATED_BIN_MODEL_DIR', 'Not set')}")
    print(f"   MODEL_NAME: {os.environ.get('MODEL_NAME', 'Not set')}")

def main():
    # Get paths
    bin_dir, output_bin = get_model_paths()
    
    if not bin_dir or not os.path.exists(bin_dir):
        print_error_help()
        sys.exit(1)
    
    # Check if output already exists
    if os.path.exists(output_bin):
        print(f"Merged checkpoint already exists at {output_bin}")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Skipping merge operation.")
            sys.exit(0)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_bin)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Binary model directory: {bin_dir}")
    print(f"Output file: {output_bin}")
    print()
    
    # Check if there are any .bin files to merge
    bin_files = [f for f in os.listdir(bin_dir) if f.endswith(".bin")]
    if not bin_files:
        print(f"Error: No .bin files found in {bin_dir}")
        print("Make sure the convert_safetensors.py script has been run first.")
        sys.exit(1)
    
    print(f"Found {len(bin_files)} .bin files to merge")
    
    # We'll merge everything into this dict
    merged_state = {}
    
    # Always load on CPU for merging
    map_loc = "cpu"
    
    # Iterate in sorted order so names don't collide unpredictably
    for fname in sorted(bin_files):
        shard_path = os.path.join(bin_dir, fname)
        print(f"Loading shard: {fname}")
        
        try:
            shard = torch.load(shard_path, map_location=map_loc)
            
            # If the file wrapped weights under "state_dict", pull it out
            if isinstance(shard, dict) and "state_dict" in shard:
                shard = shard["state_dict"]
            
            # Merge into our master dict
            merged_state.update(shard)
            
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            sys.exit(1)
    
    # Save the flattened checkpoint
    print(f"\nSaving merged checkpoint to {output_bin}")
    print(f"Total parameters in merged checkpoint: {len(merged_state)}")
    
    try:
        torch.save(merged_state, output_bin)
        print("Done! Merge completed successfully.")
        
        # Print file size
        file_size = os.path.getsize(output_bin) / (1024 ** 3)  # Convert to GB
        print(f"Output file size: {file_size:.2f} GB")
        
    except Exception as e:
        print(f"Error saving merged checkpoint: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()