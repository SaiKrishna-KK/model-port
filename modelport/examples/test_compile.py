#!/usr/bin/env python3
"""
Test script for ModelPort v2.0 native compilation.

This script demonstrates the workflow for compiling an ONNX model to a native
shared library using TVM and running inference on the compiled model.

Usage:
    1. First, export a model to ONNX:
       modelport export path/to/model.pt

    2. Then compile the ONNX model:
       python examples/test_compile.py path/to/exported/model.onnx
"""
import os
import sys
import time
import argparse
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to allow importing from modelport
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from modelport.core.compiler import (
        compile_model, test_compiled_model, get_system_arch
    )
except ImportError as e:
    logger.error(f"Error importing modelport compiler: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Test ModelPort v2.0 native compilation")
    parser.add_argument("model_path", type=str, help="Path to the ONNX model")
    parser.add_argument("--output-dir", "-o", type=str, default="modelport_native",
                        help="Output directory for compiled artifacts")
    parser.add_argument("--target-arch", "-a", type=str, default=None,
                        help="Target architecture (auto-detect if not specified)")
    parser.add_argument("--target-device", "-d", type=str, default="cpu",
                        help="Target device (cpu, cuda, metal, opencl)")
    parser.add_argument("--opt-level", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Optimization level (0-3)")
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="Run benchmark after compilation")
    parser.add_argument("--iterations", "-i", type=int, default=10,
                        help="Number of iterations for benchmarking")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file '{args.model_path}' not found")
        return 1
    
    # Check if model is ONNX
    if not args.model_path.lower().endswith(".onnx"):
        logger.error("Input model must be in ONNX format (.onnx extension)")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-detect architecture if not specified
    if args.target_arch is None:
        args.target_arch = get_system_arch()
        logger.info(f"Auto-detected architecture: {args.target_arch}")
    
    # Compile the model
    logger.info(f"Compiling model: {args.model_path}")
    logger.info(f"Target: {args.target_arch} ({args.target_device})")
    logger.info(f"Optimization level: {args.opt_level}")
    
    try:
        start_time = time.time()
        config = compile_model(
            model_path=args.model_path,
            output_dir=args.output_dir,
            target_arch=args.target_arch,
            target_device=args.target_device,
            opt_level=args.opt_level
        )
        end_time = time.time()
        
        compile_time = end_time - start_time
        logger.info(f"Compilation took {compile_time:.2f} seconds")
        
        logger.info(f"Model compiled successfully: {config['compiled_lib']}")
        
        # Test the compiled model
        logger.info("Testing compiled model...")
        test_results = test_compiled_model(args.output_dir)
        
        logger.info(f"Test passed: {test_results['num_outputs']} outputs generated")
        logger.info(f"Output shapes: {test_results['output_shapes']}")
        
        # Benchmark if requested
        if args.benchmark:
            logger.info(f"Running benchmark with {args.iterations} iterations...")
            
            total_time = 0
            times = []
            
            for i in range(args.iterations):
                start_time = time.time()
                test_compiled_model(args.output_dir)
                end_time = time.time()
                
                iter_time = end_time - start_time
                total_time += iter_time
                times.append(iter_time)
                
                if args.verbose:
                    logger.info(f"Iteration {i+1}/{args.iterations}: {iter_time:.6f} seconds")
            
            avg_time = total_time / args.iterations
            std_dev = np.std(times)
            
            logger.info(f"Benchmark results:")
            logger.info(f"  Average time: {avg_time:.6f} seconds")
            logger.info(f"  Standard deviation: {std_dev:.6f} seconds")
            logger.info(f"  Performance: {1/avg_time:.2f} inferences/second")
        
        logger.info(f"All tests completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during compilation or testing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 