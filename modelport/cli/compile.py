"""
CLI for the ModelPort compile command.

This module provides command-line functionality for compiling models to native code.
"""

import os
import sys
import time
import typer
import logging
import platform
from typing import Optional, List, Dict, Any

from ..core.compiler import compile_model, SUPPORTED_ARCHS, SUPPORTED_DEVICES

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def compile_command(
    model_path: str = typer.Argument(..., help="Path to the ONNX model file to compile"),
    output_dir: str = typer.Option("modelport_native", "--output-dir", "-o", 
                                   help="Directory to save compiled model"),
    target_arch: Optional[str] = typer.Option(
        None, "--target-arch", "-a",
        help=f"Target architecture. Supported: {', '.join(SUPPORTED_ARCHS)}. Auto-detected if not specified."
    ),
    target_device: str = typer.Option(
        "cpu", "--target-device", "-d",
        help=f"Target device. Supported: {', '.join(SUPPORTED_DEVICES)}"
    ),
    opt_level: int = typer.Option(
        3, "--opt-level", "-O",
        help="Optimization level (0-3). Higher levels enable more optimizations.",
        min=0, max=3
    ),
    input_shape: Optional[str] = typer.Option(
        None, "--input-shape", "-s",
        help="Optional input shape override (comma-separated). Example: '1,3,224,224'"
    ),
    test: bool = typer.Option(
        True, "--test/--no-test",
        help="Test the compiled model after compilation"
    ),
    benchmark: bool = typer.Option(
        False, "--benchmark",
        help="Benchmark the compiled model after compilation"
    ),
    iterations: int = typer.Option(
        10, "--iterations", "-i",
        help="Number of iterations for benchmarking"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose output"
    ),
):
    """
    Compile an ONNX model to a native shared library.
    
    This command uses TVM to compile an ONNX model into a highly optimized
    platform-specific shared library. The compiled model can run without
    Python or ONNX Runtime dependencies.
    
    Example usage:
    
      # Basic compilation
      modelport compile model.onnx
      
      # Specify target architecture and device
      modelport compile model.onnx --target-arch arm64 --target-device cpu
      
      # Set optimization level
      modelport compile model.onnx --opt-level 3
      
      # Override input shape
      modelport compile model.onnx --input-shape 1,3,224,224
      
      # Run benchmarking after compilation
      modelport compile model.onnx --benchmark --iterations 100
    """
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate model path
    if not os.path.exists(model_path):
        typer.echo(f"Error: Model file not found: {model_path}")
        raise typer.Exit(code=1)
    
    # Check file extension
    _, ext = os.path.splitext(model_path)
    if ext.lower() != ".onnx":
        typer.echo(f"Error: Only ONNX models are supported for compilation. Got {ext}")
        raise typer.Exit(code=1)
    
    # Parse input shape if provided
    input_shapes = None
    if input_shape:
        try:
            # Simple parsing for single input
            shape = [int(dim) for dim in input_shape.split(",")]
            input_shapes = {"input": shape}
            logger.info(f"Using custom input shape: {shape}")
        except ValueError:
            typer.echo(f"Error: Invalid input shape format. Example: '1,3,224,224'")
            raise typer.Exit(code=1)
    
    # Show compilation configuration
    typer.echo("ModelPort Compiler Configuration:")
    typer.echo(f"  Model: {os.path.basename(model_path)}")
    typer.echo(f"  Output directory: {output_dir}")
    typer.echo(f"  Target architecture: {target_arch or 'auto-detect'}")
    typer.echo(f"  Target device: {target_device}")
    typer.echo(f"  Optimization level: {opt_level}")
    
    try:
        # Start timer
        start_time = time.time()
        
        # Compile the model
        typer.echo("\nCompiling model...")
        config = compile_model(
            model_path=model_path,
            output_dir=output_dir,
            target_arch=target_arch,
            target_device=target_device,
            opt_level=opt_level,
            input_shapes=input_shapes,
            test=test
        )
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Display results
        typer.echo("\nCompilation Results:")
        typer.echo(f"  Status: {'Success' if config else 'Failed'}")
        typer.echo(f"  Time taken: {total_time:.2f} seconds")
        typer.echo(f"  Output directory: {os.path.abspath(output_dir)}")
        
        # List the generated files
        if "compiled_files" in config:
            files = config["compiled_files"]
            typer.echo(f"  Generated files:")
            for key, filename in files.items():
                typer.echo(f"    - {key}: {filename}")
        
        # Show test results
        if test and "test_results" in config:
            test_results = config["test_results"]
            typer.echo("\nTest Results:")
            typer.echo(f"  Success: {test_results.get('success', False)}")
            if "inference_time" in test_results:
                typer.echo(f"  Inference time: {test_results['inference_time']:.4f} seconds")
            if "output_shapes" in test_results:
                typer.echo(f"  Output shapes: {test_results['output_shapes']}")
        
        # Run benchmarking if requested
        if benchmark:
            typer.echo("\nRunning benchmark...")
            total_time = 0
            
            # Import here to avoid circular imports
            from ..core.runtime import run_native_model
            
            for i in range(iterations):
                start = time.time()
                result = run_native_model(output_dir)
                end = time.time()
                iter_time = end - start
                total_time += iter_time
                
                if verbose:
                    typer.echo(f"  Iteration {i+1}/{iterations}: {iter_time:.4f} seconds")
            
            avg_time = total_time / iterations
            typer.echo(f"\nBenchmark Results ({iterations} iterations):")
            typer.echo(f"  Average inference time: {avg_time:.4f} seconds")
            typer.echo(f"  Throughput: {1.0/avg_time:.2f} inferences/second")
        
        typer.echo("\nCompilation complete! Use 'modelport run-native' to run the compiled model.")
        
    except Exception as e:
        typer.echo(f"Error: {str(e)}")
        if verbose:
            import traceback
            typer.echo(traceback.format_exc())
        raise typer.Exit(code=1) 