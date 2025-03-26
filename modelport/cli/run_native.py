"""
CLI for the ModelPort run-native command.

This module provides command-line functionality for running compiled models.
"""

import os
import sys
import time
import typer
import json
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_native_command(
    model_dir: str = typer.Argument(..., help="Directory containing the compiled model"),
    input_shape: Optional[str] = typer.Option(
        None, "--input-shape", "-s",
        help="Custom input shape (comma-separated). Example: '1,3,224,224'"
    ),
    input_file: Optional[str] = typer.Option(
        None, "--input-file", "-f",
        help="Path to input data file (.npy or .json)"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output-file", "-o",
        help="Path to save output data (.npy)"
    ),
    iterations: int = typer.Option(
        1, "--iterations", "-i",
        help="Number of inference iterations to run"
    ),
    benchmark: bool = typer.Option(
        False, "--benchmark", "-b",
        help="Run inference benchmark"
    ),
    warmup: int = typer.Option(
        3, "--warmup", "-w",
        help="Number of warmup iterations for benchmarking"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose output"
    ),
):
    """
    Run inference using a compiled model.
    
    This command runs inference on a model that was previously compiled with
    'modelport compile'. It can use random test data or custom inputs.
    
    Example usage:
    
      # Basic inference with default settings
      modelport run-native modelport_native
      
      # Custom input shape
      modelport run-native modelport_native --input-shape 1,3,448,448
      
      # Run benchmark
      modelport run-native modelport_native --benchmark --iterations 100
      
      # Use custom input data
      modelport run-native modelport_native --input-file input.npy
      
      # Save output to file
      modelport run-native modelport_native --output-file output.npy
    """
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        typer.echo(f"Error: Model directory not found: {model_dir}")
        raise typer.Exit(code=1)
    
    # Check config file
    config_file = os.path.join(model_dir, "compile_config.json")
    if not os.path.exists(config_file):
        typer.echo(f"Error: Compiled model configuration not found: {config_file}")
        raise typer.Exit(code=1)
    
    # Load model configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        typer.echo(f"Error loading configuration: {str(e)}")
        raise typer.Exit(code=1)
    
    # Get target architecture
    target_arch = config.get("target_arch", "x86_64")
    
    # Check for model files
    lib_file = os.path.join(model_dir, f"model_{target_arch}.so")
    graph_file = os.path.join(model_dir, f"model_{target_arch}.json")
    params_file = os.path.join(model_dir, f"model_{target_arch}.params")
    
    for file_path in [lib_file, graph_file, params_file]:
        if not os.path.exists(file_path):
            typer.echo(f"Error: Required model file not found: {file_path}")
            raise typer.Exit(code=1)
    
    # Parse input shape if provided
    custom_input_shapes = None
    if input_shape:
        try:
            # Simple parsing for single input
            shape = [int(dim) for dim in input_shape.split(",")]
            custom_input_shapes = {"input": shape}
            logger.info(f"Using custom input shape: {shape}")
        except ValueError:
            typer.echo(f"Error: Invalid input shape format. Example: '1,3,224,224'")
            raise typer.Exit(code=1)
    
    # Load input data from file if provided
    custom_input_data = None
    if input_file:
        if not os.path.exists(input_file):
            typer.echo(f"Error: Input file not found: {input_file}")
            raise typer.Exit(code=1)
        
        try:
            if input_file.endswith('.npy'):
                custom_input_data = {"input": np.load(input_file)}
                logger.info(f"Loaded input data from: {input_file}")
            elif input_file.endswith('.json'):
                with open(input_file, 'r') as f:
                    data = json.load(f)
                    # Convert JSON data to numpy arrays
                    custom_input_data = {}
                    for name, values in data.items():
                        custom_input_data[name] = np.array(values, dtype=np.float32)
                logger.info(f"Loaded input data from: {input_file}")
            else:
                typer.echo(f"Error: Unsupported input file format. Use .npy or .json")
                raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Error loading input data: {str(e)}")
            raise typer.Exit(code=1)
    
    # Show run configuration
    typer.echo("ModelPort Runtime Configuration:")
    typer.echo(f"  Model directory: {model_dir}")
    typer.echo(f"  Target architecture: {target_arch}")
    typer.echo(f"  Iterations: {iterations}")
    if custom_input_shapes:
        typer.echo(f"  Custom input shape: {custom_input_shapes}")
    if custom_input_data:
        typer.echo(f"  Using input data from: {input_file}")
    
    try:
        # Import TVM here to avoid import errors if not installed
        try:
            import tvm
            from tvm.contrib import graph_executor
        except ImportError:
            typer.echo("Error: TVM is required for running compiled models.")
            typer.echo("Install with: pip install apache-tvm")
            raise typer.Exit(code=1)
        
        # Load the compiled model
        lib = tvm.runtime.load_module(lib_file)
        with open(graph_file, "r") as f:
            graph_json = f.read()
        with open(params_file, "rb") as f:
            params_bytes = f.read()
        
        # Create TVM runtime module
        device = config.get("target_device", "cpu")
        if device == "cuda":
            ctx = tvm.cuda()
        else:
            ctx = tvm.cpu()
        
        # Create graph executor
        module = graph_executor.GraphModule(lib["default"](ctx))
        
        # Load parameters
        module.load_params(params_bytes)
        
        # Create input data or use provided data
        input_data = custom_input_data
        if not input_data:
            # Get input shapes from config or use custom shapes
            input_shapes = custom_input_shapes or config.get("input_shapes", {"input": [1, 3, 224, 224]})
            
            # Create random input data
            input_data = {}
            for name, shape in input_shapes.items():
                input_data[name] = np.random.uniform(size=shape).astype(np.float32)
                
            typer.echo(f"  Using random input data with shapes:")
            for name, data in input_data.items():
                typer.echo(f"    - {name}: {data.shape}")
        
        # Warmup runs if benchmarking
        if benchmark and warmup > 0:
            typer.echo(f"\nWarming up for {warmup} iterations...")
            for i in range(warmup):
                # Set inputs
                for name, data in input_data.items():
                    module.set_input(name, tvm.nd.array(data, ctx))
                # Run inference
                module.run()
        
        # Run inference
        results = []
        total_time = 0.0
        
        typer.echo(f"\nRunning inference for {iterations} iterations...")
        for i in range(iterations):
            # Set inputs
            for name, data in input_data.items():
                module.set_input(name, tvm.nd.array(data, ctx))
            
            # Run inference and measure time
            start_time = time.time()
            module.run()
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            
            # Get outputs
            num_outputs = module.get_num_outputs()
            outputs = []
            
            for j in range(num_outputs):
                output = module.get_output(j).numpy()
                outputs.append(output)
            
            # Store results
            results.append({
                "outputs": outputs,
                "time": inference_time
            })
            
            if verbose or iterations == 1:
                typer.echo(f"  Iteration {i+1}/{iterations}: {inference_time:.4f} seconds")
        
        # Display inference results
        if not benchmark:
            typer.echo("\nInference Results:")
            typer.echo(f"  Number of outputs: {num_outputs}")
            
            for i, output in enumerate(results[0]["outputs"]):
                typer.echo(f"  Output {i} shape: {output.shape}")
                if output.size <= 10 or verbose:
                    # For small outputs or in verbose mode, print the values
                    typer.echo(f"  Output {i} values: {output.flatten()[:10]}...")
                else:
                    # Otherwise just print summary statistics
                    typer.echo(f"  Output {i} stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        # Save output to file if requested
        if output_file:
            try:
                # Save the first iteration's output
                output_data = results[0]["outputs"][0]
                np.save(output_file, output_data)
                typer.echo(f"  Output saved to: {output_file}")
            except Exception as e:
                typer.echo(f"Error saving output data: {str(e)}")
        
        # Display benchmark results
        if benchmark or iterations > 1:
            avg_time = total_time / iterations
            typer.echo(f"\nBenchmark Results ({iterations} iterations):")
            typer.echo(f"  Average inference time: {avg_time:.4f} seconds")
            typer.echo(f"  Throughput: {1.0/avg_time:.2f} inferences/second")
            
            # Additional statistics
            times = [result["time"] for result in results]
            typer.echo(f"  Min time: {min(times):.4f} seconds")
            typer.echo(f"  Max time: {max(times):.4f} seconds")
            typer.echo(f"  Standard deviation: {np.std(times):.4f} seconds")
        
    except Exception as e:
        typer.echo(f"Error: {str(e)}")
        if verbose:
            import traceback
            typer.echo(traceback.format_exc())
        raise typer.Exit(code=1) 