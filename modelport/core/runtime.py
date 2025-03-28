"""
ModelPort Runtime Module for TVM-compiled models.

This module provides functionality to run models compiled with the ModelPort compiler.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

# Import TVM-related packages
try:
    import tvm
    from tvm.contrib import graph_executor
    HAS_TVM = True
except ImportError:
    HAS_TVM = False
    logging.warning("TVM not found. Native model execution will not be available.")

# Setup logging
logger = logging.getLogger(__name__)

class ModelRunner:
    """
    Class for running TVM-compiled models.
    """
    
    def __init__(self, 
                model_dir: str,
                device: str = "cpu"):
        """
        Initialize the ModelRunner.
        
        Args:
            model_dir: Directory containing the compiled model files
            device: Device to run the model on (cpu, cuda)
        """
        self.model_dir = model_dir
        self.device = device
        self.module = None
        self.input_info = None
        self.output_info = None
        
        # Validate and load the model
        self._load_model()
    
    def _load_model(self):
        """Load the compiled model and its configuration."""
        if not HAS_TVM:
            raise ImportError("TVM is required for running compiled models. Install with: pip install apache-tvm")
        
        # Check if model directory exists
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Load model configuration
        config_file = os.path.join(self.model_dir, "compile_config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Compiled model configuration not found: {config_file}")
        
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Get target architecture from config
        self.target_arch = self.config.get("target_arch", "x86_64")
        
        # Check for model files
        self.lib_file = os.path.join(self.model_dir, f"model_{self.target_arch}.so")
        self.graph_file = os.path.join(self.model_dir, f"model_{self.target_arch}.json")
        self.params_file = os.path.join(self.model_dir, f"model_{self.target_arch}.params")
        
        for file_path in [self.lib_file, self.graph_file, self.params_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required model file not found: {file_path}")
        
        # Load the compiled model
        lib = tvm.runtime.load_module(self.lib_file)
        with open(self.graph_file, "r") as f:
            self.graph_json = f.read()
        with open(self.params_file, "rb") as f:
            self.params_bytes = f.read()
        
        # Create TVM runtime module
        if self.device == "cuda":
            ctx = tvm.cuda()
        else:
            ctx = tvm.cpu()
        
        # Create graph executor
        self.module = graph_executor.GraphModule(lib["default"](ctx))
        
        # Load parameters
        self.module.load_params(self.params_bytes)
        
        # Get input and output information
        self.input_info = self.config.get("input_shapes", {})
        if "model_info" in self.config and "inputs" in self.config["model_info"]:
            self.input_info = self.config["model_info"]["inputs"]
        
        if "model_info" in self.config and "outputs" in self.config["model_info"]:
            self.output_info = self.config["model_info"]["outputs"]
        
        logger.info(f"Model loaded successfully from {self.model_dir}")
    
    def create_input_tensors(self, 
                           custom_input_data: Optional[Dict[str, np.ndarray]] = None,
                           custom_shapes: Optional[Dict[str, List[int]]] = None) -> Dict[str, np.ndarray]:
        """
        Create input tensors for the model.
        
        Args:
            custom_input_data: Dictionary mapping input names to numpy arrays
            custom_shapes: Dictionary mapping input names to shapes (for random data)
            
        Returns:
            Dictionary mapping input names to numpy arrays
        """
        # If custom input data is provided, use it
        if custom_input_data:
            # Validate that all required inputs are provided
            for input_name in self.input_info:
                if input_name not in custom_input_data:
                    logger.warning(f"Input '{input_name}' not provided in custom_input_data. Using random data.")
            return custom_input_data
        
        # Otherwise, create random input data
        input_shapes = self.input_info
        if custom_shapes:
            # Override with custom shapes
            input_shapes = {**input_shapes, **custom_shapes}
        
        # Create random tensor for each input
        input_data = {}
        for name, shape in input_shapes.items():
            input_data[name] = np.random.uniform(size=shape).astype(np.float32)
        
        return input_data
    
    def run(self, 
           input_data: Optional[Dict[str, np.ndarray]] = None,
           custom_shapes: Optional[Dict[str, List[int]]] = None) -> List[np.ndarray]:
        """
        Run inference on the model.
        
        Args:
            input_data: Dictionary mapping input names to numpy arrays
            custom_shapes: Dictionary mapping input names to shapes (for random data)
            
        Returns:
            List of output tensors
        """
        # Create input tensors if not provided
        if input_data is None:
            input_data = self.create_input_tensors(custom_shapes=custom_shapes)
        
        # Set inputs
        for name, data in input_data.items():
            if self.device == "cuda":
                ctx = tvm.cuda()
            else:
                ctx = tvm.cpu()
            self.module.set_input(name, tvm.nd.array(data, ctx))
        
        # Run inference
        self.module.run()
        
        # Get outputs
        num_outputs = self.module.get_num_outputs()
        outputs = []
        
        for i in range(num_outputs):
            output = self.module.get_output(i).numpy()
            outputs.append(output)
        
        return outputs
    
    def benchmark(self, 
                iterations: int = 10, 
                warmup: int = 3,
                input_data: Optional[Dict[str, np.ndarray]] = None,
                custom_shapes: Optional[Dict[str, List[int]]] = None) -> Dict[str, Any]:
        """
        Benchmark the model by running multiple iterations.
        
        Args:
            iterations: Number of iterations to run
            warmup: Number of warmup iterations
            input_data: Dictionary mapping input names to numpy arrays
            custom_shapes: Dictionary mapping input names to shapes (for random data)
            
        Returns:
            Dictionary with benchmark results
        """
        # Create input tensors if not provided
        if input_data is None:
            input_data = self.create_input_tensors(custom_shapes=custom_shapes)
        
        # Warmup runs
        logger.info(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            self.run(input_data)
        
        # Benchmark runs
        logger.info(f"Running {iterations} benchmark iterations...")
        import time
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            self.run(input_data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        results = {
            "iterations": iterations,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_time": std_time,
            "throughput": 1.0 / avg_time,
            "times": times.tolist()
        }
        
        return results


def run_native_model(
    model_dir: str,
    input_data: Optional[Dict[str, np.ndarray]] = None,
    custom_shapes: Optional[Dict[str, List[int]]] = None,
    device: str = "cpu"
) -> List[np.ndarray]:
    """
    Run inference on a compiled model.
    
    Args:
        model_dir: Directory containing the compiled model
        input_data: Dictionary mapping input names to numpy arrays
        custom_shapes: Dictionary mapping input names to shapes (for random data)
        device: Device to run the model on (cpu, cuda)
        
    Returns:
        List of output tensors
    """
    if not HAS_TVM:
        raise ImportError("TVM is required for running compiled models. Install with: pip install apache-tvm")
    
    # Create a model runner and run inference
    runner = ModelRunner(model_dir, device)
    outputs = runner.run(input_data, custom_shapes)
    
    return outputs


def benchmark_native_model(
    model_dir: str,
    iterations: int = 10,
    warmup: int = 3,
    input_data: Optional[Dict[str, np.ndarray]] = None,
    custom_shapes: Optional[Dict[str, List[int]]] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Benchmark a compiled model.
    
    Args:
        model_dir: Directory containing the compiled model
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
        input_data: Dictionary mapping input names to numpy arrays
        custom_shapes: Dictionary mapping input names to shapes (for random data)
        device: Device to run the model on (cpu, cuda)
        
    Returns:
        Dictionary with benchmark results
    """
    if not HAS_TVM:
        raise ImportError("TVM is required for running compiled models. Install with: pip install apache-tvm")
    
    # Create a model runner and run benchmark
    runner = ModelRunner(model_dir, device)
    results = runner.benchmark(iterations, warmup, input_data, custom_shapes)
    
    return results 