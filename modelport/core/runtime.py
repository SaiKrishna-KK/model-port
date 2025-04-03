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
    
    def __init__(self, model_path, **kwargs):
        """
        Initialize the model runner.
        
        Args:
            model_path: Path to the model
            **kwargs: Additional arguments for specific runners
        """
        self.model_path = model_path
        self.kwargs = kwargs
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize")
    
    def run(self, inputs):
        """
        Run inference on the model.
        
        Args:
            inputs: Model inputs
            
        Returns:
            Model outputs
        """
        raise NotImplementedError("Subclasses must implement run")
    
    def get_metadata(self):
        """
        Get metadata about the model.
        
        Returns:
            dict: Model metadata
        """
        return {
            "model_path": self.model_path,
            "runner_type": self.__class__.__name__
        }


class TVMRunner(ModelRunner):
    """Runner for TVM models."""
    
    def _initialize(self):
        """Initialize the TVM model."""
        try:
            import tvm
            from tvm.contrib import graph_executor
            import os
            import json
            
            # Check if directory or file
            if os.path.isdir(self.model_path):
                # Directory with lib, graph, and params
                model_dir = self.model_path
                
                # Load metadata if exists
                metadata_path = os.path.join(model_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                # Get paths
                lib_path = metadata.get("lib_path", os.path.join(model_dir, "model.so"))
                graph_path = metadata.get("graph_path", os.path.join(model_dir, "model.json"))
                params_path = metadata.get("params_path", os.path.join(model_dir, "model.params"))
                
                # Load components
                self.lib = tvm.runtime.load_module(lib_path)
                with open(graph_path, "r") as f:
                    self.graph = f.read()
                with open(params_path, "rb") as f:
                    self.params_bytes = f.read()
                
                # Create runtime
                device = self.kwargs.get("device", "cpu")
                if device == "cuda":
                    self.ctx = tvm.cuda()
                else:
                    self.ctx = tvm.cpu()
                
                # Create module
                self.module = graph_executor.create(self.graph, self.lib, self.ctx)
                
                # Load parameters
                import tvm.relay as relay
                params = relay.load_param_dict(self.params_bytes)
                self.module.load_params(params)
                
                # Store metadata
                self.metadata = metadata
            else:
                raise ValueError(f"Unsupported model path: {self.model_path}")
        except ImportError:
            raise ImportError(
                "TVM is required for TVM model execution. "
                "Please install it with: pip install apache-tvm==0.12.0 ml_dtypes==0.2.0"
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing TVM model: {e}")
    
    def run(self, inputs):
        """
        Run inference on the TVM model.
        
        Args:
            inputs: Dictionary of input name to numpy array
            
        Returns:
            list: Output tensors
        """
        # Set inputs
        for name, data in inputs.items():
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Ensure data is float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # Handle batch dimension
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            self.module.set_input(name, data)
        
        # Run inference
        self.module.run()
        
        # Get outputs
        outputs = []
        for i in range(self.module.get_num_outputs()):
            outputs.append(self.module.get_output(i).numpy())
        
        return outputs
    
    def get_metadata(self):
        """
        Get metadata about the TVM model.
        
        Returns:
            dict: Model metadata
        """
        base_metadata = super().get_metadata()
        if hasattr(self, "metadata"):
            return {**base_metadata, **self.metadata}
        return base_metadata


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
    runner = TVMRunner(model_dir, device=device)
    outputs = runner.run(input_data)
    
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
    runner = TVMRunner(model_dir, device=device)
    results = runner.benchmark(iterations, warmup, input_data, custom_shapes)
    
    return results 