"""
ModelPort Compiler Module for TVM-based model compilation.

This module provides functionality to compile ONNX models to 
platform-specific native libraries using the TVM (Tensor Virtual Machine) framework.
"""

import os
import json
import time
import logging
import platform
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

# Import TVM-related packages
try:
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor
    HAS_TVM = True
except ImportError:
    HAS_TVM = False
    logging.warning("TVM not found. Native compilation will not be available.")

# Import ONNX-related packages
try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logging.warning("ONNX not found. ONNX model loading will not be available.")

# Setup logging
logger = logging.getLogger(__name__)

# Define supported architectures and devices
SUPPORTED_ARCHS = ["x86_64", "arm64", "aarch64"]
SUPPORTED_DEVICES = ["cpu", "cuda", "metal", "opencl"]
DEFAULT_OPT_LEVEL = 3  # Default optimization level

class ModelCompiler:
    """
    Class for compiling ONNX models to native code using TVM.
    """
    
    def __init__(self, 
                model_path: str, 
                output_dir: str = "modelport_native",
                target_arch: Optional[str] = None,
                target_device: str = "cpu",
                opt_level: int = DEFAULT_OPT_LEVEL):
        """
        Initialize the ModelCompiler.
        
        Args:
            model_path: Path to the ONNX model file
            output_dir: Directory to save compiled model
            target_arch: Target architecture (x86_64, arm64, etc.)
            target_device: Target device (cpu, cuda, etc.)
            opt_level: Optimization level (0-3)
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.target_arch = target_arch or self._detect_architecture()
        self.target_device = target_device
        self.opt_level = opt_level
        
        # Validate parameters
        self._validate_parameters()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        self.compile_config = {
            "source_model": os.path.basename(model_path),
            "target_arch": self.target_arch,
            "target_device": self.target_device,
            "opt_level": self.opt_level,
            "compile_time": None,
            "test_results": None
        }
    
    def _validate_parameters(self):
        """Validate the compiler parameters."""
        if not HAS_TVM:
            raise ImportError("TVM is required for native compilation. Install with: pip install apache-tvm")
        
        if not HAS_ONNX:
            raise ImportError("ONNX is required for model loading. Install with: pip install onnx onnxruntime")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Verify file is ONNX
        _, ext = os.path.splitext(self.model_path)
        if ext.lower() != ".onnx":
            raise ValueError(f"Only ONNX models are supported for compilation. Got file with extension: {ext}")
        
        # Validate architecture
        if self.target_arch not in SUPPORTED_ARCHS:
            raise ValueError(f"Unsupported architecture: {self.target_arch}. Supported: {SUPPORTED_ARCHS}")
        
        # Validate device
        if self.target_device not in SUPPORTED_DEVICES:
            raise ValueError(f"Unsupported device: {self.target_device}. Supported: {SUPPORTED_DEVICES}")
        
        # Validate optimization level
        if not (0 <= self.opt_level <= 3):
            raise ValueError(f"Optimization level must be between 0 and 3, got: {self.opt_level}")
    
    def _detect_architecture(self) -> str:
        """
        Detect the system architecture.
        
        Returns:
            The detected architecture (x86_64, arm64, etc.)
        """
        arch = platform.machine().lower()
        
        # Map common architecture names
        if arch in ["x86_64", "amd64", "x64"]:
            return "x86_64"
        elif arch in ["arm64", "aarch64", "armv8"]:
            if platform.system() == "Darwin":  # Apple Silicon
                return "arm64"
            else:
                return "aarch64"
        else:
            logger.warning(f"Unknown architecture: {arch}. Falling back to x86_64")
            return "x86_64"
    
    def _build_target(self) -> str:
        """
        Build the TVM target string based on architecture and device.
        
        Returns:
            TVM target string
        """
        # Base target string based on architecture
        if self.target_arch == "x86_64":
            base_target = "llvm -mcpu=core-avx2"
        elif self.target_arch in ["arm64", "aarch64"]:
            base_target = "llvm -mtriple=aarch64-linux-gnu"
            if platform.system() == "Darwin":  # macOS
                base_target = "llvm -mtriple=arm64-apple-darwin"
        else:
            base_target = "llvm"
        
        # Add device-specific options
        if self.target_device == "cuda":
            return f"{base_target} -device=cuda"
        elif self.target_device == "metal":
            return "metal"
        elif self.target_device == "opencl":
            return "opencl"
        else:  # Default to CPU
            return base_target
    
    def _get_model_info(self) -> Dict[str, Any]:
        """
        Extract model information from the ONNX model.
        
        Returns:
            Dictionary with model information
        """
        try:
            # Load ONNX model
            onnx_model = onnx.load(self.model_path)
            
            # Get input shapes and names
            input_info = {}
            for input_tensor in onnx_model.graph.input:
                name = input_tensor.name
                shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        # Dynamic dimension
                        shape.append(1)  # Default to 1 for dynamic dimensions
                    else:
                        shape.append(dim.dim_value)
                input_info[name] = shape
            
            # Get output shapes and names
            output_info = {}
            for output_tensor in onnx_model.graph.output:
                name = output_tensor.name
                shape = []
                for dim in output_tensor.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        # Dynamic dimension
                        shape.append(1)
                    else:
                        shape.append(dim.dim_value)
                output_info[name] = shape
            
            return {
                "inputs": input_info,
                "outputs": output_info,
                "model_name": os.path.basename(self.model_path).split(".")[0]
            }
        except Exception as e:
            logger.error(f"Error extracting model info: {e}")
            # Return minimal information
            return {
                "inputs": {"input": [1, 3, 224, 224]},  # Default values
                "outputs": {"output": [1, 1000]},
                "model_name": os.path.basename(self.model_path).split(".")[0]
            }
    
    def compile(self, input_shapes: Optional[Dict[str, List[int]]] = None) -> Dict[str, Any]:
        """
        Compile the ONNX model to a native library.
        
        Args:
            input_shapes: Optional override for input shapes
            
        Returns:
            Dictionary with compilation information
        """
        start_time = time.time()
        logger.info(f"Starting compilation of {self.model_path} for {self.target_arch}/{self.target_device}")
        
        try:
            # Get model information or use provided input shapes
            model_info = self._get_model_info()
            model_name = model_info["model_name"]
            
            # Override input shapes if provided
            if input_shapes:
                model_info["inputs"] = input_shapes
            
            # Save model information
            self.compile_config["model_info"] = model_info
            self.compile_config["input_shapes"] = model_info["inputs"]
            
            # Load ONNX model
            onnx_model = onnx.load(self.model_path)
            
            # Prepare input shapes for TVM
            input_shapes = {name: shape for name, shape in model_info["inputs"].items()}
            
            # Convert ONNX model to Relay IR
            logger.info("Converting ONNX model to Relay IR")
            mod, params = relay.frontend.from_onnx(onnx_model, input_shapes)
            
            # Build TVM target
            target = self._build_target()
            logger.info(f"Building for target: {target}")
            
            # Compile model
            with tvm.transform.PassContext(opt_level=self.opt_level):
                lib = relay.build(mod, target=target, params=params)
            
            # Save compiled model
            lib_file = os.path.join(self.output_dir, f"model_{self.target_arch}.so")
            graph_file = os.path.join(self.output_dir, f"model_{self.target_arch}.json")
            params_file = os.path.join(self.output_dir, f"model_{self.target_arch}.params")
            config_file = os.path.join(self.output_dir, "compile_config.json")
            
            # Save lib, graph, and params
            lib.export_library(lib_file)
            with open(graph_file, "w") as f:
                f.write(lib.get_graph_json())
            with open(params_file, "wb") as f:
                f.write(relay.save_param_dict(lib.get_params()))
            
            # Update and save compile config
            end_time = time.time()
            self.compile_config["compile_time"] = end_time - start_time
            self.compile_config["compiled_files"] = {
                "lib": os.path.basename(lib_file),
                "graph": os.path.basename(graph_file),
                "params": os.path.basename(params_file)
            }
            
            with open(config_file, "w") as f:
                json.dump(self.compile_config, f, indent=2)
            
            logger.info(f"Compilation successful. Files saved to {self.output_dir}")
            logger.info(f"Compilation time: {end_time - start_time:.2f} seconds")
            
            return self.compile_config
        
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            raise RuntimeError(f"Compilation failed: {e}")
    
    def test_compiled_model(self) -> Dict[str, Any]:
        """
        Test the compiled model with a dummy input.
        
        Returns:
            Dictionary with test results
        """
        try:
            # Check if files exist
            lib_file = os.path.join(self.output_dir, f"model_{self.target_arch}.so")
            graph_file = os.path.join(self.output_dir, f"model_{self.target_arch}.json")
            params_file = os.path.join(self.output_dir, f"model_{self.target_arch}.params")
            
            for file in [lib_file, graph_file, params_file]:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Required file not found: {file}")
            
            # Load compiled model
            lib = tvm.runtime.load_module(lib_file)
            with open(graph_file, "r") as f:
                graph_json = f.read()
            
            # Create TVM runtime module
            if self.target_device == "cuda":
                ctx = tvm.cuda()
            else:
                ctx = tvm.cpu()
            
            # Create graph executor
            module = graph_executor.GraphModule(lib["default"](ctx))
            
            # Create dummy input data
            input_shapes = self.compile_config.get("input_shapes", {"input": [1, 3, 224, 224]})
            dummy_inputs = {}
            
            for name, shape in input_shapes.items():
                dummy_inputs[name] = np.random.uniform(size=shape).astype("float32")
            
            # Set inputs and run
            start_time = time.time()
            for name, data in dummy_inputs.items():
                module.set_input(name, data)
            
            module.run()
            end_time = time.time()
            
            # Get outputs
            num_outputs = module.get_num_outputs()
            outputs = []
            output_shapes = []
            
            for i in range(num_outputs):
                output = module.get_output(i).numpy()
                outputs.append(output)
                output_shapes.append(list(output.shape))
            
            # Create test results
            test_results = {
                "success": True,
                "inference_time": end_time - start_time,
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
                "outputs_valid": all(not np.isnan(o).any() for o in outputs),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            
            # Update and save compile config
            self.compile_config["test_results"] = test_results
            config_file = os.path.join(self.output_dir, "compile_config.json")
            
            with open(config_file, "w") as f:
                json.dump(self.compile_config, f, indent=2)
            
            logger.info(f"Model test successful. Inference time: {test_results['inference_time']:.4f} seconds")
            return test_results
        
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            test_results = {
                "success": False,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            
            # Update and save compile config
            self.compile_config["test_results"] = test_results
            config_file = os.path.join(self.output_dir, "compile_config.json")
            
            with open(config_file, "w") as f:
                json.dump(self.compile_config, f, indent=2)
            
            return test_results


def compile_model(
    model_path: str, 
    output_dir: str = "modelport_native",
    target_arch: Optional[str] = None,
    target_device: str = "cpu",
    opt_level: int = DEFAULT_OPT_LEVEL,
    input_shapes: Optional[Dict[str, List[int]]] = None,
    test: bool = True
) -> Dict[str, Any]:
    """
    Compile an ONNX model to a native shared library.
    
    Args:
        model_path: Path to the ONNX model file
        output_dir: Directory to save compiled model
        target_arch: Target architecture (x86_64, arm64, etc.)
        target_device: Target device (cpu, cuda, etc.)
        opt_level: Optimization level (0-3)
        input_shapes: Optional override for input shapes
        test: Whether to test the compiled model
        
    Returns:
        Dictionary with compilation information
    """
    if not HAS_TVM:
        raise ImportError("TVM is required for native compilation. Install with: pip install apache-tvm")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    compiler = ModelCompiler(
        model_path=model_path,
        output_dir=output_dir,
        target_arch=target_arch,
        target_device=target_device,
        opt_level=opt_level
    )
    
    # Compile model
    config = compiler.compile(input_shapes)
    
    # Test compiled model if requested
    if test:
        test_results = compiler.test_compiled_model()
        config["test_results"] = test_results
    
    return config 