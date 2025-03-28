# core/exporter.py
import torch
import os
import shutil
import sys
import importlib.util
import inspect
import pathlib
from typing import Optional, Dict, Any
import json

# Import utility functions conditionally
def detect_framework(model_path: str) -> str:
    """Detect model framework based on file extension"""
    file_ext = pathlib.Path(model_path).suffix.lower()
    
    if file_ext == '.onnx':
        return 'onnx'
    elif file_ext in ['.pt', '.pth', '.ckpt']:
        return 'pytorch'
    elif file_ext in ['.h5', '.pb', '.savedmodel']:
        return 'tensorflow'
    else:
        raise ValueError(f"Unsupported model format: {file_ext}")

def get_model_metadata(model_path: str, framework: str) -> Dict[str, Any]:
    """Extract model metadata"""
    metadata = {
        "framework": framework,
        "input_shape": None,
        "input_dtype": None,
        "output_names": [],
        "output_shapes": [],
        "output_dtypes": []
    }
    
    if framework == 'pytorch':
        try:
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'forward'):
                # Get input shape from first layer
                for layer in model.modules():
                    if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                        if isinstance(layer, torch.nn.Conv2d):
                            metadata["input_shape"] = [1, layer.in_channels, 224, 224]
                        else:
                            metadata["input_shape"] = [1, layer.in_features]
                        metadata["input_dtype"] = str(next(model.parameters()).dtype)
                        break
        except Exception as e:
            print(f"Warning: Could not extract PyTorch metadata: {e}")
            # Set default values
            metadata["input_shape"] = [1, 3, 224, 224]
            metadata["input_dtype"] = "float32"
            
    # Default values if no metadata was extracted
    if metadata["input_shape"] is None:
        metadata["input_shape"] = [1, 3, 224, 224]
    if metadata["input_dtype"] is None:
        metadata["input_dtype"] = "float32"
        
    return metadata

def validate_onnx_model(model_path: str) -> tuple:
    """Validate ONNX model with a test run"""
    try:
        import onnxruntime as ort
        import numpy as np
        
        session = ort.InferenceSession(model_path)
        input_info = session.get_inputs()[0]
        input_shape = input_info.shape
        
        # Convert ONNX type to numpy dtype
        input_type = input_info.type
        if input_type == 'tensor(float)':
            numpy_dtype = np.float32
        elif input_type == 'tensor(double)':
            numpy_dtype = np.float64
        elif input_type == 'tensor(int64)':
            numpy_dtype = np.int64
        elif input_type == 'tensor(int32)':
            numpy_dtype = np.int32
        else:
            numpy_dtype = np.float32  # Default to float32
        
        # Create dummy input
        dummy_input = np.random.rand(*input_shape).astype(numpy_dtype)
        
        # Run inference
        output = session.run(None, {input_info.name: dummy_input})
        
        return True, ""
    except Exception as e:
        return False, str(e)

def save_config(metadata: Dict[str, Any], output_dir: str) -> None:
    """Save metadata to config.json"""
    config_file = os.path.join(output_dir, "config.json")
    
    # Normalize dtype if it's from torch
    if isinstance(metadata.get("input_dtype"), str) and metadata["input_dtype"].startswith("torch."):
        metadata["input_dtype"] = metadata["input_dtype"].replace("torch.", "")
    
    with open(config_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def export_model(
    model_path: str,
    output_dir: str,
    framework: Optional[str] = None,
    input_shape: Optional[str] = None,
    force: bool = False,
    test: bool = False
) -> str:
    """
    Export a model to ONNX format and prepare a portable capsule.
    
    Args:
        model_path (str): Path to the model file
        output_dir (str): Directory where the exported model and assets will be stored
        framework (Optional[str]): Framework name (auto-detected if not provided)
        input_shape (Optional[str]): Input shape as comma-separated string (e.g., "1,3,224,224")
        force (bool): Whether to overwrite existing output directory
        test (bool): Whether to test the exported model with a dummy input
        
    Returns:
        str: Path to the output directory containing the exported model
    """
    if os.path.exists(output_dir) and not force:
        raise ValueError(f"Output directory {output_dir} already exists. Use --force to overwrite.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect framework if not provided
    if framework is None:
        framework = detect_framework(model_path)
        print(f"Detected framework: {framework}")
    
    # Get model metadata
    metadata = get_model_metadata(model_path, framework)
    
    # Override input shape if provided
    if input_shape:
        try:
            metadata["input_shape"] = [int(x) for x in input_shape.split(",")]
        except ValueError:
            raise ValueError("Input shape must be comma-separated integers")
    
    # Export to ONNX if needed
    onnx_path = os.path.join(output_dir, "model.onnx")
    if framework == 'pytorch':
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        dummy_input = torch.randn(*metadata["input_shape"])
        torch.onnx.export(model, dummy_input, onnx_path)
    elif framework == 'onnx':
        shutil.copy(model_path, onnx_path)
    else:
        raise ValueError(f"Framework {framework} not yet supported for export")
    
    # Always validate the ONNX model during export to ensure basic functionality
    success, error_msg = validate_onnx_model(onnx_path)
    if not success:
        raise RuntimeError(f"ONNX model validation failed: {error_msg}")
    
    # If test flag is set, perform more extensive testing
    if test:
        print("Running model validation test...")
        try:
            import onnxruntime as ort
            import numpy as np
            import datetime
            
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0]
            input_shape = input_name.shape
            
            # Convert ONNX type to numpy dtype
            input_type = input_name.type
            if input_type == 'tensor(float)':
                numpy_dtype = np.float32
            elif input_type == 'tensor(double)':
                numpy_dtype = np.float64
            elif input_type == 'tensor(int64)':
                numpy_dtype = np.int64
            elif input_type == 'tensor(int32)':
                numpy_dtype = np.int32
            else:
                numpy_dtype = np.float32  # Default to float32
            
            # Generate random input data based on shape and type
            dummy_input = np.random.rand(*input_shape).astype(numpy_dtype)
            
            # Run inference to verify the model works
            outputs = session.run(None, {input_name.name: dummy_input})
            
            # Save test results to metadata
            metadata["test_results"] = {
                "success": True,
                "input_shape": list(input_shape),
                "output_shapes": [list(o.shape) for o in outputs],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            print(f"Model test passed. Input shape: {input_shape}, Output shapes: {[o.shape for o in outputs]}")
        except Exception as e:
            raise RuntimeError(f"Model test failed: {str(e)}")
    
    # Save metadata to config.json
    save_config(metadata, output_dir)
    
    # Generate requirements.txt
    try:
        import subprocess
        result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
        requirements = result.stdout
    except:
        requirements = """torch>=1.8.0
onnx>=1.12.0
onnxruntime>=1.8.0
numpy>=1.19.0
"""
    
    with open(os.path.join(output_dir, "requirements.txt"), 'w') as f:
        f.write(requirements)

    # Create capsule_spec.json if it doesn't exist
    create_capsule_spec(output_dir, metadata, framework)
    
    # Copy inference resources
    module_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(os.path.dirname(module_dir))
    
    # Copy inference script
    inference_src = os.path.join(package_dir, "modelport", "examples", "inference.py")
    inference_dst = os.path.join(output_dir, "inference.py")
    
    if os.path.exists(inference_src):
        shutil.copy(inference_src, inference_dst)
    else:
        # Try a relative path if the package structure is different
        alt_src = os.path.join(os.path.dirname(module_dir), "examples", "inference.py")
        if os.path.exists(alt_src):
            shutil.copy(alt_src, inference_dst)
        else:
            # Create a default inference script
            with open(inference_dst, 'w') as f:
                f.write("""import onnxruntime as ort
import numpy as np
import json

# Load model configuration
with open('config.json', 'r') as f:
    config = json.load(f)

print("Running inference on model.onnx...")
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# Create input with correct shape and dtype
input_shape = config['input_shape']
input_dtype = config['input_dtype']
dummy_input = np.random.rand(*input_shape).astype(input_dtype)
output = session.run(None, {input_name: dummy_input})

print("✅ Inference output shapes:", [o.shape for o in output])
print("✅ Inference successful!")
""")

    # Copy Docker templates
    templates_src = os.path.join(package_dir, "modelport", "templates")
    templates_dst = os.path.join(output_dir, "runtime")
    
    if os.path.exists(templates_src) and os.path.isdir(templates_src):
        shutil.copytree(templates_src, templates_dst, dirs_exist_ok=True)
    else:
        # Try a relative path if the package structure is different
        alt_src = os.path.join(os.path.dirname(module_dir), "templates")
        if os.path.exists(alt_src) and os.path.isdir(alt_src):
            shutil.copytree(alt_src, templates_dst, dirs_exist_ok=True)
        else:
            # Create a default Dockerfile
            os.makedirs(templates_dst, exist_ok=True)
            with open(os.path.join(templates_dst, "Dockerfile.x86_64"), 'w') as f:
                f.write("""FROM python:3.10-slim

RUN pip install onnxruntime numpy

COPY . /app
WORKDIR /app

CMD ["python", "inference.py"]
""")
            # Copy the same content for ARM64
            shutil.copy(
                os.path.join(templates_dst, "Dockerfile.x86_64"),
                os.path.join(templates_dst, "Dockerfile.arm64")
            )
            
            # Add GPU Dockerfile if needed
            with open(os.path.join(templates_dst, "Dockerfile.gpu"), 'w') as f:
                f.write("""FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \\
    onnxruntime-gpu \\
    numpy

# Copy application files
COPY . /app
WORKDIR /app

# Run the inference script
CMD ["python3", "inference.py"]
""")
    
    return output_dir

def create_capsule_spec(output_dir: str, metadata: Dict[str, Any], framework: str) -> None:
    """
    Create or update capsule_spec.json file.
    
    Args:
        output_dir: Path to the output directory
        metadata: Model metadata
        framework: Model framework
    """
    import importlib
    import datetime
    import json
    
    spec_path = os.path.join(output_dir, "capsule_spec.json")
    
    # Initialize default spec
    spec = {
        "version": "1.0",
        "name": os.path.basename(os.path.abspath(output_dir)),
        "framework": framework,
        "created_at": datetime.datetime.now().isoformat(),
        "input_shape": metadata.get("input_shape"),
        "input_dtype": metadata.get("input_dtype"),
        "runtime": {
            "supports_gpu": True,
            "supports_cpu": True,
            "supported_platforms": ["linux/amd64", "linux/arm64"],
            "onnx_runtime_version": ">=1.8.0"
        }
    }
    
    # Add test results if available
    if "test_results" in metadata:
        spec["test_results"] = metadata["test_results"]
    
    # Save spec file
    with open(spec_path, 'w') as f:
        json.dump(spec, f, indent=2) 