# core/exporter.py
import torch
import os
import shutil
import sys
import importlib.util
import inspect
import pathlib
from typing import Optional, Dict, Any
from .model_utils import (
    detect_framework,
    get_model_metadata,
    generate_requirements,
    validate_onnx_model,
    save_config
)

def export_model(
    model_path: str,
    output_dir: str,
    framework: Optional[str] = None,
    input_shape: Optional[str] = None,
    force: bool = False
) -> str:
    """
    Export a model to ONNX format and prepare a portable capsule.
    
    Args:
        model_path (str): Path to the model file
        output_dir (str): Directory where the exported model and assets will be stored
        framework (Optional[str]): Framework name (auto-detected if not provided)
        input_shape (Optional[str]): Input shape as comma-separated string (e.g., "1,3,224,224")
        force (bool): Whether to overwrite existing output directory
        
    Returns:
        str: Path to the output directory containing the exported model
    """
    if os.path.exists(output_dir) and not force:
        raise ValueError(f"Output directory {output_dir} already exists. Use --force to overwrite.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect framework if not provided
    if framework is None:
        framework = detect_framework(model_path)
    
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
    
    # Validate the ONNX model
    success, error_msg = validate_onnx_model(onnx_path)
    if not success:
        raise RuntimeError(f"ONNX model validation failed: {error_msg}")
    
    # Save metadata to config.json
    save_config(metadata, output_dir)
    
    # Generate requirements.txt
    requirements = generate_requirements()
    with open(os.path.join(output_dir, "requirements.txt"), 'w') as f:
        f.write(requirements)
    
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
dummy_input = np.random.rand(*config['input_shape']).astype(config['input_dtype'])
output = session.run(None, {input_name: dummy_input})

print("Inference output shapes:", [o.shape for o in output])
print("Inference successful!")
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
            
    return output_dir 