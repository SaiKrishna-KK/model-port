import torch
import tensorflow as tf
import onnx
import numpy as np
from typing import Dict, Any, Tuple, Optional
import json
import os
import subprocess
from pathlib import Path

def detect_framework(model_path: str) -> str:
    """
    Detect the framework of the model file.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        str: Framework name ('pytorch', 'tensorflow', or 'onnx')
    """
    file_ext = Path(model_path).suffix.lower()
    
    if file_ext == '.onnx':
        return 'onnx'
    elif file_ext in ['.pt', '.pth', '.ckpt']:
        return 'pytorch'
    elif file_ext in ['.h5', '.pb', '.savedmodel']:
        return 'tensorflow'
    else:
        raise ValueError(f"Unsupported model format: {file_ext}")

def get_model_metadata(model_path: str, framework: str) -> Dict[str, Any]:
    """
    Extract metadata from the model including input/output shapes and dtypes.
    
    Args:
        model_path (str): Path to the model file
        framework (str): Framework name
        
    Returns:
        Dict[str, Any]: Model metadata
    """
    metadata = {
        "framework": framework,
        "input_shape": None,
        "input_dtype": None,
        "output_names": [],
        "output_shapes": [],
        "output_dtypes": []
    }
    
    if framework == 'pytorch':
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
    elif framework == 'onnx':
        model = onnx.load(model_path)
        input_info = model.graph.input[0]
        metadata["input_shape"] = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        metadata["input_dtype"] = onnx.TensorProto.DataType.Name(input_info.type.tensor_type.elem_type)
        
        for output in model.graph.output:
            metadata["output_names"].append(output.name)
            metadata["output_shapes"].append([dim.dim_value for dim in output.type.tensor_type.shape.dim])
            metadata["output_dtypes"].append(onnx.TensorProto.DataType.Name(output.type.tensor_type.elem_type))
    
    return metadata

def generate_requirements() -> str:
    """
    Generate requirements.txt from current environment.
    
    Returns:
        str: Contents of requirements.txt
    """
    try:
        result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
        return result.stdout
    except:
        return "onnxruntime>=1.8.0\nnumpy>=1.19.0"

def validate_onnx_model(model_path: str) -> Tuple[bool, str]:
    """
    Validate an ONNX model by attempting to run inference.
    
    Args:
        model_path (str): Path to the ONNX model
        
    Returns:
        Tuple[bool, str]: (Success status, Error message if failed)
    """
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        
        # Get input info
        input_info = session.get_inputs()[0]
        input_shape = input_info.shape
        input_dtype = input_info.type
        
        # Create dummy input
        dummy_input = np.random.rand(*input_shape).astype(input_dtype)
        
        # Run inference
        output = session.run(None, {input_info.name: dummy_input})
        
        return True, ""
    except Exception as e:
        return False, str(e)

def save_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Save configuration to config.json in the output directory.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        output_dir (str): Output directory path
    """
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2) 