"""Functions for exporting models to ONNX format."""

import os
import torch
import torch.onnx

def to_onnx(
    model, 
    input_shape, 
    output_path, 
    input_names=None,
    output_names=None,
    dynamic_axes=None,
    opset_version=13
):
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        input_shape: Shape of the input tensor (e.g., (1, 3, 224, 224))
        output_path: Path to save the ONNX model
        input_names: Names of the input tensors (default: ["input"])
        output_names: Names of the output tensors (default: ["output"])
        dynamic_axes: Dictionary of dynamic axes (default: None)
        opset_version: ONNX opset version (default: 13)
        
    Returns:
        str: Path to the exported ONNX model
    """
    if input_names is None:
        input_names = ["input"]
    
    if output_names is None:
        output_names = ["output"]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Create dummy input
    if isinstance(input_shape, tuple):
        dummy_input = torch.randn(*input_shape)
    else:
        dummy_input = torch.randn(input_shape)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"Model exported to {output_path}")
    return output_path 