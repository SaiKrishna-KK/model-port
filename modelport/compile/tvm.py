"""TVM-based model compilation functions."""

import os
import numpy as np

def compile_model(
    model_path,
    target_arch="x86_64",
    target_device="cpu",
    opt_level=3,
    output_dir=None
):
    """
    Compile a model using TVM.
    
    Args:
        model_path: Path to the ONNX model
        target_arch: Target architecture (e.g., "x86_64", "aarch64")
        target_device: Target device (e.g., "cpu", "cuda")
        opt_level: Optimization level (0-3)
        output_dir: Directory to save the compiled model
        
    Returns:
        str: Path to the compiled model directory
    """
    try:
        import tvm
        from tvm import relay
        from tvm.relay import frontend
    except ImportError:
        raise ImportError(
            "TVM is required for model compilation. "
            "Please install it with: pip install apache-tvm==0.12.0 ml_dtypes==0.2.0"
        )
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.splitext(model_path)[0] + "_tvm"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set target based on architecture and device
    if target_arch == "x86_64":
        if target_device == "cpu":
            target = "llvm -mcpu=core-avx2"
        elif target_device == "cuda":
            target = "cuda"
        else:
            raise ValueError(f"Unsupported device for x86_64: {target_device}")
    elif target_arch == "aarch64":
        if target_device == "cpu":
            target = "llvm -mtriple=aarch64-linux-gnu"
        else:
            raise ValueError(f"Unsupported device for aarch64: {target_device}")
    else:
        raise ValueError(f"Unsupported architecture: {target_arch}")
    
    # Load the ONNX model
    onnx_model = frontend.from_onnx(model_path)
    
    # Build the model
    with tvm.transform.PassContext(opt_level=opt_level):
        mod, params = onnx_model
        lib = relay.build(mod, target=target, params=params)
    
    # Save the compiled model
    lib_path = os.path.join(output_dir, "model.so")
    params_path = os.path.join(output_dir, "model.params")
    graph_path = os.path.join(output_dir, "model.json")
    
    lib.export_library(lib_path)
    with open(graph_path, "w") as f:
        f.write(lib.get_graph_json())
    with open(params_path, "wb") as f:
        f.write(relay.save_param_dict(lib.get_params()))
    
    # Save metadata
    metadata = {
        "target_arch": target_arch,
        "target_device": target_device,
        "opt_level": opt_level,
        "lib_path": lib_path,
        "graph_path": graph_path,
        "params_path": params_path
    }
    
    # Save metadata as JSON
    import json
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model compiled to {output_dir}")
    return output_dir 