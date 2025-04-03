"""Functions for running inference on compiled models."""

import os
import json
import numpy as np

def run(
    model_dir,
    input_data,
    device="cpu"
):
    """
    Run inference on a compiled model.
    
    Args:
        model_dir: Directory containing the compiled model
        input_data: Dictionary of input tensors (name -> numpy array)
        device: Device to run inference on (default: "cpu")
        
    Returns:
        list: List of output tensors
    """
    try:
        import tvm
    except ImportError:
        raise ImportError(
            "TVM is required for inference. "
            "Please install it with: pip install apache-tvm==0.12.0 ml_dtypes==0.2.0"
        )
    
    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        # If no metadata, use default paths
        metadata = {
            "lib_path": os.path.join(model_dir, "model.so"),
            "graph_path": os.path.join(model_dir, "model.json"),
            "params_path": os.path.join(model_dir, "model.params")
        }
    
    # Load compiled model
    lib_path = metadata.get("lib_path", os.path.join(model_dir, "model.so"))
    graph_path = metadata.get("graph_path", os.path.join(model_dir, "model.json"))
    params_path = metadata.get("params_path", os.path.join(model_dir, "model.params"))
    
    if not os.path.exists(lib_path) or not os.path.exists(graph_path) or not os.path.exists(params_path):
        raise FileNotFoundError(f"Model files not found in directory: {model_dir}")
    
    # Load the model
    lib = tvm.runtime.load_module(lib_path)
    with open(graph_path, "r") as f:
        graph = f.read()
    
    # Load params
    with open(params_path, "rb") as f:
        params_bytes = f.read()
    
    # Create TVM runtime module
    if device == "cuda":
        ctx = tvm.cuda()
    else:
        ctx = tvm.cpu()
    
    # Create module
    from tvm.contrib import graph_executor
    module = graph_executor.create(graph, lib, ctx)
    
    # Load parameters
    import tvm.relay as relay
    params = relay.load_param_dict(params_bytes)
    module.load_params(params)
    
    # Set inputs
    for name, data in input_data.items():
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Ensure data is float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        module.set_input(name, data)
    
    # Run inference
    module.run()
    
    # Get outputs
    outputs = []
    for i in range(module.get_num_outputs()):
        outputs.append(module.get_output(i).numpy())
    
    return outputs 