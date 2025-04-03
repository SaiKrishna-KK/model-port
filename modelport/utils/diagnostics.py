"""Diagnostic utilities for ModelPort."""

import sys
import platform
import importlib.util

def check_environment():
    """
    Check the environment for required dependencies.
    
    Returns:
        dict: Environment information
    """
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "dependencies": {}
    }
    
    # Check for required packages
    packages = [
        "torch",
        "onnx",
        "onnxruntime",
        "numpy",
        "tvm"
    ]
    
    for package in packages:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            try:
                module = importlib.import_module(package)
                if hasattr(module, "__version__"):
                    env_info["dependencies"][package] = module.__version__
                else:
                    env_info["dependencies"][package] = "installed (no version info)"
            except ImportError:
                env_info["dependencies"][package] = "found but not importable"
        else:
            env_info["dependencies"][package] = "not found"
    
    # Check TVM capabilities
    if env_info["dependencies"].get("tvm") not in ["not found", "found but not importable"]:
        try:
            import tvm
            
            # Check LLVM support
            env_info["tvm_llvm_enabled"] = tvm.target.Target.current().has_feature("llvm")
            
            # Check CUDA support
            env_info["tvm_cuda_enabled"] = tvm.runtime.enabled("cuda")
            
            # Check ml_dtypes
            try:
                import ml_dtypes
                env_info["ml_dtypes_version"] = ml_dtypes.__version__
                env_info["ml_dtypes_float4_e2m1fn_available"] = hasattr(ml_dtypes, "float4_e2m1fn")
            except ImportError:
                env_info["ml_dtypes_version"] = "not found"
                env_info["ml_dtypes_float4_e2m1fn_available"] = False
                
        except Exception as e:
            env_info["tvm_error"] = str(e)
    
    return env_info

def print_diagnostics():
    """Print diagnostic information about the environment."""
    env_info = check_environment()
    
    print("\n=== ModelPort Environment Diagnostics ===")
    print(f"Python version: {env_info['python_version']}")
    print(f"Platform: {env_info['platform']}")
    print(f"Architecture: {env_info['architecture']}")
    print(f"Processor: {env_info['processor']}")
    
    print("\n=== Dependencies ===")
    for package, version in env_info["dependencies"].items():
        print(f"{package}: {version}")
    
    if "tvm_llvm_enabled" in env_info:
        print("\n=== TVM Configuration ===")
        print(f"LLVM enabled: {env_info['tvm_llvm_enabled']}")
        print(f"CUDA enabled: {env_info['tvm_cuda_enabled']}")
        print(f"ml_dtypes version: {env_info['ml_dtypes_version']}")
        print(f"float4_e2m1fn available: {env_info['ml_dtypes_float4_e2m1fn_available']}")
    
    if "tvm_error" in env_info:
        print("\n=== TVM Error ===")
        print(env_info["tvm_error"])
    
    return env_info

if __name__ == "__main__":
    print_diagnostics() 