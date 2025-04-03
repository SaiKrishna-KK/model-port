"""ModelPort - ML Model Deployment, Portability, and Compilation."""

__version__ = "2.0.0"

# Import submodules
from modelport import core
from modelport import export
from modelport import compile
from modelport import inference
from modelport import utils

# Convenience functions
from modelport.export import to_onnx
from modelport.compile import compile_model
from modelport.inference import run

__all__ = [
    "core",
    "export",
    "compile",
    "inference",
    "utils",
    "to_onnx",
    "compile_model",
    "run",
] 