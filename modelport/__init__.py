"""
ModelPort: Export and run ML models anywhere.

This package allows you to:
1. Export PyTorch models to ONNX format
2. Package models with inference code
3. Run models on different architectures using Docker
"""

__version__ = "0.1.0"
__author__ = "Sai Krishna Vishnumolakala"
__email__ = "saikrishna.v1970@gmail.com"

from modelport.core.exporter import export_model
from modelport.core.docker_runner import run_capsule
from modelport.core.deployer import deploy_capsule

# Only import compile_model if TVM is available
try:
    from modelport.core.compiler import compile_model
    HAS_TVM = True
except ImportError:
    HAS_TVM = False
    compile_model = None

# Expose key functions directly at package level
__all__ = [
    "export_model",
    "run_capsule",
    "deploy_capsule",
    "compile_model",
    "HAS_TVM",
] 