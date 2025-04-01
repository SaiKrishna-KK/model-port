"""
ModelPort - Export and run machine learning models anywhere.

This package allows you to:
1. Export PyTorch models to ONNX format
2. Package models with inference code
3. Run models on different architectures using Docker
"""

__version__ = "0.1.0"

from modelport.core.exporter import export_model
from modelport.core.docker_runner import run_capsule

# Expose key functions directly at package level
__all__ = ["export_model", "run_capsule"] 