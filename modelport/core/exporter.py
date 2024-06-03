# core/exporter.py
import torch
import os
import shutil

def export_model(model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Dummy model (replace with user-defined loading later)
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(output_dir, "model.onnx")
    torch.onnx.export(model, dummy_input, onnx_path)

    # Copy default inference script
    shutil.copy("examples/inference.py", os.path.join(output_dir, "inference.py"))

    # Copy Docker templates
    shutil.copytree("templates", os.path.join(output_dir, "runtime"), dirs_exist_ok=True) 