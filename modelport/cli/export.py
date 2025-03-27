# cli/export.py
import typer
from modelport.core.exporter import export_model

def export_command(
    model_path: str = typer.Argument(..., help="Path to PyTorch model (.pt)"),
    output_path: str = typer.Option("modelport_export", help="Output directory"),
):
    """
    Export a PyTorch model to ONNX and generate a portable capsule.
    """
    export_model(model_path, output_path)
    typer.echo(f"✅ Exported model capsule to: {output_path}") 