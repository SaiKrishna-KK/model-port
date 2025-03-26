# cli/export.py
import typer
from modelport.core.exporter import export_model

def export_command(
    model_path: str = typer.Argument(..., help="Path to model file"),
    output_path: str = typer.Option("modelport_export", help="Output directory"),
    framework: str = typer.Option(None, help="Model framework (auto-detected if not provided)"),
    input_shape: str = typer.Option(None, help="Input shape as comma-separated string (e.g., 1,3,224,224)"),
    force: bool = typer.Option(False, help="Overwrite existing output directory"),
    test: bool = typer.Option(False, help="Test the exported model with a dummy input"),
):
    """
    Export a model to ONNX and generate a portable capsule.
    
    The framework is automatically detected based on the model file extension.
    You can override the input shape if needed.
    Use --test to validate the model after export.
    """
    try:
        if framework is None:
            typer.echo(f"Auto-detecting framework for {model_path}...")
        
        output_dir = export_model(
            model_path=model_path,
            output_dir=output_path,
            framework=framework,
            input_shape=input_shape,
            force=force,
            test=test
        )
        
        typer.echo(f"✅ Exported model capsule to: {output_path}")
        
        # If test flag is set, show that testing was performed
        if test:
            typer.echo("✅ Model validation successful")
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1) 