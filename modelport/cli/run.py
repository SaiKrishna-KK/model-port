# cli/run.py
import typer
from modelport.core.docker_runner import run_capsule

def run_command(
    capsule_path: str = typer.Argument(..., help="Path to model capsule folder"),
    arch: str = typer.Option("linux/amd64", help="Target architecture"),
):
    """
    Run the model capsule using Docker.
    """
    run_capsule(capsule_path, arch)
    typer.echo("🚀 Capsule launched!") 