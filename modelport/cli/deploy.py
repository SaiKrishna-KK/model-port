import typer
from modelport.core.deployer import deploy_capsule

def deploy_command(
    capsule_path: str = typer.Argument(..., help="Path to the model capsule folder"),
    tag: str = typer.Option(None, help="Docker image tag (e.g., username/image:tag)"),
    platform: str = typer.Option("linux/amd64,linux/arm64", help="Target platform(s) to build for"),
    push: bool = typer.Option(False, help="Push the image to Docker Hub after building"),
    gpu: bool = typer.Option(False, help="Build GPU-enabled Docker image"),
    registry: str = typer.Option("docker.io", help="Docker registry to push to"),
):
    """
    Deploy a ModelPort capsule as a Docker image.
    
    This builds and optionally pushes a Docker image of your model capsule.
    You can specify a custom tag, target platforms, and whether to push to a registry.
    """
    if not tag:
        # Extract directory name as default tag name
        import os
        dir_name = os.path.basename(os.path.abspath(capsule_path))
        tag = f"modelport/{dir_name}:latest"
        typer.echo(f"No tag specified, using default: {tag}")
    
    try:
        image_tag = deploy_capsule(
            capsule_path=capsule_path,
            tag=tag,
            platform=platform,
            push=push,
            gpu=gpu,
            registry=registry
        )
        if push:
            typer.echo(f"🚀 Deployed capsule as {image_tag} to {registry}")
        else:
            typer.echo(f"✅ Built Docker image: {image_tag}")
            typer.echo("  To push to Docker Hub, use the --push flag")
            typer.echo(f"  To run locally: docker run --rm {image_tag}")
    except Exception as e:
        typer.echo(f"❌ Deployment failed: {str(e)}", err=True)
        raise typer.Exit(1) 