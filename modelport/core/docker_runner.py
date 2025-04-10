# core/docker_runner.py
import subprocess
import os

def run_capsule(path, arch):
    """
    Run a model capsule using Docker.
    
    Args:
        path (str): Path to the model capsule folder
        arch (str): Target architecture (e.g., "linux/amd64", "linux/arm64")
        
    Returns:
        bool: True if the container ran successfully, False otherwise
    """
    # Determine which Dockerfile to use based on architecture
    dockerfile = "Dockerfile.x86_64" if "amd64" in arch else "Dockerfile.arm64"
    full_path = os.path.abspath(path)
    
    # Check if the runtime directory and Dockerfile exist
    dockerfile_path = os.path.join(full_path, "runtime", dockerfile)
    if not os.path.exists(dockerfile_path):
        print(f"Error: Could not find Dockerfile at {dockerfile_path}")
        return False

    # Build the Docker image
    print(f"Building Docker image for {arch}...")
    build_result = subprocess.run([
        "docker", "buildx", "build",
        "--platform", arch,
        "-f", dockerfile_path,
        "-t", f"modelport_container_{arch.replace('/', '_')}",
        full_path,
        "--load"
    ])
    
    if build_result.returncode != 0:
        print("Docker build failed")
        return False
    
    # Run the container
    print(f"Running container on {arch}...")
    run_result = subprocess.run([
        "docker", "run", "--rm", f"modelport_container_{arch.replace('/', '_')}"
    ])
    
    return run_result.returncode == 0 