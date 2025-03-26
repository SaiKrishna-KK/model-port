import os
import subprocess
import shutil
import tempfile
from typing import Optional, List, Dict, Any
import json

def deploy_capsule(
    capsule_path: str,
    tag: str,
    platform: str = "linux/amd64,linux/arm64",
    push: bool = False,
    gpu: bool = False,
    registry: str = "docker.io"
) -> str:
    """
    Deploy a ModelPort capsule as a Docker image.
    
    Args:
        capsule_path: Path to the model capsule folder
        tag: Docker image tag (e.g., username/image:tag)
        platform: Target platform(s) to build for
        push: Whether to push the image to Docker Hub
        gpu: Whether to build GPU-enabled Docker image
        registry: Docker registry to push to
        
    Returns:
        str: The full image tag used for the deployment
    """
    # Validate the capsule path
    if not os.path.exists(capsule_path):
        raise ValueError(f"Capsule path does not exist: {capsule_path}")
    
    if not os.path.isdir(capsule_path):
        raise ValueError(f"Capsule path is not a directory: {capsule_path}")
    
    # Check required files
    required_files = ["model.onnx", "inference.py"]
    for file in required_files:
        if not os.path.exists(os.path.join(capsule_path, file)):
            raise ValueError(f"Capsule is missing required file: {file}")
    
    # Select the appropriate Dockerfile
    dockerfile_path = os.path.join(capsule_path, "runtime")
    
    if gpu:
        # Use GPU Dockerfile if available, otherwise create one
        gpu_dockerfile = os.path.join(dockerfile_path, "Dockerfile.gpu")
        if not os.path.exists(gpu_dockerfile):
            # Create GPU Dockerfile if it doesn't exist
            create_gpu_dockerfile(dockerfile_path)
        dockerfile = "Dockerfile.gpu"
    else:
        # For ARM64 (Apple Silicon) use arm64 Dockerfile, otherwise use x86_64
        import platform as sys_platform
        if "arm" in sys_platform.machine().lower():
            dockerfile = "Dockerfile.arm64"
        else:
            dockerfile = "Dockerfile.x86_64"
    
    dockerfile_full_path = os.path.join(dockerfile_path, dockerfile)
    if not os.path.exists(dockerfile_full_path):
        raise ValueError(f"Dockerfile not found: {dockerfile_full_path}")
    
    # Create .dockerignore if it doesn't exist
    dockerignore_path = os.path.join(capsule_path, ".dockerignore")
    if not os.path.exists(dockerignore_path):
        create_dockerignore(capsule_path)
    
    # Create or update capsule_spec.json
    update_capsule_spec(capsule_path, tag, platform, gpu)
    
    # Build the Docker image
    build_args = [
        "docker", "buildx", "build",
        "--platform", platform,
        "-f", dockerfile_full_path,
        "-t", tag,
    ]
    
    # Add --push flag if specified
    if push:
        build_args.append("--push")
    else:
        build_args.append("--load")
    
    # Add the capsule path as the build context
    build_args.append(capsule_path)
    
    # Run the build command
    try:
        subprocess.run(build_args, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Docker build failed with exit code {e.returncode}")
    
    return tag

def create_gpu_dockerfile(runtime_path: str) -> None:
    """
    Create a GPU-enabled Dockerfile if it doesn't exist.
    
    Args:
        runtime_path: Path to the runtime directory
    """
    os.makedirs(runtime_path, exist_ok=True)
    
    gpu_dockerfile = os.path.join(runtime_path, "Dockerfile.gpu")
    with open(gpu_dockerfile, "w") as f:
        f.write("""FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \\
    onnxruntime-gpu \\
    numpy

# Copy application files
COPY . /app
WORKDIR /app

# Run the inference script
CMD ["python3", "inference.py"]
""")

def create_dockerignore(capsule_path: str) -> None:
    """
    Create a .dockerignore file to exclude unnecessary files.
    
    Args:
        capsule_path: Path to the model capsule folder
    """
    dockerignore_path = os.path.join(capsule_path, ".dockerignore")
    with open(dockerignore_path, "w") as f:
        f.write("""# Git
.git
.gitignore

# Docker
.dockerignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Editors
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
""")

def update_capsule_spec(capsule_path: str, tag: str, platform: str, gpu: bool) -> None:
    """
    Create or update the capsule_spec.json file.
    
    Args:
        capsule_path: Path to the model capsule folder
        tag: Docker image tag
        platform: Target platform(s)
        gpu: Whether it's a GPU-enabled build
    """
    spec_path = os.path.join(capsule_path, "capsule_spec.json")
    
    # Load existing spec if it exists
    if os.path.exists(spec_path):
        with open(spec_path, "r") as f:
            spec = json.load(f)
    else:
        # Create a new spec
        spec = {
            "version": "1.0",
            "name": os.path.basename(os.path.abspath(capsule_path)),
            "created_at": None,  # Will be updated on deployment
            "framework": "onnx",
        }
    
    # Load config.json if it exists for more metadata
    config_path = os.path.join(capsule_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            spec.update({
                "framework": config.get("framework", "onnx"),
                "input_shape": config.get("input_shape", None),
                "input_dtype": config.get("input_dtype", None),
            })
    
    # Update deployment info
    import datetime
    spec.update({
        "deployment": {
            "image": tag,
            "platforms": platform.split(","),
            "gpu_enabled": gpu,
            "updated_at": datetime.datetime.now().isoformat(),
            "registry": tag.split("/")[0] if "/" in tag else "docker.io",
        }
    })
    
    # Write updated spec
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2) 