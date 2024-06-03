# core/docker_runner.py
import subprocess
import os

def run_capsule(path, arch):
    dockerfile = "Dockerfile.x86_64" if "amd64" in arch else "Dockerfile.arm64"
    full_path = os.path.abspath(path)

    subprocess.run([
        "docker", "buildx", "build",
        "--platform", arch,
        "-f", os.path.join(full_path, "runtime", dockerfile),
        "-t", "modelport_container",
        full_path,
        "--load"
    ])

    subprocess.run([
        "docker", "run", "--rm", "modelport_container"
    ]) 