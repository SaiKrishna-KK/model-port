from setuptools import setup, find_packages

with open("modelport/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="modelport",
    version="0.1.0",
    author="Sai Krishna Vishnumolakala",
    author_email="saikrishna.v1970@gmail.com",
    description="Export and run ML models anywhere",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SaiKrishna-KK/model-port",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "onnx",
        "onnxruntime",
        "typer[all]",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "modelport=modelport.modelport:app",
        ],
    },
    include_package_data=True,
    package_data={
        "modelport": ["templates/*", "examples/*"],
    },
)

if __name__ == "__main__":
    setup() 