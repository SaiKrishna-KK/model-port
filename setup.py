"""
Backward compatibility setup.py file.

This file is not required for modern Python packaging with pyproject.toml,
but it's provided for compatibility with tools that don't yet support PEP 517/518.
"""

import setuptools

if __name__ == "__main__":
    setuptools.setup() 