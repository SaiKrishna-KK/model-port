"""
This setup.py file is a thin wrapper that defers to setuptools.build_meta
configured in pyproject.toml. It exists primarily for backward compatibility.
"""

from setuptools import setup

# All metadata in pyproject.toml now
setup() 