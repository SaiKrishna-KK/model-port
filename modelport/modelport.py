#!/usr/bin/env python3
# modelport.py - CLI entry point for ModelPort
import typer
from cli.export import export_command
from cli.run import run_command
from cli.deploy import deploy_command
from cli.compile import compile_command
from cli.run_native import run_native_command

# Create Typer app that can be used as a CLI or imported
app = typer.Typer(help="ModelPort: Export and run ML models anywhere")
app.command(name="export")(export_command)
app.command(name="run")(run_command)
app.command(name="deploy")(deploy_command)
app.command(name="compile")(compile_command)
app.command(name="run-native")(run_native_command)

def main():
    """Run the ModelPort CLI application"""
    app()

if __name__ == "__main__":
    main() 