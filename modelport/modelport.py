#!/usr/bin/env python3
# modelport.py
import typer
from cli.export import export_command
from cli.run import run_command

app = typer.Typer()
app.command()(export_command)
app.command()(run_command)

if __name__ == "__main__":
    app() 