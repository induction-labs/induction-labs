from __future__ import annotations

import logging

import typer
from synapse.utils.logging import configure_logging

logger = configure_logging(
    __name__,
    level=logging.DEBUG,
)

k8s_app = typer.Typer()


@k8s_app.command()
def submit(
    config_path: str = typer.Argument(..., help="File to submit"),
):
    """
    Submit runs to kubernetes kueue.
    """
