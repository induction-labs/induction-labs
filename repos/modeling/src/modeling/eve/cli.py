from __future__ import annotations

from synapse.utils.async_typer import AsyncTyper

from modeling.eve.load_balance_server import cli_app as lb_app
from modeling.eve.os_world.main import app as osworld_app
from modeling.eve.run_procs import app as run_procs_app
from modeling.eve.vllm_server import app as vllm_app

app = AsyncTyper()

# Add sub-applications
app.add_typer(vllm_app, name="vllm")
app.add_typer(lb_app, name="lb")
app.add_typer(osworld_app, name="osworld")
app.add_typer(run_procs_app, name="run-procs")


if __name__ == "__main__":
    app()
