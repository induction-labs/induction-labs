from __future__ import annotations

from pathlib import Path

import typer
from synapse.utils.logging import configure_logging

import logging
import functools
import os

import warnings
import builtins
from typing import Optional
from synapse.video_loader.main import AsyncTyper

logger = configure_logging(
    __name__,
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
)
app = AsyncTyper(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)


@functools.lru_cache(maxsize=None)
def get_rank():
    import torch.distributed as dist

    if os.environ.get("LOCAL_RANK") is not None:
        # If LOCAL_RANK is set, use it to determine the rank
        return int(os.environ["LOCAL_RANK"])

    # If torch.distributed isn’t even set up, fall back to 0
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def silence_everything():
    # 1) Disable all logging calls up to and including CRITICAL
    logging.disable(logging.CRITICAL)
    # 2) Suppress all warnings
    warnings.filterwarnings("ignore")
    # 3) (Optional) Override print so nothing is printed
    builtins.print = lambda *args, **kwargs: None


if __name__ == "__main__":
    if get_rank() != 0:
        print("Running in silent mode for rank", get_rank())
        silence_everything()


def tensorboard_trace_handler(
    dir_name: str, worker_name: Optional[str] = None, use_gzip: bool = False
):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import os
    import socket
    import time

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        output_path = os.path.join(dir_name, file_name)
        prof.export_chrome_trace(output_path)
        logger.info(f"Exported profiling trace to {output_path}")

    return handler_fn


@app.async_command()
async def run(
    config_path: str = typer.Argument(
        ..., help="Path to experiment configuration toml file"
    ),
    # extra_args: str = typer.Option("", help="Additional arguments for the module"),
):
    """
    Run the modeling application with the specified configuration and extra arguments.

    Args:
        config_path (str): Path to the configuration file.
        extra_args (str): Additional arguments for the module.
    """
    if get_rank() != 0:
        print("Running in silent mode for rank", get_rank())
        silence_everything()
        logger.setLevel(logging.ERROR)  # Set logger to ERROR to reduce output

    try:
        from modeling.config.serde import (
            build_experiment_config,
        )
        from modeling.initialization import ExperimentInstance
        # import os

        experiment_config = build_experiment_config(Path(config_path))
        logger.info("Running with config")
        logger.info(experiment_config.model_dump_json(serialize_as_any=True, indent=2))
        with ExperimentInstance.init_experiment(
            exp_config=experiment_config,
        ) as experiment:
            await experiment.run()

    except Exception as e:
        if get_rank() == 0:
            raise e
        pass


@app.command()
def export(
    config_path: str = typer.Argument(
        ..., help="Path to the module configuration file"
    ),
    submit: bool = typer.Option(
        False, help="Run the experiment after exporting the configuration"
    ),
):
    """
    Export the module configuration to a file.

    Args:
        module (str): Path to the module configuration file.
    """
    assert get_rank() == 0, "Export command should only be run on rank 0"
    from modeling.config import ExperimentConfig
    from modeling.config.serde import (
        serialize_experiment_config,
    )
    from modeling.utils.dynamic_import import import_from_string
    from modeling.utils.exp_module_path import exp_module_path

    exp_config = import_from_string(config_path)
    assert isinstance(exp_config, ExperimentConfig), (
        f"{exp_config=} is not an ExperimentConfig"
    )
    export_path = exp_module_path(config_path, file_extension=".toml")
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # run_command = (
    #     f"torchrun --nproc_per_node={exp_config.run.distributed.devices_per_node}"
    #     f" --nnodes {exp_config.run.distributed.num_nodes}"
    #     " --node_rank 0"
    #     f" src/modeling/main.py run {export_path}"
    # )

    run_command = f"mdl run {export_path}"

    serialize_experiment_config(exp_config, export_path, eof_comments=run_command)
    logger.info(
        "##################################\n"
        f"Experiment configuration exported to: {export_path}\n"
        "Run the following command to execute the experiment:\n"
        f"{run_command}\n"
        "##################################"
    )

    if submit:
        logger.info("Running the experiment...")
        import subprocess

        subprocess.run(
            run_command,
            shell=True,
            check=True,
        )


if __name__ == "__main__":
    app()
