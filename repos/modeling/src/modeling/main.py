from __future__ import annotations

from pathlib import Path

import typer
from synapse.utils.logging import configure_logging

import logging
import functools
import os

import warnings
import builtins

logger = configure_logging(
    __name__,
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
)
app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)


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


@app.command()
def run(
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
        from modeling.initialization import Initializer
        import torch

        logger.debug("Initializing the modeling application...")

        torch.set_float32_matmul_precision("high")

        experiment_config = build_experiment_config(Path(config_path))
        # Here you would typically initialize and run your model training or evaluation
        logger.info("Running with config")
        logger.info(experiment_config.model_dump_json(serialize_as_any=True, indent=2))

        # Initialize the experiment configuration
        with Initializer.init_experiment(experiment_config) as (
            trainer,
            datapack,
            lit_module,
        ):
            logger.debug("Experiment initialized successfully.")

            trainer.fit(
                lit_module,
                datamodule=datapack,
            )
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
