from __future__ import annotations

from pathlib import Path

import typer
from synapse.utils.logging import configure_logging

import logging
import os

import warnings
import builtins
from synapse.video_loader.main import AsyncTyper

logger = configure_logging(
    __name__,
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
)
app = AsyncTyper(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)


def get_rank():
    assert os.environ.get("LOCAL_RANK") is not None, (
        "LOCAL_RANK environment variable is not set. "
        "Make sure to run your script with torchrun or set LOCAL_RANK manually."
    )
    assert os.environ["LOCAL_RANK"].isdigit(), "LOCAL_RANK must be an integer"
    return int(os.environ["LOCAL_RANK"])


def silence_everything():
    # 1) Disable all logging calls up to and including CRITICAL
    logging.disable(logging.ERROR)
    # 2) Suppress all warnings
    warnings.filterwarnings("ignore")
    # 3) (Optional) Override print so nothing is printed
    builtins.print = lambda *args, **kwargs: None


if __name__ == "__main__":
    pass
    # if get_rank() != 0:
    #     print("Running in silent mode for rank", get_rank())
    #     silence_everything()


@app.command()
def run(
    config_path: str = typer.Argument(
        ..., help="Path to experiment configuration toml file"
    ),
    node_rank: int = typer.Option(
        ..., help="Node rank of the process (default is 0 for single-node runs)"
    ),
):
    """
    Run the modeling application with the specified configuration and extra arguments.

    Args:
        config_path (str): Path to the configuration file.
        extra_args (str): Additional arguments for the module.
    """
    local_rank = get_rank()
    if get_rank() != 0:
        print("Running in silent mode for rank", get_rank())
        silence_everything()
        logger.setLevel(logging.ERROR)  # Set logger to ERROR to reduce output

    try:
        from modeling.config.serde import (
            build_experiment_config,
        )
        from modeling.initialization import ExperimentInstance
        from modeling.config import InstanceConfig

        instance_config = InstanceConfig(
            node_rank=node_rank,
            device_rank=local_rank,
        )
        logger.debug(
            f"Running on instance: {instance_config.model_dump_json(serialize_as_any=True, indent=2)}"
        )

        experiment_config = build_experiment_config(Path(config_path))
        logger.info("Running with config")
        logger.info(experiment_config.model_dump_json(serialize_as_any=True, indent=2))
        with ExperimentInstance.init_experiment(
            exp_config=experiment_config, instance_config=instance_config
        ) as experiment:
            experiment.run()

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
    assert "LOCAL_RANK" not in os.environ, (
        "This command should not be run with torchrun. "
    )
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

    run_command = (
        f"torchrun --nproc_per_node={exp_config.run.distributed.devices_per_node}"
        f" --nnodes {exp_config.run.distributed.num_nodes}"
        " --node_rank 0"
        " --rdzv_endpoint=127.0.0.1:29500"
        f" src/modeling/main.py run {export_path} --node-rank 0"
    )

    # run_command = f"mdl run {export_path}"

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
