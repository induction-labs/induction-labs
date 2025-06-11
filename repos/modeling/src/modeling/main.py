from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


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
    from modeling.config.serde import (
        build_experiment_config,
    )
    from modeling.initialization import Initializer
    import torch

    torch.set_float32_matmul_precision("high")

    experiment_config = build_experiment_config(Path(config_path))
    # Here you would typically initialize and run your model training or evaluation
    print("Running with config")
    print(experiment_config)

    # Initialize the experiment configuration
    trainer, datapack, lit_module = Initializer.init_experiment(experiment_config)

    trainer.fit(
        lit_module,
        datamodule=datapack,
    )


@app.command()
def export(
    config_path: str = typer.Argument(
        ..., help="Path to the module configuration file"
    ),
):
    """
    Export the module configuration to a file.

    Args:
        module (str): Path to the module configuration file.
    """
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
        f" src/modeling/main.py run {export_path}"
    )

    serialize_experiment_config(exp_config, export_path, eof_comments=run_command)
    print("##################################")
    print(f"Experiment configuration exported to: {export_path}")
    print("Run the following command to execute the experiment:")
    print(run_command)
    print("##################################")


if __name__ == "__main__":
    app()
