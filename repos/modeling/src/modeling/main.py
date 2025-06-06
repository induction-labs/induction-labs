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
    from modeling.config import build_experiment_config

    experiment_config = build_experiment_config(Path(config_path))

    # Here you would typically initialize and run your model training or evaluation
    print("Running with config")
    print(experiment_config)


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
    from modeling.config import ExperimentConfig, serialize_experiment_config
    from modeling.experiments import transform_module_path
    from modeling.utils.dynamic_import import import_from_string

    exp_config = import_from_string(config_path)
    assert isinstance(exp_config, ExperimentConfig), (
        f"{exp_config=} is not an ExperimentConfig"
    )
    export_path = transform_module_path(config_path, file_extension=".toml")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    serialize_experiment_config(exp_config, export_path)
    print("##################################")
    print(f"Experiment configuration exported to: {export_path}")
    print("Run the following command to execute the experiment:")
    print(f"mdl run {export_path}")
    print("##################################")


if __name__ == "__main__":
    app()
