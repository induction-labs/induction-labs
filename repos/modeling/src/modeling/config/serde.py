from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli

from . import ExperimentConfig


def build_experiment_config(experiment_config_toml: Path) -> ExperimentConfig[Any]:
    # We import this dynamically, this lets us patch over and override this import in tests.

    with open(experiment_config_toml, "rb") as f:
        data = tomli.load(f)
    try:
        exp_config = ExperimentConfig.model_validate(data)
        return exp_config
    except TypeError as e:
        raise ValueError(
            f"Failed to validate the experiment config from {experiment_config_toml}. "
            f"{data=}"
        ) from e


def serialize_experiment_config(
    experiment_config: ExperimentConfig[Any],
    output_path: Path,
    eof_comments: str = "",
) -> None:
    """
    Serialize the ExperimentConfig to a TOML file. Adds eof_comments to the end of the file.
    """
    toml_str = experiment_config.serialize_to_toml()

    with open(output_path, "w") as f:
        f.write(toml_str)

    if eof_comments:
        # Write each line of eof_comments to the end of the file
        comment_lines = eof_comments.strip().split("\n")
        with open(output_path, "a") as f:
            f.write("\n")
            for line in comment_lines:
                f.write(f"# {line}\n")
