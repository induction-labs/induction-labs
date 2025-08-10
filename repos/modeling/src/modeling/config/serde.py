from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli

from . import DatapackConfig, ExperimentConfig, ModuleConfig, SerializedExperimentConfig


def build_experiment_config(experiment_config_toml: Path) -> ExperimentConfig[Any]:
    # We import this dynamically, this lets us patch over and override this import in tests.
    from modeling.utils.dynamic_import import import_from_string

    with open(experiment_config_toml, "rb") as f:
        data = tomli.load(f)
    try:
        serialized_exp_config = SerializedExperimentConfig.model_validate(data)
    except TypeError as e:
        raise ValueError(
            f"Failed to validate the experiment config from {experiment_config_toml}. "
            f"{data=}"
        ) from e

    module_config_path = serialized_exp_config.module.config_path
    module_config_cls = import_from_string(module_config_path)
    assert issubclass(module_config_cls, ModuleConfig)
    module_config = module_config_cls.model_validate(
        serialized_exp_config.module.model_dump(serialize_as_any=True)
    )

    train_datapack_config_path = serialized_exp_config.train_datapack.config_path
    train_datapack_config_cls = import_from_string(train_datapack_config_path)
    assert issubclass(train_datapack_config_cls, DatapackConfig)
    train_datapack_config = train_datapack_config_cls.model_validate(
        serialized_exp_config.train_datapack.model_dump(serialize_as_any=True)
    )

    validation_datapack_config_path = (
        serialized_exp_config.validation_datapack.config_path
    )
    validation_datapack_config_cls = import_from_string(validation_datapack_config_path)
    assert issubclass(validation_datapack_config_cls, DatapackConfig)
    validation_datapack_config = validation_datapack_config_cls.model_validate(
        serialized_exp_config.validation_datapack.model_dump(serialize_as_any=True)
    )

    return ExperimentConfig.model_validate(
        {
            **serialized_exp_config.model_dump(),
            "module": module_config,
            "train_datapack": train_datapack_config,
            "validation_datapack": validation_datapack_config,
        }
    )


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
