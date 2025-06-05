from __future__ import annotations

import tomllib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import lightning as L
from pydantic import BaseModel, ConfigDict

# _T = TypeVar("_T")
# _T_co = TypeVar("_T_co", covariaint=True)

_LitModule = TypeVar("_LitModule", bound="L.LightningModule")


class WandbConfig(BaseModel):
    project: str
    name: str

    @classmethod
    def mock_data(cls) -> WandbConfig:
        """
        Create a mock instance of WandbConfig for testing purposes.
        """
        return cls(project="test-project", name="test-experiment")


class ExperimentMetadata(BaseModel):
    wandb: WandbConfig
    output_dir: str

    @classmethod
    def mock_data(cls) -> ExperimentMetadata:
        """
        Create a mock instance of ExperimentMetadata for testing purposes.
        """
        return cls(
            wandb=WandbConfig.mock_data(),
            output_dir="/tmp/experiment_output",
        )

    # start_time:
    # commit_hash


class DistributedConfig(BaseModel):
    @classmethod
    def mock_data(cls) -> DistributedConfig:
        """
        Create a mock instance of DistributedConfig for testing purposes.
        """
        return cls()


class ModuleConfig(ABC, BaseModel, Generic[_LitModule]):
    config_path: str

    @abstractmethod
    def create_module(self) -> _LitModule:
        """
        Create a Lightning module instance.
        This method should be implemented by subclasses to return an instance of the Lightning module.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class SerializedModuleConfig(ModuleConfig[L.LightningModule]):
    """
    Configuration for a serialized Lightning module.
    This class is used to load a Lightning module from a specified path.
    """

    # Explicity allow extra config to go through, because this will be used to initialize the module
    model_config = ConfigDict(extra="allow")

    def create_module(self) -> L.LightningModule:
        """
        Create a Lightning module instance by loading it from the specified path.
        """
        raise NotImplementedError(
            "SerializedModuleConfig should never be used to start an experiment directly."
        )


class ExperimentConfig(BaseModel, Generic[_LitModule]):
    metadata: ExperimentMetadata
    # For now, include distributed config here.
    distributed: DistributedConfig

    module_config: ModuleConfig[_LitModule]


class SerializedExperimentConfig(ExperimentConfig[L.LightningModule]):
    metadata: ExperimentMetadata
    # For now, include distributed config here. Controlling GPU allocation maybe should be its own thing but this is fine for now.
    distributed: DistributedConfig

    module_config: SerializedModuleConfig


def build_experiment_config(experiment_config_toml: Path) -> ExperimentConfig[Any]:
    # We import this dynamically, this lets us patch over and override this import in tests.
    from modeling.utils.dynamic_import import import_from_string

    with open(experiment_config_toml, "rb") as f:
        data = tomllib.load(f)
    serialized_exp_config = SerializedExperimentConfig.model_validate(data)
    serialized_module_config = serialized_exp_config.module_config
    config_path = serialized_module_config.config_path
    module_config_cls = import_from_string(config_path)
    assert issubclass(module_config_cls, ModuleConfig)
    module_config = module_config_cls.model_validate(
        serialized_module_config.model_dump()
    )
    return ExperimentConfig(
        metadata=serialized_exp_config.metadata,
        distributed=serialized_exp_config.distributed,
        module_config=module_config,
    )
