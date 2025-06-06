from __future__ import annotations

import tomllib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Self, TypeVar

import lightning as L
from pydantic import BaseModel, ConfigDict, model_validator

# _T = TypeVar("_T")
# _T_co = TypeVar("_T_co", covariaint=True)

_LitModule = TypeVar("_LitModule", bound="L.LightningModule")
_LITDataModule = TypeVar("_LITDataModule", bound="L.LightningDataModule")


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
    def validate_datapack_compatibility(
        self, datapack_config: DatapackConfig[Any]
    ) -> DatapackConfig[Any]:
        """
        Validate that the Lightning module is compatible with the data module.
        This method should be implemented by subclasses to perform any necessary checks.
        """
        raise NotImplementedError("Subclasses must implement this method.")

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

    def validate_datapack_compatibility(
        self, datapack_config: DatapackConfig[Any]
    ) -> SerializedDatapackConfig:
        """
        Validate that the Lightning module is compatible with the data module.
        """
        assert isinstance(datapack_config, SerializedDatapackConfig), (
            "SerializedModuleConfig can only be used with SerializedDatapackConfig."
        )
        return datapack_config

    def create_module(self) -> L.LightningModule:
        """
        Create a Lightning module instance by loading it from the specified path.
        """
        raise NotImplementedError(
            "SerializedModuleConfig should never be used to start an experiment directly."
        )


class DatapackConfig(ABC, BaseModel, Generic[_LITDataModule]):
    config_path: str

    @abstractmethod
    def validate_module_compatibility(
        self, module_config: ModuleConfig[Any]
    ) -> ModuleConfig[Any]:
        """
        Validate that the Lightning module is compatible with the data module.
        This method should be implemented by subclasses to perform any necessary checks.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def create_datapack(
        self, full_config: ExperimentConfig[Any, _LITDataModule]
    ) -> _LITDataModule:
        """
        Create a Lightning data module instance.
        This method should be implemented by subclasses to return an instance of the Lightning data module.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class SerializedDatapackConfig(DatapackConfig[L.LightningDataModule]):
    """
    Configuration for a serialized Lightning data module.
    This class is used to load a Lightning data module from a specified path.
    """

    # Explicity allow extra config to go through, because this will be used to initialize the module
    model_config = ConfigDict(extra="allow")

    def validate_module_compatibility(
        self, module_config: ModuleConfig[Any]
    ) -> SerializedModuleConfig:
        """
        Validate that the Lightning module is compatible with the data module.
        """
        assert isinstance(module_config, SerializedModuleConfig), (
            "SerializedDatapackConfig can only be used with SerializedModuleConfig."
        )
        return module_config

    def create_datapack(
        self, full_config: ExperimentConfig[Any, _LITDataModule]
    ) -> L.LightningDataModule:
        """
        Create a Lightning data module instance by loading it from the specified path.
        """
        raise NotImplementedError(
            "SerializedDatapackConfig should never be used to start an experiment directly."
        )


class ExperimentConfig(BaseModel, Generic[_LitModule, _LITDataModule]):
    metadata: ExperimentMetadata
    # For now, include distributed config here.
    distributed: DistributedConfig

    module_config: ModuleConfig[_LitModule]
    datapack_config: DatapackConfig[_LITDataModule]

    # These maybe should be moved to module_config, but seem standard enough to keep here
    sequence_length: int
    batch_size: int

    @model_validator(mode="after")
    def check_compatibility(self) -> Self:
        """
        Validate that the module and datapack configurations are compatible.
        This method is called after the model is initialized to ensure compatibility.
        """
        self.module_config.validate_datapack_compatibility(self.datapack_config)
        self.datapack_config.validate_module_compatibility(self.module_config)
        return self


class SerializedExperimentConfig(
    ExperimentConfig[L.LightningModule, L.LightningDataModule]
):
    metadata: ExperimentMetadata
    # For now, include distributed config here. Controlling GPU allocation maybe should be its own thing but this is fine for now.
    distributed: DistributedConfig

    module_config: SerializedModuleConfig
    datapack_config: SerializedDatapackConfig


def build_experiment_config(experiment_config_toml: Path) -> ExperimentConfig[Any, Any]:
    # We import this dynamically, this lets us patch over and override this import in tests.
    from modeling.utils.dynamic_import import import_from_string

    with open(experiment_config_toml, "rb") as f:
        data = tomllib.load(f)
    serialized_exp_config = SerializedExperimentConfig.model_validate(data)

    serialized_module_config = serialized_exp_config.module_config
    serialized_datapack_config = serialized_exp_config.datapack_config

    module_config_path = serialized_module_config.config_path
    module_config_cls = import_from_string(module_config_path)
    assert issubclass(module_config_cls, ModuleConfig)
    module_config = module_config_cls.model_validate(
        serialized_module_config.model_dump()
    )

    datapack_config_path = serialized_datapack_config.config_path
    datapack_config_cls = import_from_string(datapack_config_path)
    assert issubclass(datapack_config_cls, DatapackConfig)
    datapack_config = datapack_config_cls.model_validate(
        serialized_datapack_config.model_dump()
    )

    return ExperimentConfig.model_validate(
        {
            **serialized_exp_config.model_dump(),
            "module_config": module_config,
            "datapack_config": datapack_config,
        }
    )


def serialize_experiment_config(
    experiment_config: ExperimentConfig[Any, Any], output_path: Path
) -> None:
    """
    Serialize the ExperimentConfig to a TOML file.
    """
    import tomli_w

    with open(output_path, "wb") as f:
        tomli_w.dump(experiment_config.model_dump(serialize_as_any=True), f)
