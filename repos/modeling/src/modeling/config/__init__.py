from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Generic, Self, TypeVar

import lightning as L
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modeling.utils.git import get_git_commit_sha, get_git_commit_sha_short

from .distributed import DistributedConfig
from .wandb import WandbConfig

# _T = TypeVar("_T")
# _T_co = TypeVar("_T_co", covariaint=True)

# _LitModule = TypeVar("_LitModule", bound="L.LightningModule", covariant=True)
_LITDataModule = TypeVar("_LITDataModule", bound="L.LightningDataModule")


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

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def reset_loaded_at(self) -> None:
        """
        Reset the loaded_at timestamp to the current time.
        This is called when the config is loaded from toml.
        """
        self.loaded_at = datetime.now(UTC)

    commit: str = Field(default_factory=get_git_commit_sha)
    commit_short: str = Field(default_factory=get_git_commit_sha_short)


class ModuleConfig(ABC, BaseModel):
    config_path: str

    @model_validator(mode="after")
    def check_config_path(self) -> Self:
        """
        Validate that the module and datapack configurations are compatible.
        This method is called after the model is initialized to ensure compatibility.
        """
        assert self.config_path == (
            self.__class__.__module__ + "." + self.__class__.__name__
        ), (
            f"ModuleConfig config_path {self.config_path} does not match expected path "
            f"{self.__class__.__module__}.{self.__class__.__name__}."
        )

        return self

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
    def create_module(self) -> L.LightningModule:
        """
        Create a Lightning module instance.
        This method should be implemented by subclasses to return an instance of the Lightning module.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class SerializedModuleConfig(ModuleConfig):
    """
    Configuration for a serialized Lightning module.
    This class is used to load a Lightning module from a specified path.
    """

    # Explicity allow extra config to go through, because this will be used to initialize the module
    model_config = ConfigDict(extra="allow")

    def check_config_path(self) -> Self:
        # SerializedModuleConfig does not need to check the config_path,
        # as it is expected to be loaded from a path specified in the config.
        return self

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

    @model_validator(mode="after")
    def check_config_path(self) -> Self:
        """
        Validate that the module and datapack configurations are compatible.
        This method is called after the model is initialized to ensure compatibility.
        """
        assert self.config_path == (
            self.__class__.__module__ + "." + self.__class__.__name__
        ), (
            f"DatapackConfig config_path {self.config_path} does not match expected path "
            f"{self.__class__.__module__}.{self.__class__.__name__}."
        )

        return self

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
        self, full_config: ExperimentConfig[_LITDataModule]
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

    def check_config_path(self) -> Self:
        # SerializedDatapackConfig does not need to check the config_path,
        # as it is expected to be loaded from a path specified in the config.
        return self

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
        self, full_config: ExperimentConfig[_LITDataModule]
    ) -> L.LightningDataModule:
        """
        Create a Lightning data module instance by loading it from the specified path.
        """
        raise NotImplementedError(
            "SerializedDatapackConfig should never be used to start an experiment directly."
        )


class RunConfig(BaseModel):
    """
    Configuration for a run.
    This class is used to configure the run settings for an experiment.
    """

    num_epochs: int = 1
    steps_per_epoch: int  # Number of steps per epoch
    # save_interval: int = 1000  # How often to save checkpoints
    seed: int = 42  # Random seed for reproducibility
    sequence_length: int  # Default sequence length
    batch_size: int  # Default batch size

    @classmethod
    def mock_data(cls) -> RunConfig:
        """
        Create a mock instance of RunConfig for testing purposes.
        """
        return cls(
            num_epochs=1,
            steps_per_epoch=100,
            seed=42,
            sequence_length=1024,
            batch_size=4,
        )


class ExperimentConfig(BaseModel, Generic[_LITDataModule]):
    metadata: ExperimentMetadata
    # For now, include distributed config here.
    distributed: DistributedConfig

    module_config: ModuleConfig
    datapack_config: DatapackConfig[_LITDataModule]

    # These maybe should be moved to module_config, but seem standard enough to keep here
    run_config: RunConfig

    @model_validator(mode="after")
    def check_compatibility(self) -> Self:
        """
        Validate that the module and datapack configurations are compatible.
        This method is called after the model is initialized to ensure compatibility.
        """
        self.module_config.validate_datapack_compatibility(self.datapack_config)
        self.datapack_config.validate_module_compatibility(self.module_config)
        return self


class SerializedExperimentConfig(ExperimentConfig[L.LightningDataModule]):
    metadata: ExperimentMetadata
    # For now, include distributed config here. Controlling GPU allocation maybe should be its own thing but this is fine for now.
    distributed: DistributedConfig

    module_config: SerializedModuleConfig
    datapack_config: SerializedDatapackConfig
