from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Generic, Self, TypeVar

import lightning as L
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from modeling.utils.git import get_git_commit_sha, get_git_commit_sha_short
from modeling.types import Accelerator, DType, AttentionImplementation

from .distributed import DistributedConfig
from .wandb import WandbConfig
from typing import Optional
from torch.distributed.fsdp import MixedPrecisionPolicy

from lightning.fabric.plugins.precision.precision import (
    _PRECISION_INPUT,
)

_PRECISION_INPUT
# _T = TypeVar("_T")
# _T_co = TypeVar("_T_co", covariaint=True)

# _LitModule = TypeVar("_LitModule", bound="L.LightningModule", covariant=True)
_LITDataModule = TypeVar("_LITDataModule", bound="L.LightningDataModule")


class ExperimentMetadata(BaseModel):
    wandb: Optional[WandbConfig]
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
    def create_module(self, run_config: RunConfig) -> L.LightningModule:
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

    def create_module(self, run_config: RunConfig) -> L.LightningModule:
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

    distributed: DistributedConfig

    num_epochs: int = 1
    steps_per_epoch: int  # Number of steps per epoch
    # save_interval: int = 1000  # How often to save checkpoints
    seed: int = 42  # Random seed for reproducibility
    sequence_length: int  # Default sequence length
    batch_size: int  # Default batch size

    lr: float

    attn_impl: AttentionImplementation = AttentionImplementation.SDPA
    accelerator: Accelerator = Accelerator.CUDA
    precision: DType = DType.bf16
    quantize_model: bool = True  # Quantize the model if True. If False, only cast the optimizer weights to precision

    @computed_field
    @property
    def lightning_precision(self) -> _PRECISION_INPUT:
        # if we're not quantizing the model, we use the suffix "-true" for the precision
        # since we're only using the precision for the optimizer and not for the model weights.
        lighting_mixed_suffix = "-true" if self.quantize_model else "-mixed"

        if self.precision == DType.bf16:
            return "bf16" + lighting_mixed_suffix
        elif self.precision == DType.fp16:
            return "16" + lighting_mixed_suffix
        elif self.precision == DType.fp32:
            # quantization doesn't matter
            return "32"
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")

    @property
    def mp_policy(self) -> MixedPrecisionPolicy:
        return MixedPrecisionPolicy(
            param_dtype=self.precision.torch_dtype,
            reduce_dtype=DType.fp32.torch_dtype
            if self.quantize_model
            else self.precision.torch_dtype,
            output_dtype=self.precision.torch_dtype,  # output dtype is always the same as param dtype
            cast_forward_inputs=True,  # cast inputs to BF16 before each module
        )

    @computed_field
    @property
    def process_batch_size(self) -> int:
        """
        Calculate the effective batch size per process.
        Torch is troll because batch size is handled differently depending on the distributed strategy.
        https://pytorch-lightning.readthedocs.io/en/1.5.10/advanced/multi_gpu.html#batch-size
        """
        assert (self.batch_size % (self.distributed.world_size)) == 0, (
            f"Batch size {self.batch_size=} must be divisible by "
            f"{self.distributed.world_size=} "
        )

        return self.batch_size // (self.distributed.world_size)

    @model_validator(mode="after")
    def check_batch_size(self) -> Self:
        _ = self.process_batch_size  # Trigger the property to validate batch size
        return self

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
            lr=1e-3,
            distributed=DistributedConfig.mock_data(),
            attn_impl=AttentionImplementation.SDPA,
            accelerator=Accelerator.CUDA,
            precision=DType.bf16,
            quantize_model=True,
        )


class ExperimentConfig(BaseModel, Generic[_LITDataModule]):
    metadata: ExperimentMetadata
    # For now, include distributed config here.

    module: ModuleConfig
    datapack: DatapackConfig[_LITDataModule]

    # These maybe should be moved to module_config, but seem standard enough to keep here
    run: RunConfig

    @model_validator(mode="after")
    def check_compatibility(self) -> Self:
        """
        Validate that the module and datapack configurations are compatible.
        This method is called after the model is initialized to ensure compatibility.
        """
        self.module.validate_datapack_compatibility(self.datapack)
        self.datapack.validate_module_compatibility(self.module)
        return self


class SerializedExperimentConfig(ExperimentConfig[L.LightningDataModule]):
    module: SerializedModuleConfig
    datapack: SerializedDatapackConfig
