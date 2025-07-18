from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Optional,
    Self,
    TypeVar,
    Never,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    model_validator,
)
from synapse.utils.logging import configure_logging

from modeling.types import Accelerator, AttentionImplementation, DType
from modeling.utils.cloud_path import BeforeValidator, CloudPath, path_validator
from modeling.utils.git import get_git_commit_sha, get_git_commit_sha_short

from .distributed import DistributedConfig, InstanceConfig
from .wandb import WandbConfig
from modeling.config.data import BaseDataSample, BaseDataset

if TYPE_CHECKING:
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from modeling.modules.base_module import BaseLITModule


logger = configure_logging(__name__, logging.DEBUG)
DataSample = TypeVar("DataSample", bound="BaseDataSample")


class RuntimeConfig(BaseModel):
    id: str
    start_time: datetime
    tmp_dir: Path


# TODO: Make checkpoint config use different backends and have it dynamically loaded and stuff
# for now just only use gcs checkpoints
class GCSCheckpointConfig(BaseModel):
    """
    Configuration for GCS checkpointing.
    This class is used to configure the GCS checkpointing settings for an experiment.
    """

    checkpoint_frequency: int = Field(
        0, description="How often to save checkpoints in steps. Set to 0 to disable."
    )
    checkpoint_last_step: bool = Field(
        True,
        description="Whether to save the last step checkpoint. If True, the last step will always be saved regardless of checkpoint_frequency.",
    )
    checkpoint_first_step: bool = Field(
        False, description="Whether to save the first step checkpoint."
    )

    loaded_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the checkpoint was loaded. Used to determine if the checkpoint is fresh.",
    )
    checkpoint_prefix: CloudPath = Field(
        ...,
        description="Path to the GCS bucket where checkpoints will be stored. Should be gs://induction-labs/...",
    )

    @computed_field
    @property
    def checkpoint_path(self) -> CloudPath:
        if self.loaded_at is None:
            # If loaded_at is not set, return the prefix only
            return self.checkpoint_prefix / "{date}"
        return self.checkpoint_prefix / f"{self.loaded_at:%Y-%m-%dT%H-%M-%S}"

    @property
    def bucket_and_path(self) -> tuple[str, Path]:
        """
        Extract the bucket name and path from the checkpoint_path.
        Returns a tuple of (bucket_name, path_in_bucket).
        """

        bucket_name, *prefix_parts = self.checkpoint_path.path.parts

        return bucket_name, Path(*prefix_parts)

    @property
    def ckpt_config_path(self) -> CloudPath:
        return self.checkpoint_path / "config.toml"

    @model_validator(mode="after")
    def validate_checkpoint_path(self) -> Self:
        """
        Validate the checkpoint_path format.
        Ensures it starts with 'gs://' and contains a valid bucket name.
        """
        assert self.checkpoint_path.cloud == CloudPath.Cloud.GS, (
            f"Only GCS checkpoints are supported. {self.checkpoint_path.cloud=}"
        )
        return self

    def should_checkpoint(self, step: int) -> bool:
        """
        Determine if a checkpoint should be saved at the given step.
        Step should be strictly positive
        """
        # This is checked against with the `checkpoint_first_step` and `checkpoint_last_step` flags.
        if step == 0:
            return False
        if self.checkpoint_frequency == 0:
            return False
        return step % self.checkpoint_frequency == 0

    @classmethod
    def mock_data(cls) -> GCSCheckpointConfig:
        """
        Create a mock instance of GCSCheckpointConfig for testing purposes.
        """
        return cls(
            checkpoint_prefix=CloudPath.from_str("gs://induction-labs/checkpoints"),
            checkpoint_frequency=0,
            checkpoint_last_step=False,
            checkpoint_first_step=False,
        )


class ExperimentMetadata(BaseModel):
    wandb: Optional[WandbConfig]

    # TODO: Write a dedicated class for serializing paths to strings
    @field_serializer("output_dir")
    def serialize_output_dir(self, output_dir: Path, _info):
        return output_dir.as_posix()

    output_dir: Annotated[Path, BeforeValidator(path_validator)]
    checkpoint: Optional[GCSCheckpointConfig]

    @classmethod
    def mock_data(cls) -> ExperimentMetadata:
        """
        Create a mock instance of ExperimentMetadata for testing purposes.
        """
        return cls(
            wandb=WandbConfig.mock_data(),
            output_dir=Path("/tmp/experiment_output"),
            checkpoint=GCSCheckpointConfig.mock_data(),
        )

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def reset_loaded_at(self) -> None:
        """
        Reset the loaded_at timestamp to the current time.
        This is called when the config is loaded from toml.
        """
        logger.debug("Resetting loaded_at timestamp to current time.")
        self.loaded_at = datetime.now(UTC)
        if self.checkpoint is not None:
            assert self.checkpoint.loaded_at is None, (
                "Checkpoint loaded_at should be None when resetting the loaded_at timestamp."
            )
            self.checkpoint.loaded_at = self.loaded_at

    commit: str = Field(default_factory=get_git_commit_sha)
    commit_short: str = Field(default_factory=get_git_commit_sha_short)


class ModuleConfig(BaseModel, ABC):
    config_path: str

    @classmethod
    def module_cls(cls) -> type["BaseLITModule"]:
        """
        Return the class of the Lightning module.
        This method should be implemented by subclasses to return the actual module class.
        """
        return BaseLITModule

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
    def create_module(
        self,
        run_config: RunConfig,
        instance_config: InstanceConfig,
    ) -> "BaseLITModule":
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

    def check_config_path(self) -> Self:  # type: ignore[override]
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

    def create_module(
        self,
        run_config: RunConfig,
        instance_config: InstanceConfig,
    ) -> "BaseLITModule":
        """
        Create a Lightning module instance by loading it from the specified path.
        """
        raise NotImplementedError(
            "SerializedModuleConfig should never be used to start an experiment directly."
        )


class DatapackConfig(ABC, BaseModel, Generic[DataSample]):
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
    async def _init_train_dataset(
        self, full_config: ExperimentConfig[DataSample]
    ) -> BaseDataset[DataSample, Any]:
        """
        Create a Lightning data module instance.
        This method should be implemented by subclasses to return an instance of the Lightning data module.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def _init_val_dataset(
        self, full_config: ExperimentConfig[DataSample]
    ) -> BaseDataset[DataSample, Any]:
        """
        Create a Lightning data module instance.
        This method should be implemented by subclasses to return an instance of the Lightning data module.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # @final
    # async def train_dataloader(
    #     self, full_config: ExperimentConfig[DataSample], generator: torch.Generator
    # ) -> DataLoader[list[DataSample]]:
    #     dataset = await self._init_train_dataset(full_config)
    #     return self._make_dataloader(dataset, full_config, generator)

    # @final
    # async def val_dataloader(
    #     self, full_config: ExperimentConfig[DataSample], generator: torch.Generator
    # ) -> DataLoader[list[DataSample]]:
    #     dataset = await self._init_val_dataset(full_config)
    #     return self._make_dataloader(dataset, full_config, generator)


class SerializedDatapackConfig(DatapackConfig[DataSample]):
    """
    Configuration for a serialized Lightning data module.
    This class is used to load a Lightning data module from a specified path.
    """

    # Explicity allow extra config to go through, because this will be used to initialize the module
    model_config = ConfigDict(extra="allow")

    def check_config_path(self) -> Self:  # type: ignore[override]
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
        self,
        full_config: ExperimentConfig[DataSample],
    ) -> DataSample:
        """
        Create a Lightning data module instance by loading it from the specified path.
        """
        raise NotImplementedError(
            "SerializedDatapackConfig should never be used to start an experiment directly."
        )

    def _init_train_dataset(  # type: ignore[override]
        self, full_config: ExperimentConfig[DataSample]
    ) -> BaseDataset[DataSample, Never]:
        """
        Create a Lightning data module instance for the training dataset.
        This method should be implemented by subclasses to return an instance of the Lightning data module.
        """
        raise NotImplementedError(
            "SerializedDatapackConfig should never be used directly."
        )

    def _init_val_dataset(  # type: ignore[override]
        self, full_config: ExperimentConfig[DataSample]
    ) -> BaseDataset[DataSample, Never]:
        """
        Create a Lightning data module instance for the validation dataset.
        This method should be implemented by subclasses to return an instance of the Lightning data module.
        """
        raise NotImplementedError(
            "SerializedDatapackConfig should never be used directly."
        )


class LinearLRSchedule(BaseModel):
    peak_lr: float = Field(
        description="Peak learning rate for the learning rate schedule."
    )
    end_lr: float = Field(
        description="Final learning rate for the learning rate schedule."
    )
    warmup_steps: int = Field(
        description="Number of warmup steps for the learning rate schedule."
    )
    end_step: int = Field(
        description="Total number of steps for the learning rate schedule."
    )

    @classmethod
    def constant_lr(cls, lr: float) -> LinearLRSchedule:
        """
        Create a constant learning rate schedule.
        This is useful for testing or when a constant learning rate is desired.
        """
        return cls(
            peak_lr=lr,
            end_lr=lr,
            warmup_steps=0,
            end_step=1,  # Only one step for constant LR
        )


class ProfileConfig(BaseModel):
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: int = 0

    @property
    def total_steps(self) -> int:
        """
        Calculate the total number of steps for profiling.
        This is the sum of wait, warmup, and active steps, multiplied by the repeat count.
        """
        return (self.wait + self.warmup + self.active) * (self.repeat + 1)


class RunConfig(BaseModel):
    """
    Configuration for a run.
    This class is used to configure the run settings for an experiment.
    """

    distributed: DistributedConfig
    profile: Optional[ProfileConfig] = None

    num_steps: int  # Number of steps per epoch
    # save_interval: int = 1000  # How often to save checkpoints
    seed: int = 42  # Random seed for reproducibility
    sequence_length: int  # Default sequence length
    batch_size: int  # Default batch size

    validation_every_n_steps: int = -1
    validation_steps: int = 1

    lr: LinearLRSchedule

    attn_impl: AttentionImplementation = AttentionImplementation.SDPA
    accelerator: Accelerator = Accelerator.CUDA
    precision: DType = DType.bf16
    quantize_model: bool = True  # Quantize the model if True. If False, only cast the optimizer weights to precision

    @property
    def mp_policy(self) -> "MixedPrecisionPolicy":
        from torch.distributed.fsdp import MixedPrecisionPolicy

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
        # assert (self.batch_size % (self.distributed.world_size)) == 0, (
        #     f"Batch size {self.batch_size=} must be divisible by "
        #     f"{self.distributed.world_size=} "
        # )

        return self.batch_size  # // (self.distributed.world_size)

    @model_validator(mode="after")
    def check_batch_size(self) -> Self:
        # _ = self.process_batch_size  # Trigger the property to validate batch size
        return self

    @model_validator(mode="after")
    def check_profiler_num_steps(self) -> Self:
        if self.profile:
            assert self.num_steps >= self.profile.total_steps, (
                f"{self.num_steps=} must be greater than or equal to {self.profile.total_steps=}"
            )
        return self

    def cpu_config(self) -> RunConfig:
        """
        Create a CPU-specific configuration for the run.
        This is useful for testing or running on CPU-only environments.
        """
        return self.model_copy(
            update={
                "accelerator": Accelerator.CPU,
                "precision": DType.fp32,  # Use full precision on CPU
                "attn_impl": AttentionImplementation.SDPA,  # Use SDPA for CPU
                "quantize_model": False,  # No quantization on CPU
                "distributed": DistributedConfig(
                    devices_per_node=1,
                    num_nodes=1,  # Single node for CPU runs
                ),
            }
        )

    @classmethod
    def mock_data(cls) -> RunConfig:
        """
        Create a mock instance of RunConfig for testing purposes.
        """
        return cls(
            num_steps=100,
            seed=42,
            sequence_length=1024,
            batch_size=4,
            lr=LinearLRSchedule.constant_lr(1e-3),
            distributed=DistributedConfig.mock_data(),
            attn_impl=AttentionImplementation.SDPA,
            accelerator=Accelerator.CUDA,
            precision=DType.bf16,
            quantize_model=True,
        )


class ExperimentConfig(BaseModel):
    metadata: ExperimentMetadata
    # For now, include distributed config here.

    module: ModuleConfig
    datapack: DatapackConfig

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

    def serialize_to_toml(self) -> str:
        import tomli_w

        return tomli_w.dumps(self.model_dump(serialize_as_any=True))

    def testing_config(
        self,
        num_steps: int = 1,
        enable_wandb: bool = True,
        profile: bool = False,
        with_val: bool = True,
    ) -> Self:
        """
        Create a testing configuration for the experiment.
        This is useful for unit tests to avoid running the full experiment.
        """
        profile_config: Optional[ProfileConfig] = (
            ProfileConfig() if profile else self.run.profile
        )
        validation_every_n_steps = self.run.validation_every_n_steps if with_val else -1

        updated_wandb_config = (
            self.metadata.wandb.model_copy(
                update={
                    "project": "testing",
                    "name": self.metadata.wandb.name,
                }
            )
            if self.metadata.wandb is not None and enable_wandb
            else None
        )
        return self.model_copy(
            update={
                "metadata": self.metadata.model_copy(
                    update={
                        "wandb": updated_wandb_config,
                        "output_dir": Path("/tmp") / self.metadata.output_dir.name,
                        "checkpoint": None,
                    }
                ),
                "run": self.run.model_copy(
                    update={
                        "num_epochs": 1,
                        "num_steps": num_steps,
                        "profile": profile_config,
                        "validation_every_n_steps": validation_every_n_steps,
                    }
                ),
            }
        )


class UnifiedExperimentConfig(ExperimentConfig):
    runtime_config: RuntimeConfig


class SerializedExperimentConfig(ExperimentConfig):
    module: SerializedModuleConfig  # type: ignore[override]
    datapack: SerializedDatapackConfig  # type: ignore[override]
