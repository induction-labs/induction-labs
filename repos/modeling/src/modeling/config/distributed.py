from __future__ import annotations

from enum import Enum
from functools import reduce

from pydantic import BaseModel, Field, model_validator
from typing import TYPE_CHECKING, Self


if TYPE_CHECKING:
    import torch


class MeshAxis(str, Enum):
    """
    Enum for mesh axes used in distributed training.
    """

    DP = "data_parallel"
    FSDP = "fully_sharded_data_parallel"
    TP = "tensor_parallel"

    def __str__(self) -> str:
        return self.value


class InstanceConfig(BaseModel):
    # TODO: Serialize + deserialize torch.device
    node_rank: int  # [0, num_nodes)
    device_rank: int  # [0, 8)

    @property
    def is_main(self) -> bool:
        """
        Check if this instance is the main instance in the distributed setup.
        The main instance is typically the one with node_rank 0 and device_rank 0.
        """
        return self.node_rank == 0 and self.device_rank == 0

    @property
    def device(self) -> "torch.device":
        """
        Get the device for this instance based on its device rank.
        This assumes that the devices are ordered by their ranks.
        """
        import torch
        # NOTE that all processes only see one device, so we just use torch.device(0)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)


class DistributedConfig(BaseModel):
    devices_per_node: int = 1
    num_nodes: int = 1

    class ShardingConfig(BaseModel):
        # TODO: For now DP and TP are disabled, figure out how to implement them properly
        DP: int = Field(default=1, ge=1, le=1, description="Data parallelism factor")
        FSDP: int = Field(
            default=1, ge=1, description="Fully sharded data parallelism factor"
        )
        TP: int = Field(default=1, ge=1, le=1, description="Tensor parallelism factor")

        @classmethod
        def default(cls, world_size: int) -> DistributedConfig.ShardingConfig:
            """
            Default not sharding configuration for distributed training.
            """
            return cls(
                DP=1,  # Data parallelism across all devices
                FSDP=world_size,  # Only FSDP by default
                TP=1,  # Tensor parallelism is not used by default
            )

    sharding: DistributedConfig.ShardingConfig = Field(
        default_factory=lambda data: DistributedConfig.ShardingConfig.default(
            data["devices_per_node"] * data["num_nodes"]
        )
    )

    @model_validator(mode="after")
    def validate_sharding(self) -> Self:
        """
        Validate the checkpoint_path format.
        Ensures it starts with 'gs://' and contains a valid bucket name.
        """
        sharding_keys = set(self.sharding.model_dump().keys())
        mesh_axis_keys = set(MeshAxis.__members__.keys())
        assert sharding_keys == mesh_axis_keys, (
            f"Sharding keys {sharding_keys} do not match MeshAxis keys {mesh_axis_keys}"
        )
        sharding_values = self.sharding.model_dump().values()
        sharding_product = reduce(lambda x, y: x * y, sharding_values, 1)
        # Check sharding values multiply to world size
        assert self.world_size == sharding_product, (
            f"{self.world_size=} != {self.sharding=}"
        )
        return self

    @property
    def world_size(self) -> int:
        """
        Calculate the total number of processes across all nodes.
        """
        return self.devices_per_node * self.num_nodes

    def global_rank(self, instance: InstanceConfig) -> int:
        """
        Calculate the global rank of the instance based on its node_rank and device_rank.
        This is used to identify the unique rank of the instance in the distributed setup.
        """
        return instance.node_rank * self.devices_per_node + instance.device_rank

    @classmethod
    def mock_data(cls) -> DistributedConfig:
        """
        Create a mock instance of DistributedConfig for testing purposes.
        """
        return cls()
