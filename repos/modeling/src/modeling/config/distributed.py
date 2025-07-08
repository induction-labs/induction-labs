from __future__ import annotations

from enum import Enum
from functools import reduce

from pydantic import BaseModel, Field, model_validator
from typing import TYPE_CHECKING
from typing import Self


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
    # model_config = ConfigDict(arbitrary_types_allowed=True)
    # device: torch.device
    node_rank: int  # [0, num_nodes)
    device_rank: int  # [0, 8)

    @property
    def device(self) -> "torch.device":
        """
        Get the device for this instance based on its device rank.
        This assumes that the devices are ordered by their ranks.
        """
        import torch

        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu", self.device_rank
        )


class ShardingConfig(BaseModel):
    DP: int
    FSDP: int
    TP: int

    @classmethod
    def default(cls, world_size: int) -> ShardingConfig:
        """
        Default not sharding configuration for distributed training.
        """
        return cls(
            DP=world_size,  # Data parallelism across all devices
            FSDP=1,  # Fully sharded data parallelism is not used by default
            TP=1,  # Tensor parallelism is not used by default
        )


class DistributedConfig(BaseModel):
    devices_per_node: int = 1
    num_nodes: int = 1
    sharding: ShardingConfig = Field(
        default_factory=lambda data: ShardingConfig.default(
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
