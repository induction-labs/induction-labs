from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class DistributedStrategy(str, Enum):
    """
    Enum for distributed training strategies.
    We only support ddp for now
    https://pytorch-lightning.readthiedocs.io/en/1.5.10/advanced/multi_gpu.html#distributed-modes
    """

    ddp = "ddp_find_unused_parameters_true"


class DistributedConfig(BaseModel):
    devices_per_node: int = 1
    num_nodes: int = 1
    strategy: DistributedStrategy = DistributedStrategy.ddp

    @property
    def world_size(self) -> int:
        """
        Calculate the total number of processes across all nodes.
        """
        return self.devices_per_node * self.num_nodes

    @classmethod
    def mock_data(cls) -> DistributedConfig:
        """
        Create a mock instance of DistributedConfig for testing purposes.
        """
        return cls()
