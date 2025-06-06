from __future__ import annotations

from pydantic import BaseModel


class DistributedConfig(BaseModel):
    devices_per_node: int = 1
    num_nodes: int = 1

    @classmethod
    def mock_data(cls) -> DistributedConfig:
        """
        Create a mock instance of DistributedConfig for testing purposes.
        """
        return cls()
