from __future__ import annotations

from pydantic import BaseModel


class DistributedConfig(BaseModel):
    @classmethod
    def mock_data(cls) -> DistributedConfig:
        """
        Create a mock instance of DistributedConfig for testing purposes.
        """
        return cls()
