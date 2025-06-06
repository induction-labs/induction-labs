from __future__ import annotations

from pydantic import BaseModel


class WandbConfig(BaseModel):
    project: str
    name: str

    @classmethod
    def mock_data(cls) -> WandbConfig:
        """i
        Create a mock instance of WandbConfig for testing purposes.
        """
        return cls(project="test-project", name="test-experiment")
