from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Self
from torch.utils.data import DataLoader
from pydantic import BaseModel, ConfigDict
import torch


# TODO: Don't use pydantic for this maybe? I'm not sure how much overhead it adds.
class BaseDataSample(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def to_device(self, device: torch.device, non_blocking: bool = False) -> Self:
        """
        Move the data sample to the specified device.
        This is used to ensure that the data is on the correct device for training or evaluation.
        """
        pass


DataSample = TypeVar("DataSample", bound=BaseDataSample, covariant=True)


class BaseDataModule(ABC, Generic[DataSample]):
    def setup(self, stage: str | None = None) -> None:
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader[DataSample]:
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader[DataSample]:
        pass
