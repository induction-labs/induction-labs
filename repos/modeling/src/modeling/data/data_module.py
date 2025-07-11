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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        # Run `torch.equal` on all tensors on all attributes

        for attr in self.__dict__:
            # If it is a tensor, compare it
            if isinstance(getattr(self, attr), torch.Tensor):
                if not torch.equal(getattr(self, attr), getattr(other, attr)):
                    return False
            else:
                self_value = getattr(self, attr)
                other_value = getattr(other, attr)
                # Then must be BaseDataSample
                assert isinstance(self_value, BaseDataSample) and isinstance(
                    other_value, BaseDataSample
                ), (
                    f"Attributes {attr} are not BaseDataSample instances: {self_value}, {other_value}"
                )
                if self_value != other_value:
                    return False
        return True


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
