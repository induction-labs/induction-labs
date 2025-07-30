from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import batched
from typing import TYPE_CHECKING, Generic, Self, TypeVar, cast, final

from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import torch


T = TypeVar("T", bound="BaseDataSample")


# TODO: Don't use pydantic for this maybe? I'm not sure how much overhead it adds.
class BaseDataSample(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def to_device(self, device: "torch.device", non_blocking: bool = False) -> Self:
        """
        Move the data sample to the specified device.
        This is used to ensure that the data is on the correct device for training or evaluation.
        """

    def __eq__(self, other: object) -> bool:
        import torch

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

    @classmethod
    @abstractmethod
    def combine_batch(cls: type[Self], batch: Sequence[Self]) -> Self:
        """
        Combine a batch of data samples into a single data sample.
        This is used to prepare the data for training or evaluation.
        """


DataSample = TypeVar("DataSample", bound=BaseDataSample, covariant=True)


DatasetArgs = TypeVar("DatasetArgs", bound=BaseModel)


class SampleWithMetadata[SampleType: BaseDataSample](BaseModel):
    """
    Sample with metadata.
    This is used to store additional information about a data sample.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample: SampleType
    indices: list[int]


# We inherit BaseModel to ensure that all of the properties of the dataset stateless.
# The dataset object will be passed to other processes by the dataloader so it should not have any state.
class BaseDataset(
    Dataset[SampleWithMetadata[DataSample]], BaseModel, Generic[DataSample, DatasetArgs]
):
    length: int
    args: DatasetArgs

    # @class_property
    @classmethod
    @abstractmethod
    def data_cls(cls) -> type[DataSample]:
        """
        Class property that should return the data class type.
        """

    @final
    @classmethod
    def collate_fn(cls, batch_size: int, world_size: int):
        """
        Collate function to combine a batch of data samples into a single data sample.
        This is used by the DataLoader to prepare the data for training or evaluation.
        """
        assert batch_size % world_size == 0, (
            f"Total batch size {batch_size} must be divisible by device batch size {world_size}."
        )
        device_batch_size = batch_size // world_size
        data_cls = cast(type[DataSample], cls.data_cls())

        def _collate(
            batch: list[SampleWithMetadata[DataSample]],
        ) -> list[SampleWithMetadata[DataSample]]:
            # Return list of len `world_size` of data sample batched at `device_batch_size`
            assert len(batch) == batch_size, (
                f"Batch size {len(batch)=} does not match expected size {batch_size}."
            )
            per_device_batches = list(batched(batch, device_batch_size))
            per_device_batches = [
                SampleWithMetadata(
                    sample=data_cls.combine_batch(
                        [sample.sample for sample in device_batch]
                    ),
                    indices=[
                        index for sample in device_batch for index in sample.indices
                    ],
                )
                for device_batch in per_device_batches
            ]

            return per_device_batches

        return _collate

    @final
    def __len__(self) -> int:
        return self.length

    @classmethod
    @abstractmethod
    async def constructor(cls, args: DatasetArgs) -> Self:
        """
        Asynchronous constructor for the dataset.
        This allows for any necessary asynchronous setup before the dataset is used.
        """

    @final
    def __getitem__(self, idx: int) -> SampleWithMetadata[DataSample]:
        """
        Get an item from the dataset.
        This method should be overridden to return a specific data sample.
        """
        import asyncio

        sample = asyncio.run(self.get_item(idx))
        return SampleWithMetadata(
            sample=sample,
            indices=[idx],
        )

    @abstractmethod
    async def get_item(self, idx: int) -> DataSample:
        """
        Asynchronous method to get an item from the dataset.
        This allows for any necessary asynchronous operations to retrieve the data sample.
        This returns **1** sample, the batching logic is handled by the base class.
        """
