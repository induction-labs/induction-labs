from __future__ import annotations

from typing import Any

import datasets
from modeling.config import DatapackConfig, ExperimentConfig, ModuleConfig
from modeling.modules.text_module import TextLITConfig
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import torch
from torch.utils.data import Dataset
from typing import List, Callable, Optional
from synapse.utils.logging import configure_logging
from modeling.config.data import BaseDataSample

logger = configure_logging(__name__)


class TextPretrainDataSample(BaseDataSample):
    """
    A data sample for text pretraining tasks.
    This class extends BaseDataSample to include input IDs and labels.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: torch.Tensor
    labels: torch.Tensor

    def to_device(
        self, device: torch.device, non_blocking: bool = False
    ) -> TextPretrainDataSample:
        """
        Move the data sample to the specified device.
        This is used to ensure that the data is on the correct device for training or evaluation.
        """
        return TextPretrainDataSample(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
        )

    @classmethod
    def combine_batch(  # type: ignore[override]
        cls, batch: list[TextPretrainDataSample]
    ) -> TextPretrainDataSample:
        """
        Combine a batch of TextPretrainDataSamples into a single TextPretrainDataSample.
        This is used to prepare the data for training or evaluation.
        """
        input_ids = torch.stack([sample.input_ids for sample in batch])
        labels = torch.stack([sample.labels for sample in batch])
        return TextPretrainDataSample(input_ids=input_ids, labels=labels)


class GenericSFTDataset(Dataset[TextPretrainDataSample]):
    """
    A generic PyTorch Dataset for sequence fine-tuning (SFT) tasks.

    Args:
        texts (List[str]): A list of raw text strings to tokenize and chunk.
        tokenizer (Callable[[str], dict]): A tokenizer function that takes a string
            and returns a dict with at least the key 'input_ids'.
        seq_len (int): The sequence length for each example.
        stride (Optional[int]): The stride for sliding window, defaults to seq_len (no overlap).
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Callable[[str], dict],
        seq_len: int,
        stride: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len

        # Tokenize and flatten all texts into a single list of token IDs
        all_ids = []
        for txt in texts:
            tokens = tokenizer(txt)
            if "input_ids" not in tokens:
                raise ValueError("Tokenizer output must contain 'input_ids'")
            all_ids.extend(tokens["input_ids"])

        # Calculate number of chunks
        self.examples = []
        for start_idx in range(0, len(all_ids) - seq_len + 1, self.stride):
            chunk = all_ids[start_idx : start_idx + seq_len]
            self.examples.append(chunk)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TextPretrainDataSample:
        input_ids = self.examples[idx]
        # For SFT, labels are the same as inputs (language modeling)
        labels = input_ids.copy()
        return TextPretrainDataSample(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
        )

    @classmethod
    def combine_batch(
        cls, batch: list[TextPretrainDataSample]
    ) -> TextPretrainDataSample:
        """
        Combine a batch of TextPretrainDataSamples into a single TextPretrainDataSample.
        This is used to prepare the data for training or evaluation.
        """
        input_ids = torch.stack([sample.input_ids for sample in batch])
        labels = torch.stack([sample.labels for sample in batch])
        return TextPretrainDataSample(input_ids=input_ids, labels=labels)


class TextPretrainDataModule(BaseDataModule[TextPretrainDataSample]):
    def __init__(
        self,
        config: TextPretrainDatapackConfig,
        extra_args: TextPretrainDataModuleExtraArgs,
    ):
        super().__init__()
        self.config = config
        self.extra_args = extra_args
        # ...

    def setup(self, stage: str | None = None) -> None:
        logger.info("Loading wikitext-2 dataset...")
        dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        logger.info(f"Dataset loaded with {len(dataset)} examples")

        # Extract text from the dataset
        texts = [example["text"] for example in dataset if example["text"].strip()]
        logger.info(f"Filtered to {len(texts)} non-empty text examples")

        # Create the training dataset using GenericSFTDataset
        logger.info("Creating training dataset...")
        self.train_data = GenericSFTDataset(
            texts=texts,
            tokenizer=lambda x: self.extra_args.tokenizer(x, return_tensors=None),
            seq_len=self.extra_args.seq_length,
        )
        logger.info(f"Training dataset created with {len(self.train_data)} examples")

    def train_dataloader(self, generator: torch.Generator) -> DataLoader:
        return DataLoader(
            self.train_data,  # type: ignore # noqa: PGH003
            batch_size=self.extra_args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=GenericSFTDataset.combine_batch,  # type: ignore
            generator=generator,
        )

    def val_dataloader(self, generator: torch.Generator) -> DataLoader:
        # For pretraining, we typically don't have a validation set
        return DataLoader(
            self.train_data,  # Using the same data for validation
            batch_size=self.extra_args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=GenericSFTDataset.combine_batch,  # type: ignore
            generator=generator,
        )


class TextPretrainDataModuleExtraArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    seq_length: int = 1024  # Default sequence length
    tokenizer: PreTrainedTokenizerBase
    batch_size: int


class TextPretrainDatapackConfig(DatapackConfig[TextPretrainDataModule]):
    """
    Configuration class for the Text Pretraining Data Module.
    This class is used to configure the data module for text pretraining tasks.
    """

    config_path: str = "modeling.data.text_train.TextPretrainDatapackConfig"
    dataset_name: str = "wikitext"
    num_workers: int = 2

    def validate_module_compatibility(
        self, module_config: ModuleConfig[Any]
    ) -> TextLITConfig[Any]:
        """
        Validate that the Lightning module is compatible with the data module.
        This method should be implemented by subclasses to perform any necessary checks.
        """
        assert isinstance(module_config, TextLITConfig), (
            "TextPretrainDatapackConfig can only be used with TextLITConfig."
        )
        return module_config

    def create_datapack(
        self,
        full_config: ExperimentConfig[TextPretrainDataModule],
    ) -> TextPretrainDataModule:
        module_config = self.validate_module_compatibility(full_config.module)
        extra_args = TextPretrainDataModuleExtraArgs(
            seq_length=full_config.run.sequence_length,
            tokenizer=module_config.get_tokenizer,
            batch_size=full_config.run.batch_size,
        )
        return TextPretrainDataModule(self, extra_args)


__all__ = ["TextPretrainDataModule", "TextPretrainDatapackConfig"]
