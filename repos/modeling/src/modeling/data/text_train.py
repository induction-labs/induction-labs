from __future__ import annotations

import functools
from typing import Any, Self

import datasets
import torch
from pydantic import BaseModel, ConfigDict
from synapse.utils.logging import configure_logging
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from modeling.config import DatapackConfig, ExperimentConfig, ModuleConfig
from modeling.config.data import BaseDataSample, BaseDataset
from modeling.modules.text_module import TextLITConfig


@functools.lru_cache(maxsize=1)
def get_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    """
    Lazy load the tokenizer from the Hugging Face model hub.
    This function caches the tokenizer to avoid loading it multiple times.
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


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


class TextDatasetArgs(BaseModel):
    dataset_path: str
    dataset_name: str
    seq_len: int
    tokenizer_name: str
    stride: int | None = None


class TextDataset(BaseDataset[TextPretrainDataSample, TextDatasetArgs]):
    """
    A PyTorch Dataset for text pretraining tasks.
    """

    examples: list[list[int]]

    @classmethod
    def data_cls(cls) -> type[TextPretrainDataSample]:
        """
        Class property that should return the data class type.
        This is used to ensure that the dataset is compatible with the data class.
        """
        return TextPretrainDataSample

    @classmethod
    async def constructor(cls, args: TextDatasetArgs) -> Self:
        """
        Asynchronous constructor for the dataset.
        This allows for any necessary asynchronous setup before the dataset is used.
        """
        logger.info(f"Loading {args.dataset_name} dataset...")
        dataset = datasets.load_dataset(
            args.dataset_path, args.dataset_name, split="train"
        )
        assert isinstance(dataset, datasets.Dataset), (
            f"Expected Dataset, got {type(dataset)}"
        )
        logger.info(f"Dataset loaded with {len(dataset)} examples")

        # Extract text from the dataset
        texts = [example["text"] for example in dataset if example["text"].strip()]
        logger.info(f"Filtered to {len(texts)} non-empty text examples")

        seq_len = args.seq_len
        stride = args.stride or seq_len
        tokenizer = get_tokenizer(args.tokenizer_name)

        # Create temporary instance to access tokenizer property

        # Tokenize and flatten all texts into a single list of token IDs
        all_ids = []
        for txt in texts:
            tokens = tokenizer(txt, return_tensors=None)
            if "input_ids" not in tokens:
                raise ValueError("Tokenizer output must contain 'input_ids'")
            all_ids.extend(tokens["input_ids"])

        # Calculate number of chunks
        examples = []
        for start_idx in range(0, len(all_ids) - seq_len + 1, stride):
            chunk = all_ids[start_idx : start_idx + seq_len]
            examples.append(chunk)

        return cls(
            args=args,
            length=len(examples),
            examples=examples,
        )

    async def get_item(self, idx: int) -> TextPretrainDataSample:
        """
        Asynchronously fetch a data sample.
        This method should be implemented to load the actual data from the paths.
        """
        input_ids = self.examples[idx]
        # For SFT, labels are the same as inputs (language modeling)
        labels = input_ids.copy()
        return TextPretrainDataSample(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
        )


class TextPretrainDatapackConfig(DatapackConfig[TextPretrainDataSample]):
    """
    Configuration class for the Text Pretraining Data Module.
    This class is used to configure the data module for text pretraining tasks.
    """

    config_path: str = "modeling.data.text_train.TextPretrainDatapackConfig"
    dataset_path: str = "wikitext"
    dataset_name: str = "wikitext-2-raw-v1"
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

    async def _init_dataset(self, full_config: ExperimentConfig) -> TextDataset:
        module_config = self.validate_module_compatibility(full_config.module)
        return await TextDataset.constructor(
            TextDatasetArgs(
                dataset_path=self.dataset_path,
                dataset_name=self.dataset_name,
                seq_len=full_config.run.sequence_length,
                tokenizer_name=module_config.get_tokenizer.name_or_path,
            )
        )


__all__ = ["TextPretrainDatapackConfig"]
