from __future__ import annotations

import multiprocessing
from itertools import chain
from pathlib import Path

import datasets
import lightning as L
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class LoadDataArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dataset_name: str = "tatsu-lab/alpaca"
    seq_length: int = 1024  # Default sequence length
    tokenizer: PreTrainedTokenizerBase
    batch_size: int

    def hash(self) -> str:
        """
        Generate a hash of the dataset name and sequence length for caching purposes.
        """
        # TODO: Implement dataset hash
        raise NotImplementedError(
            "Hashing is not implemented yet. This method should return a unique identifier for the dataset and sequence length."
        )


class TextPretrainDataModule(L.LightningDataModule):
    def __init__(self, args: LoadDataArgs, tmpdir: Path):
        super().__init__()
        self.args = args
        self.tmpdir = tmpdir

    def setup(self, stage: str | None = None):
        # This method is used to download the dataset if it is not already present.
        train_data = datasets.load_from_disk(self.tmpdir / "train_data")
        assert isinstance(train_data, datasets.Dataset), (
            "Expected train_data to be a Dataset"
        )
        self.train_data = train_data

    def prepare_data(self):
        """
        Function created using code found in
        https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py
        """
        # This method is used to set up the dataset for training.
        data = datasets.load_dataset(self.args.dataset_name, trust_remote_code=True)
        column_names = data["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return self.args.tokenizer(examples[text_column_name])

        tokenized_datasets = data.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            num_proc=2,  # type: ignore  # noqa: PGH003
            load_from_cache_file=True,  # type: ignore  # noqa: PGH003
            desc="Running tokenizer on dataset",  # type: ignore  # noqa: PGH003
        )
        seq_length = self.args.seq_length

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
            total_length = len(concatenated_examples[next(iter(examples.keys()))])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            if total_length > seq_length:
                total_length = (total_length // seq_length) * seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {seq_length}",
        )
        train_data = lm_datasets["train"]
        assert isinstance(train_data, datasets.Dataset)
        train_data.save_to_disk(self.tmpdir / "train_data")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=default_data_collator,
        )


__all__ = [
    "LoadDataArgs",
    "TextPretrainDataModule",
]
