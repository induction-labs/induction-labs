from __future__ import annotations

import multiprocessing
import tempfile
import weakref
from itertools import chain
from pathlib import Path
from typing import Any

import datasets
import lightning as L
from modeling.config import DatapackConfig, ExperimentConfig, ModuleConfig
from modeling.modules.text_module import TextLITConfig
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _cleanup_temp_dir(temp_dir: tempfile.TemporaryDirectory, path: str):
    """Function to clean up temp directory - called by finalizer."""
    # print(f"Finalizer cleaning up: {path}")
    temp_dir.cleanup()


class TextPretrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: TextPretrainDatapackConfig,
        extra_args: TextPretrainDataModuleExtraArgs,
    ):
        super().__init__()
        self.config = config
        self.extra_args = extra_args
        self._temp_dir = "/tmp/tmpstut1ibm"
        self.temp_path = Path(self._temp_dir)
        self._finalizer = weakref.finalize(
            self, _cleanup_temp_dir, self._temp_dir, str(self.temp_path)
        )

    def setup(self, stage: str | None = None) -> None:
        # This method is used to download the dataset if it is not already present.
        train_data = datasets.load_from_disk(self.temp_path / "train_data")
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
        data = datasets.load_dataset(self.config.dataset_name, trust_remote_code=True)
        column_names = data["train"].column_names  # type: ignore # noqa: PGH003
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return self.extra_args.tokenizer(examples[text_column_name])

        tokenized_datasets = data.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            num_proc=1,  # type: ignore  # noqa: PGH003
            load_from_cache_file=True,  # type: ignore  # noqa: PGH003
            desc="Running tokenizer on dataset",  # type: ignore  # noqa: PGH003
        )
        seq_length = self.extra_args.seq_length

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
        print("Saving train data to disk at", self.temp_path / "train_data")
        train_data.save_to_disk(self.temp_path / "train_data")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,  # type: ignore # noqa: PGH003
            batch_size=self.extra_args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=default_data_collator,
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
    dataset_name: str = "tatsu-lab/alpaca"
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
        self, full_config: ExperimentConfig[TextPretrainDataModule]
    ) -> TextPretrainDataModule:
        module_config = self.validate_module_compatibility(full_config.module)
        extra_args = TextPretrainDataModuleExtraArgs(
            seq_length=full_config.run.sequence_length,
            tokenizer=module_config.get_tokenizer,
            batch_size=full_config.run.batch_size,
        )
        return TextPretrainDataModule(self, extra_args)


__all__ = ["TextPretrainDataModule", "TextPretrainDatapackConfig"]
