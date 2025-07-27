from __future__ import annotations

import asyncio
import functools
import json
import re
from typing import Any, Self

import fsspec
import pandas as pd
import torch
from google.cloud import storage
from pydantic import BaseModel, ConfigDict
from qwen_vl_utils import process_vision_info
from synapse.utils.logging import configure_logging
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLProcessor,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from modeling.config import DatapackConfig, ExperimentConfig, ModuleConfig
from modeling.config.data import BaseDataSample, BaseDataset

# from modeling.modules.vl_sft.qwen_25vl import VlSftLITConfig
from modeling.environments.agents.uitars15 import COMPUTER_USE_15, THOUGHT_LONG


@functools.lru_cache(maxsize=1)
def get_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    """
    Lazy load the tokenizer from the Hugging Face model hub.
    This function caches the tokenizer to avoid loading it multiple times.
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


logger = configure_logging(__name__)


class VlDataSample(BaseDataSample):
    """
    A data sample for VL pretraining tasks.
    This class extends BaseDataSample to include input IDs and labels.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor

    def to_device(
        self, device: torch.device, non_blocking: bool = False
    ) -> VlDataSample:
        """
        Move the data sample to the specified device.
        This is used to ensure that the data is on the correct device for training or evaluation.
        """
        return VlDataSample(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(device, non_blocking=non_blocking),
            pixel_values=self.pixel_values.to(device, non_blocking=non_blocking),
            image_grid_thw=self.image_grid_thw.to(device, non_blocking=non_blocking),
        )

    @classmethod
    def combine_batch(  # type: ignore[override]
        cls, batch: list[VlDataSample]
    ) -> VlDataSample:
        """
        Combine a batch of VlDataSamples into a single VlDataSample.
        This is used to prepare the data for training or evaluation.
        """
        input_ids = torch.stack([sample.input_ids for sample in batch])
        labels = torch.stack([sample.labels for sample in batch])
        attention_mask = torch.stack([sample.attention_mask for sample in batch])

        # TODO: check this is correct
        pixel_values = torch.concat([sample.pixel_values for sample in batch], dim=0)
        image_grid_thw = torch.concat(
            [sample.image_grid_thw for sample in batch], dim=0
        )

        return VlDataSample(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )


class VlDatasetArgs(BaseModel):
    dataset_path: str
    seq_len: int
    tokenizer_name: str


async def load_turns_async(path):
    fs, _, paths = fsspec.get_fs_token_paths(path)

    def _read():
        with fs.open(paths[0], "r") as f:
            records = json.load(f)
        return [{"image": r["image"], "text": r["text"]} for r in records[:-1]]

    return await asyncio.to_thread(_read)


_GS_RE = re.compile(r"^gs://([^/]+)/(.+)$")


def load_turns_gcs(gs_uri: str):
    """
    Reads a JSON array from Google Cloud Storage and returns
    [{"image": ..., "text": ...}, ...] minus the last row.
    """
    m = _GS_RE.match(gs_uri)
    if not m:
        raise ValueError(f"Not a valid gs:// URI: {gs_uri}")
    bucket_name, blob_name = m.groups()

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # download the whole object as one string
    data_str = blob.download_as_text()

    records = json.loads(data_str)  # list-of-dicts
    return [
        {"image": r["image"], "text": r["text"]} for r in records[:-1]
    ]  # mimic .iloc[0:-1]


class VlDataset(BaseDataset[VlDataSample, VlDatasetArgs]):
    """
    A PyTorch Dataset for VL model tasks.
    """

    # has rows "attempt_id", "instruction"
    examples: pd.DataFrame
    dataset_folder: str
    processor: Qwen2_5_VLProcessor
    seq_len: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def data_cls(cls) -> type[VlDataSample]:
        """
        Class property that should return the data class type.
        This is used to ensure that the dataset is compatible with the data class.
        """
        return VlDataSample

    @classmethod
    async def constructor(cls, args: VlDatasetArgs) -> Self:
        """
        Asynchronous constructor for the dataset.
        This allows for any necessary asynchronous setup before the dataset is used.
        """

        examples = pd.read_json(args.dataset_path, lines=True)
        dataset_folder = args.dataset_path.rstrip("/").rsplit("/", 1)[0]
        processor = Qwen2_5_VLProcessor.from_pretrained(args.tokenizer_name)
        return cls(
            args=args,
            examples=examples,
            dataset_folder=dataset_folder,
            length=len(examples),
            processor=processor,
            seq_len=args.seq_len,
        )

    async def get_item(self, idx: int) -> VlDataSample:
        """
        Asynchronously fetch a data sample.
        This method should be implemented to load the actual data from the paths.
        """
        sample = self.examples.iloc[idx]
        instruction = sample["instruction"]
        data_path = f"{self.dataset_folder}/metadata/{sample['attempt_id']}.json"
        # !!!XXX: you must do .to_thread bc otherwise ray will hang
        turns = await asyncio.to_thread(load_turns_gcs, data_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": COMPUTER_USE_15.format(
                            instruction=instruction,
                            language="en",
                            thought_mode=THOUGHT_LONG,
                        ),
                    }
                ],
            }
        ]
        for turn in turns:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "data:image/png;base64," + turn["image"],
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": turn["text"]},
                    ],
                }
            )

        # XXX: limit to 5 messages for now
        messages = messages[:5]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, _video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding="max_length",
            return_tensors="pt",
            max_length=self.seq_len,
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        labels = input_ids.clone()

        # Mask padding tokens in labels + image tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        image_tokens = [
            "<|vision_start|>",
            "<|vision_end|>",
            "<|vision_pad|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ]
        image_token_ids = self.processor.tokenizer.convert_tokens_to_ids(image_tokens)
        for image_token_id in image_token_ids:
            labels[labels == image_token_id] = -100

        return VlDataSample(
            input_ids=input_ids.to(torch.long),
            labels=labels.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )


class VlDatapackConfig(DatapackConfig[VlDataSample]):
    """
    Configuration class for the Text Pretraining Data Module.
    This class is used to configure the data module for text pretraining tasks.
    """

    config_path: str = "modeling.data.trajectory_train.VlDatapackConfig"
    dataset_path: str = "gs://induction-labs/jonathan/sampled_trajectories/uitars_initial_osworld_fixed/samples_correct.jsonl"
    num_workers: int = 2

    def validate_module_compatibility(
        self, module_config: ModuleConfig[Any]
    ):  # -> VlSftLITConfig[Any]:
        """
        Validate that the Lightning module is compatible with the data module.
        This method should be implemented by subclasses to perform any necessary checks.
        """
        # assert isinstance(module_config, VlSftLITConfig), (
        #     "VlDatapackConfig can only be used with VlSftConfig."
        # )
        return module_config

    async def _init_dataset(self, full_config: ExperimentConfig) -> VlDataset:
        module_config = self.validate_module_compatibility(full_config.module)
        return await VlDataset.constructor(
            VlDatasetArgs(
                dataset_path=self.dataset_path,
                seq_len=full_config.run.sequence_length,
                tokenizer_name=module_config.tokenizer_name,
            )
        )


__all__ = ["VlDatapackConfig"]

if __name__ == "__main__":
    import asyncio

    async def main():
        result = await VlDataset.constructor(
            VlDatasetArgs(
                dataset_path="gs://induction-labs/jonathan/sampled_trajectories/uitars_initial_osworld_fixed/samples_correct.jsonl",
                seq_len=8192,
                tokenizer_name="ByteDance-Seed/UI-TARS-1.5-7B",
            )
        )
        print(await result.get_item(0))

    asyncio.run(main())
