from __future__ import annotations

import asyncio
from typing import Any, Self

import lightning as L
from modeling.config import DatapackConfig, ExperimentConfig, ModuleConfig
from pydantic import BaseModel, ConfigDict, model_validator
from synapse.video_loader.read_frames import fetch_metadata_from_zarr
from synapse.video_loader.typess import StreamMetadata
from torch.utils.data import DataLoader, Dataset
from synapse.combined.combined_loader import combined_loader, CombinedLoaderArgs
import numpy as np
from typing import Literal

from transformers.models.qwen2_5_omni import (
    Qwen2_5OmniProcessor,
)

import functools
import torch


VIDEO_TOKEN_ID = 151656
PATCH_SIZE = 14
MERGE_SIZE = 2
ACTION_TOKEN_ID = 151643


@functools.lru_cache(maxsize=1)
def qwen_processor() -> Qwen2_5OmniProcessor:
    """
    Returns a cached instance of the Qwen2_5OmniProcessor.
    This is used to avoid reloading the processor multiple times.
    """
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    assert isinstance(processor, Qwen2_5OmniProcessor)
    return processor


type CursorPathArray = np.ndarray[
    tuple[int, Literal[2], Literal[3]], np.dtype[np.floating]
]  # (seq, (x,y), (m,n,a))

# (seq, frames_per_action, C, H, W)
type FramesArray = np.ndarray[
    tuple[int, int, Literal[3], int, int], np.dtype[np.integer]
]
type PaddingArray = np.ndarray[tuple[int], np.dtype[np.bool]]


class ActionDataSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class QwenInputs(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        input_ids: torch.Tensor  # [batch, seq_length, ]
        attention_mask: torch.Tensor  # [batch, seq_length, ]
        pixel_values_videos: torch.Tensor  # [num_video_patches, 1176]
        video_grid_thw: torch.Tensor  # [num_videos, 3]
        video_second_per_grid: torch.Tensor  # [nun_videos,]

        @model_validator(mode="after")
        def check_dimensions(self) -> Self:
            """
            Validate that the module and datapack configurations are compatible.
            This method is called after the model is initialized to ensure compatibility.
            """
            assert self.input_ids.shape[0] == self.attention_mask.shape[0]
            return self

    qwen_inputs: QwenInputs
    cursor_path: torch.Tensor  # [seq_length, (x,y), (m,n,a)]
    action_tokens: torch.Tensor  # [seq_length], dtype=bool

    @classmethod
    def combine_batch(cls, batch: list[ActionDataSample]) -> ActionDataSample:
        """
        Combine a batch of ActionDataSamples into a single ActionDataSample.
        This is used to prepare the data for training or evaluation.
        """

        # Assert sequence lengths are the same
        seq_length = batch[0].qwen_inputs.input_ids.shape[0]
        assert all(
            sample.qwen_inputs.input_ids.shape[0] == seq_length for sample in batch
        ), (
            f"All samples in the batch must have the same sequence length, "
            f"but got {[sample.qwen_inputs.input_ids.shape[0] for sample in batch]}."
        )

        qwen_inputs = ActionDataSample.QwenInputs(
            input_ids=torch.stack([sample.qwen_inputs.input_ids for sample in batch]),
            attention_mask=torch.stack(
                [sample.qwen_inputs.attention_mask for sample in batch]
            ),
            pixel_values_videos=torch.concat(
                [sample.qwen_inputs.pixel_values_videos for sample in batch],
                dim=0,
            ),
            video_grid_thw=torch.concat(
                [sample.qwen_inputs.video_grid_thw for sample in batch],
                dim=0,
            ),
            video_second_per_grid=torch.concat(
                [sample.qwen_inputs.video_second_per_grid for sample in batch],
                dim=0,
            ),
        )
        cursor_path = torch.stack([sample.cursor_path for sample in batch])
        action_tokens = torch.stack([sample.action_tokens for sample in batch])

        return cls(
            qwen_inputs=qwen_inputs,
            cursor_path=cursor_path,
            action_tokens=action_tokens,
        )


# <|im_start|>system\nYou are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.<|im_end|>\n<|im_start|>user\n<|vision_bos|><|IMAGE|><|vision_eos|><|vision_bos|><|VIDEO|><|vision_eos|><|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n<|im_start|>assistant\n
def calc_tokens_per_action(image_pixels: int) -> int:
    """
    Compute the number of tokens per frame based on the image pixels and patch size.
    This is a placeholder function and should be replaced with actual logic.

    THIS IS ACTUALLY THE NUMBER OF VIDEO TOKENS PER ACTION
    """
    # For now, we assume each frame corresponds to a fixed number of tokens.
    # This can be adjusted based on the actual dataset and requirements.
    tokens_per_frame = image_pixels / ((PATCH_SIZE * MERGE_SIZE) ** 2)
    assert tokens_per_frame.is_integer(), (
        f"Expected tokens_per_frame to be an integer, got {tokens_per_frame}. {image_pixels=}, {PATCH_SIZE=}"
    )
    return int(tokens_per_frame)


def calc_num_actions_per_sequence(
    stream_metadata: StreamMetadata, max_seq: int, frames_per_action: int
) -> int:
    """
    Calculate the number of actions per sequence based on the image pixels, max sequence length, and frames per action.
    This is a placeholder function and should be replaced with actual logic.
    """

    def max_actions_per_sequence(image_pixels: int, max_seq: int) -> int:
        """
        Compute the number of frames per sequence based on the frames_per_action.
        This is a placeholder function and should be replaced with actual logic.
        """
        max_video_tokens = max_seq - default_preamble_token_len()
        return max_video_tokens // (calc_tokens_per_action(image_pixels) + 1)

    max_fittable_frames = max_actions_per_sequence(
        stream_metadata.output_video.resolution.pixels, max_seq
    )

    total_actions_in_stream = (
        stream_metadata.output_video.total_frames // frames_per_action
    )
    num_actions = min(max_fittable_frames, total_actions_in_stream)

    return num_actions


DEFAULT_PREAMBLE = "You are Qwen, a helpful video processing assistant."
# RAW_DEFAULT_PREAMBLE = f"<|im_start|>system\n{DEFAULT_PREAMBLE}<|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|>"
RAW_DEFAULT_PREAMBLE = "<|im_start|>user\n<|vision_bos|><|VIDEO|>"


@functools.lru_cache(maxsize=1)
def default_preamble_token_len() -> int:
    """
    Returns the length of the default preamble in tokens.
    This is used to ensure that the preamble is correctly tokenized and included in the input sequence.
    """
    processor = qwen_processor()
    preamble_without_video = RAW_DEFAULT_PREAMBLE.replace("<|VIDEO|>", "")
    inputs = processor(
        text=[preamble_without_video],
        return_tensors="pt",
        padding=False,
    )
    # Assert no video tokens are present in the input_ids
    assert inputs.input_ids.ne(VIDEO_TOKEN_ID).all(), (
        f"Expected input_ids to not contain VIDEO_TOKEN_ID, but got {inputs.input_ids}."
    )
    # assert inputs.input_ids.shape[1] == 34

    return inputs.input_ids.shape[1]


def insert_every_n(t: torch.Tensor, n: int, fill_value=0) -> torch.Tensor:
    assert t.ndim == 1
    L = t.numel()
    # how many zeros we’ll end up inserting
    num_insert = L // n

    # new length = original + inserted zeros
    new_len = L + num_insert

    # build a boolean mask of length new_len:
    #  - False where we want the zero
    #  - True everywhere else (to hold the original entries)
    mask = torch.ones(new_len, dtype=torch.bool, device=t.device)
    mask[n :: n + 1] = False

    # allocate output, fill zeros at mask==False, then scatter original values
    out = torch.full(
        (new_len,),
        fill_value=fill_value,
        dtype=t.dtype,
        device=t.device,
    )
    out[mask] = t
    return out


async def fetch_data(
    path: str,
    num_actions: int,
    seq_length: int,
    frames_per_action: int = 2,
) -> ActionDataSample:
    """
    Asynchronously fetch data from the given path and metadata.
    This is a placeholder implementation and should be replaced with actual data loading logic.
    """
    combined_loader_args = CombinedLoaderArgs(
        frames_zarr_path=path,
        actions_range=(0, num_actions),
        frames_per_action=frames_per_action,
    )
    (frames, cursor_path), stream_metadata = await combined_loader(combined_loader_args)
    assert len(frames) == len(cursor_path) == num_actions, (
        f"Frames and mouse movements length mismatch: {len(frames)} != {len(cursor_path)}"
    )

    assert frames.ndim == 5
    print(frames.shape)
    frames = frames.reshape(
        frames.shape[0] * frames.shape[1],
        frames.shape[2],
        frames.shape[3],
        frames.shape[4],
    )

    # template to chat

    processor = qwen_processor()
    # processor takes videos as inpuit as a list of videos as [frames, channels, height, width]
    # where frames = actions_per_sequence * frames_per_action
    inputs = processor(
        text=[RAW_DEFAULT_PREAMBLE],
        videos=[frames],
        return_tensors="pt",
        padding=False,
    )
    qwen_inputs = ActionDataSample.QwenInputs(**inputs)

    # Insert 0 on every action prediction event in input_ids and attention_mask
    tokens_per_action = calc_tokens_per_action(
        stream_metadata.output_video.resolution.pixels
    )
    num_video_tokens = tokens_per_action * num_actions
    assert (
        qwen_inputs.pixel_values_videos.shape[0]
        == num_video_tokens * MERGE_SIZE * MERGE_SIZE
    ), (
        f"Expected pixel_values_videos to have {num_video_tokens} frames, "
        f"but got {qwen_inputs.pixel_values_videos.shape[0]}."
    )
    # pixel_values_videos.shape =
    # (batch_size,
    # grid_t * grid_h * MERGE_SIZE * grid_w * MERGE_SIZE,
    # channel * temporal_patch_size * patch_size * patch_size) (1176)
    # num_video_tokens = grid_t * grid_h * grid_w
    assert qwen_inputs.input_ids.shape[0] == 1

    non_video_input_ids, video_input_ids = (
        qwen_inputs.input_ids[0][:-num_video_tokens],
        qwen_inputs.input_ids[0][-num_video_tokens:],
    )
    assert qwen_inputs.attention_mask.eq(1).all(), (
        f"Expected attention_mask to be all ones, but got {qwen_inputs.attention_mask}."
    )
    assert video_input_ids.eq(VIDEO_TOKEN_ID).all(), (
        f"Expected input_ids to have {num_video_tokens} VIDEO_TOKEN_IDs at the end, "
        f"but got {video_input_ids}."
    )
    assert non_video_input_ids.ne(VIDEO_TOKEN_ID).all(), (
        f"Expected input_ids to have {num_video_tokens} VIDEO_TOKEN_IDs at the end, "
        f"but got {non_video_input_ids}."
    )
    print(
        f"{len(non_video_input_ids)=}, {len(video_input_ids)=}, {num_video_tokens=}"
        f"{seq_length=} {num_actions=} {tokens_per_action=}"
    )

    expanded_video_input_ids = insert_every_n(
        video_input_ids, tokens_per_action, ACTION_TOKEN_ID
    )
    assert len(expanded_video_input_ids) - len(video_input_ids) == num_actions, (
        f"Expected expanded_video_input_ids length to be {len(video_input_ids) + num_actions}, "
        f"but got {len(expanded_video_input_ids)}."
    )

    real_input_ids = torch.cat((non_video_input_ids, expanded_video_input_ids), dim=0)
    real_attention_mask = torch.ones_like(
        real_input_ids
    )  # .ne(ACTION_TOKEN_ID).to(qwen_inputs.attention_mask.dtype)
    action_tokens = real_input_ids.eq(ACTION_TOKEN_ID)
    input_cursor_path = torch.zeros(
        (len(real_input_ids), 2, 3), dtype=torch.float32
    )  # [seq_length, (x,y), (m,n,a)]
    input_cursor_path[action_tokens] = torch.tensor(cursor_path)

    assert len(real_input_ids) <= seq_length, (
        f"Expected input_ids length {len(real_input_ids)} to be less than or equal to seq_length {seq_length}, "
        f"but got {real_input_ids}."
    )
    if len(real_input_ids) < seq_length:
        padding_len = seq_length - len(real_input_ids)
        real_input_ids = torch.cat(
            (
                real_input_ids,
                torch.zeros(padding_len, dtype=qwen_inputs.input_ids.dtype),
            )
        )
        real_attention_mask = torch.cat(
            (
                real_attention_mask,
                torch.zeros(padding_len, dtype=qwen_inputs.attention_mask.dtype),
            )
        )
        action_tokens = torch.cat(
            (action_tokens, torch.zeros(padding_len, dtype=torch.bool))
        )
        input_cursor_path = torch.cat(
            (input_cursor_path, torch.zeros((padding_len, 2, 3), dtype=torch.float32))
        )

    return ActionDataSample(
        qwen_inputs=ActionDataSample.QwenInputs(
            input_ids=real_input_ids,
            attention_mask=real_attention_mask,
            pixel_values_videos=qwen_inputs.pixel_values_videos,
            video_grid_thw=qwen_inputs.video_grid_thw,
            video_second_per_grid=qwen_inputs.video_second_per_grid,
        ),
        cursor_path=input_cursor_path,
        action_tokens=action_tokens,
    )


class ActionDatasetArgs(BaseModel):
    data_paths: list[str]
    max_seq_length: int
    frames_per_action: int


class ActionDataset(Dataset[ActionDataSample]):
    @property
    def data_paths(self) -> list[str]:
        return self.config.data_paths

    def __init__(
        self,
        config: ActionDatasetArgs,
    ):
        super().__init__()
        self.config = config

        # TODO: get rid of running asyncio run in sync functions :///
        stream_metadatas: list[StreamMetadata] = asyncio.run(
            self._fetch_metadatas(self.data_paths)
        )

        self.datas = list(zip(self.data_paths, stream_metadatas, strict=True))

    def __getitem__(self, index) -> ActionDataSample:
        # TODO: Make everything async native
        return asyncio.run(self.get_item(index))

    @staticmethod
    async def _fetch_metadatas(metadata_paths: list[str]) -> list[StreamMetadata]:
        """
        Asynchronously fetch metadata for the given paths.
        This method should be implemented to load the actual metadata from the paths.
        """
        # Placeholder implementation, replace with actual metadata loading logic
        return await asyncio.gather(
            *[fetch_metadata_from_zarr(path) for path in metadata_paths]
        )

    async def get_item(self, index: int) -> ActionDataSample:
        """
        Asynchronously fetch a batch of data samples.
        This method should be implemented to load the actual data from the paths.
        """
        # TODO: Handle case where index is out of bounds by returning a 0 sample and emitting a warning
        path, stream_metadata = self.datas[index]

        # Placeholder implementation, replace with actual data loading logic
        # For now, we just return empty samples
        return await fetch_data(
            path=path,
            num_actions=calc_num_actions_per_sequence(
                stream_metadata,
                self.config.max_seq_length,
                self.config.frames_per_action,
            ),
            seq_length=self.config.max_seq_length,
        )

    def __len__(self) -> int:
        # For now only pull the first sequence from each.
        # TODO: Worry about randomization and shit.
        return len(self.datas)


class ActionDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: ActionDatapackConfig,
        extra_args: ActionDataModuleExtraArgs,
    ):
        super().__init__()
        self.config = config
        self.extra_args = extra_args

    def setup(self, stage: str | None = None) -> None:
        # This method is used to download the dataset if it is not already present.
        data_paths = [
            f"gs://induction-labs/jonathan/synth/cursor_follow_v2/sample_{i}.zarr"
            for i in range(0, 10_000)
        ]

        self.train_data = ActionDataset(
            ActionDatasetArgs(
                data_paths=data_paths,
                max_seq_length=self.extra_args.seq_length,
                frames_per_action=self.config.frames_per_action,
            )
        )

    def train_dataloader(self) -> DataLoader[ActionDataSample]:
        return DataLoader(
            self.train_data,
            batch_size=self.extra_args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=ActionDataSample.combine_batch,
        )


class ActionDataModuleExtraArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    seq_length: int = 1024  # Default sequence length
    batch_size: int


class ActionDatapackConfig(DatapackConfig[ActionDataModule]):
    """
    Configuration class for the Text Pretraining Data Module.
    This class is used to configure the data module for text pretraining tasks.
    """

    config_path: str = "modeling.data.video_action.ActionDatapackConfig"
    processed_data_paths: list[str]
    frames_per_action: int = 2
    patch_size: int = PATCH_SIZE

    def validate_module_compatibility(
        self, module_config: ModuleConfig[Any]
    ) -> ModuleConfig[Any]:
        """
        Validate that the Lightning module is compatible with the data module.
        This method should be implemented by subclasses to perform any necessary checks.
        """
        return module_config

    def create_datapack(
        self, full_config: ExperimentConfig[ActionDataModule]
    ) -> ActionDataModule:
        self.validate_module_compatibility(full_config.module)
        extra_args = ActionDataModuleExtraArgs(
            seq_length=full_config.run.sequence_length,
            batch_size=full_config.run.batch_size,
        )
        return ActionDataModule(self, extra_args)


__all__ = ["ActionDataModule", "ActionDatapackConfig"]
