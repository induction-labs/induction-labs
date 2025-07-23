from __future__ import annotations

import asyncio
import functools
import logging
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic, Literal, Self, TypeVar, cast

import numpy as np
import torch
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from synapse.combined.combined_loader import CombinedLoaderArgs, combined_loader
from synapse.utils.logging import configure_logging
from synapse.video_loader.read_frames import fetch_metadata_from_zarr
from synapse.video_loader.typess import StreamMetadata
from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin

from modeling.config import DatapackConfig, ExperimentConfig, ModuleConfig
from modeling.config.data import BaseDataSample, BaseDataset

logger = configure_logging(__name__, level=logging.INFO)


R = TypeVar("R")


class RemoveKwargProxy(Generic[R]):
    """
    Proxy that wraps any callable (or callable-like object) and strips
    a specified keyword from **kwargs before delegating.
    """

    _target: Callable[..., R]
    _kw_to_strip: str

    def __init__(self, target: Callable[..., R], kw_to_strip: str) -> None:
        self._target = target
        self._kw_to_strip = kw_to_strip

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        # Drop the unwanted kwarg if present
        kwargs.pop(self._kw_to_strip, None)
        return self._target(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Forward attribute access to the wrapped object
        # (e.g. methods, properties, etc.)
        return getattr(self._target, name)


# Example usage:


# TODO: Make this depend on the model loll
@functools.lru_cache
def qwen_processor(processor_name: str) -> ProcessorMixin:
    """
    Returns a cached instance of the Qwen2_5OmniProcessor.
    This is used to avoid reloading the processor multiple times.
    """
    processor = AutoProcessor.from_pretrained(processor_name, use_fast=True)
    if hasattr(processor, "video_processor"):
        processor.video_processor = RemoveKwargProxy(
            processor.video_processor, "images"
        )  # type: ignore[assignment]
    # Otherwise everytime we call it it prints "Unused or unrecognized kwargs: images." kms

    return processor


type CursorPathArray = np.ndarray[
    tuple[int, Literal[2], Literal[3]], np.dtype[np.floating]
]  # (seq, (x,y), (m,n,a))

# (seq, frames_per_action, C, H, W)
type FramesArray = np.ndarray[
    tuple[int, int, Literal[3], int, int], np.dtype[np.integer]
]
type PaddingArray = np.ndarray[tuple[int], np.dtype[np.bool]]


class ActionDataSample(BaseDataSample):
    class QwenInputs(BaseDataSample):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        input_ids: torch.Tensor  # [batch, seq_length, ]
        attention_mask: torch.Tensor  # [batch, seq_length, ]
        pixel_values_videos: torch.Tensor  # [num_video_patches, 1176]
        video_grid_thw: torch.Tensor  # [num_videos, 3]
        video_second_per_grid: torch.Tensor = Field(
            validation_alias=AliasChoices("video_second_per_grid", "second_per_grid_ts")
        )

        def to_device(
            self, device: torch.device | str, non_blocking: bool = False
        ) -> Self:
            """
            Move the QwenInputs to the specified device.
            This method is used to ensure that the inputs are on the correct device for training.
            """
            self.input_ids = self.input_ids.to(device, non_blocking=non_blocking)
            self.attention_mask = self.attention_mask.to(
                device, non_blocking=non_blocking
            )
            self.pixel_values_videos = self.pixel_values_videos.to(
                device, non_blocking=non_blocking
            )
            self.video_grid_thw = self.video_grid_thw.to(
                device, non_blocking=non_blocking
            )
            self.video_second_per_grid = self.video_second_per_grid.to(
                device, non_blocking=non_blocking
            )
            return self

        @model_validator(mode="after")
        def check_dimensions(self) -> Self:
            """
            Validate that the module and datapack configurations are compatible.
            This method is called after the model is initialized to ensure compatibility.
            """
            assert self.input_ids.shape[0] == self.attention_mask.shape[0]
            return self

        @classmethod
        def combine_batch(cls: type[Self], batch: list[Self]) -> Self:  # type: ignore[override]
            """
            Combine a batch of QwenInputs into a single QwenInputs.
            This is used to prepare the inputs for training or evaluation.
            """
            input_ids = torch.stack([sample.input_ids for sample in batch])
            attention_mask = torch.stack([sample.attention_mask for sample in batch])
            pixel_values_videos = torch.concat(
                [sample.pixel_values_videos for sample in batch], dim=0
            )
            video_grid_thw = torch.concat(
                [sample.video_grid_thw for sample in batch], dim=0
            )
            video_second_per_grid = torch.concat(
                [sample.video_second_per_grid for sample in batch], dim=0
            )

            return cls(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                video_second_per_grid=video_second_per_grid,
            )

    qwen_inputs: QwenInputs
    cursor_path: torch.Tensor  # [seq_length, (x,y), (m,n,a)]
    action_tokens: torch.Tensor  # [seq_length], dtype=bool

    def to_device(self, device: torch.device | str, non_blocking: bool = False) -> Self:
        """
        Move the ActionDataSample to the specified device.
        This method is used to ensure that the sample is on the correct device for training.
        """
        self.qwen_inputs = self.qwen_inputs.to_device(device, non_blocking)
        self.cursor_path = self.cursor_path.to(device, non_blocking=non_blocking)
        self.action_tokens = self.action_tokens.to(device, non_blocking=non_blocking)
        return self

    @classmethod
    def combine_batch(cls, batch: list[Self]) -> Self:  # type: ignore[override]
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

        qwen_inputs = ActionDataSample.QwenInputs.combine_batch(
            [sample.qwen_inputs for sample in batch]
        )
        cursor_path = torch.stack([sample.cursor_path for sample in batch])
        action_tokens = torch.stack([sample.action_tokens for sample in batch])

        return cls(
            qwen_inputs=qwen_inputs,
            cursor_path=cursor_path,
            action_tokens=action_tokens,
        )


class VideoProcessorConfig(BaseModel):
    model_config = {"frozen": True}
    processor_name: str = "Qwen/Qwen2.5-Omni-7B"
    video_pad_token: str = "<|VIDEO|>"
    video_token_id: int = 151656
    action_token: str = "<|endoftext|>"
    vision_pad_token: str = "<|vision_pad|>"
    vision_pad_id: int = 151654
    action_token_id: int = 151643
    patch_size: int = 14
    merge_size: int = 2
    frames_per_action: int = 2

    @staticmethod
    def Qwen25O():
        return VideoProcessorConfig()

    @staticmethod
    def Qwen25VL(name: str = "Qwen/Qwen2.5-VL-7B") -> VideoProcessorConfig:
        return VideoProcessorConfig(
            processor_name=name,
            video_pad_token="<|video_pad|>",
            video_token_id=151656,
        )


# <|im_start|>system\nYou are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.<|im_end|>\n<|im_start|>user\n<|vision_bos|><|IMAGE|><|vision_eos|><|vision_bos|><|VIDEO|><|vision_eos|><|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n<|im_start|>assistant\n
def calc_tokens_per_action(image_pixels: int, config: VideoProcessorConfig) -> int:
    """
    Compute the number of tokens per frame based on the image pixels and patch size.
    This is a placeholder function and should be replaced with actual logic.

    THIS IS ACTUALLY THE NUMBER OF VIDEO TOKENS PER ACTION
    """
    # For now, we assume each frame corresponds to a fixed number of tokens.
    # This can be adjusted based on the actual dataset and requirements.
    tokens_per_frame = image_pixels / ((config.patch_size * config.merge_size) ** 2)
    assert tokens_per_frame.is_integer(), (
        f"Expected tokens_per_frame to be an integer, got {tokens_per_frame}. {image_pixels=}, {config.patch_size=}"
    )
    return int(tokens_per_frame)


def calc_num_actions_per_sequence(
    stream_metadata: StreamMetadata,
    max_seq: int,
    raw_prompt: str,
    config: VideoProcessorConfig,
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
        max_video_tokens = max_seq - prompt_token_len(raw_prompt, config)
        return max_video_tokens // (calc_tokens_per_action(image_pixels, config) + 1)

    max_fittable_frames = max_actions_per_sequence(
        stream_metadata.output_video.resolution.pixels, max_seq
    )

    total_actions_in_stream = (
        stream_metadata.output_video.total_frames // config.frames_per_action
    )
    num_actions = min(max_fittable_frames, total_actions_in_stream)

    return num_actions


def calc_min_num_tokens_for_n_actions(
    resolution_pixels: int,
    num_actions: int,
    raw_prompt: str,
    config: VideoProcessorConfig,
) -> int:
    """
    Calculate the minimum number of tokens required for a given number of actions.
    This is a placeholder function and should be replaced with actual logic.
    """
    tokens_per_action = (
        calc_tokens_per_action(resolution_pixels, config) + 1
    )  # # +1 for the action token
    return prompt_token_len(raw_prompt, config) + num_actions * (tokens_per_action)


DEFAULT_PREAMBLE = "You are a GUI agent. Find the cursor at the end of video."
RAW_DEFAULT_PREAMBLE = f"<|im_start|>system\n{DEFAULT_PREAMBLE}<|im_end|>\n"
RAW_PREAMBLE_SUFFIX = "<|vision_end|><|im_end|>\n<|im_start|>assistant\n"
# RAW_DEFAULT_PREAMBLE = ""<|vision_start|><|video_pad|><|vision_end|>


def make_raw_prompt(
    config: VideoProcessorConfig,
    prefix=RAW_DEFAULT_PREAMBLE,
    suffix=RAW_PREAMBLE_SUFFIX,
) -> str:
    """
    Create a raw prompt for the Qwen model.
    This is used to ensure that the preamble is correctly formatted and included in the input sequence.
    """
    return f"{prefix}<|im_start|>user\n<|vision_start|>{config.video_pad_token}{suffix}"


@functools.lru_cache
def prompt_token_len(raw_prompt: str, config: VideoProcessorConfig) -> int:
    """
    Returns the length of the default preamble in tokens.
    This is used to ensure that the preamble is correctly tokenized and included in the input sequence.
    """
    processor = qwen_processor(config.processor_name)
    assert config.video_pad_token in raw_prompt, (
        f"Expected raw_prompt to contain video_pad_token {config.video_pad_token}, "
        f"but got {raw_prompt}."
    )
    # NOTE: Qwen you can't get rid of VIDEO_PAD entirely because otherwise `\n<|vision_bos|>` will be tokenized differently
    # But we can't use VIDEO_PAD directly otherwise qwen processor tries to calculate how many video tokens it needs
    # but we don't want to pass in video_thw
    # So instead we use vision_bos as a placeholder.
    preamble_without_video = raw_prompt.replace(
        config.video_pad_token, config.vision_pad_token
    )
    inputs = processor(
        text=[preamble_without_video],
        return_tensors="pt",
        padding=False,
    )
    # Assert no video tokens are present in the input_ids
    assert inputs.input_ids[0].eq(config.vision_pad_id).sum() == 1, (
        f"Expected input_ids to contain exactly one video_token_id {config.video_token_id}, "
        f"but got {inputs.input_ids[0]}."
    )
    assert inputs.input_ids.ne(config.video_token_id).all(), (
        f"Expected input_ids to not contain VIDEO_TOKEN_ID, but got {inputs.input_ids}."
    )
    return inputs.input_ids.shape[1] - 1


def insert_every_n(t: torch.Tensor, n: int, fill_value=0) -> torch.Tensor:
    assert t.ndim == 1
    L = t.numel()
    # how many zeros we'll end up inserting
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
    raw_prompt: str,
    config: VideoProcessorConfig,
    start: int = 0,
) -> tuple[ActionDataSample, FramesArray]:
    """
    Asynchronously fetch data from the given path and metadata.
    This is a placeholder implementation and should be replaced with actual data loading logic.
    """
    combined_loader_args = CombinedLoaderArgs(
        frames_zarr_path=path,
        actions_range=(start, start + num_actions),
        frames_per_action=config.frames_per_action,
    )
    (frames, cursor_path), stream_metadata = await combined_loader(combined_loader_args)
    # frames[:, 0, :] = frames[:, 1, :]
    assert len(frames) == len(cursor_path) == num_actions, (
        f"Frames and mouse movements length mismatch: {len(frames)} != {len(cursor_path)}"
    )

    assert frames.ndim == 5
    frames = frames.reshape(
        frames.shape[0] * frames.shape[1],
        frames.shape[2],
        frames.shape[3],
        frames.shape[4],
    )
    # frames[:, :, 0::96] = 0  # Set every 48th pixel to 0

    processor = qwen_processor(config.processor_name)
    # processor takes videos as inpuit as a list of videos as [frames, channels, height, width]
    # where frames = actions_per_sequence * frames_per_action
    inputs = processor(
        text=[raw_prompt],
        videos=[frames],
        return_tensors="pt",
        padding=False,
    )
    qwen_inputs = ActionDataSample.QwenInputs(**inputs)
    # qwen_inputs.attention_mask = qwen_inputs.attention_mask[0]
    # qwen_inputs.input_ids = qwen_inputs.input_ids[0]
    # return ActionDataSample(
    #     qwen_inputs=qwen_inputs,
    #     cursor_path=torch.tensor(
    #         cursor_path, dtype=torch.float32
    #     ),  # [seq_length, (x,y), (m,n,a)]
    #     action_tokens=torch.zeros((seq_length, 1), dtype=torch.float32),
    # ), frames

    # Insert 0 on every action prediction event in input_ids and attention_mask
    tokens_per_action = calc_tokens_per_action(
        stream_metadata.output_video.resolution.pixels, config
    )
    num_video_tokens = tokens_per_action * num_actions
    assert (
        qwen_inputs.pixel_values_videos.shape[0]
        == num_video_tokens * config.merge_size * config.merge_size
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
    assert qwen_inputs.attention_mask.eq(1).all(), (
        f"Expected attention_mask to be all ones, but got {qwen_inputs.attention_mask}."
    )
    is_video_token = qwen_inputs.input_ids[0].eq(config.video_token_id)
    first_video_token_index = torch.where(is_video_token)[0][0].item()

    non_video_input_ids, video_input_ids, post_video_input_ids = (
        qwen_inputs.input_ids[0][:first_video_token_index],
        qwen_inputs.input_ids[0][
            first_video_token_index : first_video_token_index + num_video_tokens
        ],
        qwen_inputs.input_ids[0][first_video_token_index + num_video_tokens :],
    )

    assert video_input_ids.eq(config.video_token_id).all(), (
        f"Expected input_ids to have {num_video_tokens} config.video_token_ids at the end, "
        f"but got {video_input_ids}."
    )
    assert non_video_input_ids.ne(config.video_token_id).all(), (
        f"Expected input_ids to have {num_video_tokens} config.video_token_ids at the end, "
        f"but got {non_video_input_ids}."
    )
    assert post_video_input_ids.ne(config.video_token_id).all()
    logger.debug(
        f"{len(non_video_input_ids)=}, {len(video_input_ids)=}, {num_video_tokens=}"
        f"{seq_length=} {num_actions=} {tokens_per_action=}"
    )

    expanded_video_input_ids = insert_every_n(
        video_input_ids, tokens_per_action, config.action_token_id
    )
    assert len(expanded_video_input_ids) - len(video_input_ids) == num_actions, (
        f"Expected expanded_video_input_ids length to be {len(video_input_ids) + num_actions}, "
        f"but got {len(expanded_video_input_ids)}."
    )

    real_input_ids = torch.cat(
        (non_video_input_ids, expanded_video_input_ids, post_video_input_ids), dim=0
    )

    real_attention_mask = torch.ones_like(
        real_input_ids
    )  # .ne(config.action_token_id).to(qwen_inputs.attention_mask.dtype)
    action_tokens = real_input_ids.eq(config.action_token_id)
    input_cursor_path = torch.zeros(
        (len(real_input_ids), 2, 3), dtype=torch.float32
    )  # [seq_length, (x,y), (m,n,a)]
    input_cursor_path[action_tokens] = torch.tensor(cursor_path)
    # real_attention_mask = real_attention_mask & ~action_tokens

    assert len(real_input_ids) <= seq_length, (
        f"Expected input_ids length {len(real_input_ids)} to be less than or equal to seq_length {seq_length}, "
        f"but got {len(non_video_input_ids)=}, {len(expanded_video_input_ids)=},  {len(post_video_input_ids)=}, "
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
    ), cast(FramesArray, frames)


class ActionDatasetArgs(BaseModel):
    data_paths: list[str]
    max_seq_length: int
    raw_prompt: str
    processor_config: VideoProcessorConfig


class ActionDataset(BaseDataset[ActionDataSample, ActionDatasetArgs]):
    # @class_property
    @classmethod
    def data_cls(cls) -> type[ActionDataSample]:
        """
        Class property that should return the data class type.
        This is used to ensure that the dataset is compatible with the data class.
        """
        return ActionDataSample

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

    async def get_item(self, idx: int) -> ActionDataSample:
        """
        Asynchronously fetch a batch of data samples.
        This method should be implemented to load the actual data from the paths.
        """

        # TODO: Handle case where index is out of bounds by returning a 0 sample and emitting a warning
        path = self.args.data_paths[idx]
        stream_metadata = await fetch_metadata_from_zarr(path)

        sample, _ = await fetch_data(
            path=path,
            raw_prompt=self.args.raw_prompt,
            num_actions=calc_num_actions_per_sequence(
                stream_metadata,
                self.args.max_seq_length,
                raw_prompt=self.args.raw_prompt,
                config=self.args.processor_config,
            ),
            seq_length=self.args.max_seq_length,
            config=self.args.processor_config,
        )
        return sample

    @classmethod
    async def constructor(cls, args: ActionDatasetArgs) -> Self:
        # For now we are not going to do stream preprocessing and we are just going to trust that
        # the data is already in the correct format.

        # TODO: Later maybe do initialization / preprocessing to do dynamic length calcs?
        # stream_metadatas: list[StreamMetadata] = await (
        #     self._fetch_metadatas(self.data_paths)
        # )
        return cls(
            args=args,
            length=len(args.data_paths),
        )


class ActionDatapackConfig(DatapackConfig[ActionDataSample]):
    """
    Configuration class for the Text Pretraining Data Module.
    This class is used to configure the data module for text pretraining tasks.
    """

    config_path: str = "modeling.data.video_action.ActionDatapackConfig"
    raw_prompt: str

    @property
    @abstractmethod
    def data_paths(self) -> list[str]:
        """
        List of paths to the processed data files.
        This should be implemented by subclasses to provide the actual data paths.
        """
        raise NotImplementedError("Subclasses must implement data_paths.")

    frames_per_action: int = 2
    processor_config: VideoProcessorConfig

    def validate_module_compatibility(
        self, module_config: ModuleConfig[Any]
    ) -> ModuleConfig[Any]:
        """
        Validate that the Lightning module is compatible with the data module.
        This method should be implemented by subclasses to perform any necessary checks.
        """
        return module_config

    async def _init_dataset(self, full_config: ExperimentConfig) -> ActionDataset:
        return await ActionDataset.constructor(
            ActionDatasetArgs(
                data_paths=self.data_paths,
                max_seq_length=full_config.run.sequence_length,
                processor_config=self.processor_config,
                raw_prompt=self.raw_prompt,
            )
        )


class ListActionDatapackConfig(ActionDatapackConfig):
    """
    A specific implementation of ActionDatapackConfig that uses a list of data paths.
    This is useful for scenarios where the data paths are provided as a list.
    """

    config_path: str = "modeling.data.video_action.ListActionDatapackConfig"
    data_paths_list: list[str] = Field(
        default_factory=list,
    )

    @property
    def data_paths(self) -> list[str]:
        return self.data_paths_list


class RangeActionDatapackConfig(ActionDatapackConfig):
    """
    A specific implementation of ActionDatapackConfig that uses a range of data paths.
    This is useful for scenarios where the data paths are generated based on a range.
    """

    config_path: str = "modeling.data.video_action.RangeActionDatapackConfig"
    prefix: str = "gs://induction-labs/jonathan/synth/cursor_follow_v2/sample_"
    start_index: int = 0
    end_index: int

    @property
    def data_paths(self) -> list[str]:
        return [
            f"{self.prefix}{i}.zarr" for i in range(self.start_index, self.end_index)
        ]


__all__ = ["ActionDatapackConfig"]
