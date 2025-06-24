from __future__ import annotations

import math
from collections.abc import Generator

import numpy as np
import torch
from pydantic import BaseModel
from smart_open import open as smart_open
from video_reader import PyVideoReader

from synapse.video_loader.types import VideoResolution

from .video_process import IMAGE_FACTOR, get_video_reader_backend, smart_resize


class StreamVideoArgs(BaseModel):
    output_fps: float
    video_path: str
    max_pixels: int
    frames_per_chunk: int


class VideoMetadata(BaseModel):
    fps: float
    total_frames: int
    resolution: VideoResolution

    @property
    def duration(self) -> float:
        return self.total_frames / self.fps


class StreamMetadata(BaseModel):
    input_video: VideoMetadata
    output_video: VideoMetadata
    output_frames_per_chunk: int

    @property
    def fps_ratio(self) -> float:
        return self.input_video.fps / self.output_video.fps

    @property
    def input_frames_per_chunk(self) -> float:
        """Number of frames in the input video per chunk."""
        return self.output_frames_per_chunk * self.fps_ratio

    @property
    def chunk_duration(self) -> float:
        """Duration of each chunk in seconds."""
        return self.output_frames_per_chunk / self.output_video.fps

    @property
    def total_num_chunks(self) -> int:
        return math.ceil(self.output_video.total_frames / self.output_frames_per_chunk)


class VideoReaderMetadata(BaseModel):
    width: int
    height: int
    duration: float
    fps: float
    frame_count: int


def stream_video_to_tensors(
    args: StreamVideoArgs,
    tmp_video_path: str,
) -> tuple[StreamMetadata, Generator[torch.Tensor, None, None]]:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    assert get_video_reader_backend() == "decord", (
        "fetch_video_decord should only be used when decord is available."
    )
    # import decord

    with smart_open(args.video_path, "rb") as src, open(tmp_video_path, "wb") as dst:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            dst.write(chunk)
        dst.flush()
    vr = PyVideoReader(tmp_video_path)
    reader_metadata = VideoReaderMetadata(**vr.get_info())

    input_video_metadata = VideoMetadata(
        fps=reader_metadata.fps,
        total_frames=reader_metadata.frame_count,
        resolution=VideoResolution(
            height=reader_metadata.height,
            width=reader_metadata.width,
        ),
    )

    # This can be odd as well, shouldn't matter. We can clip to even number of frames if needed later.
    output_nframes = math.floor(args.output_fps * input_video_metadata.duration)

    resized_height, resized_width = smart_resize(
        input_video_metadata.resolution.height,
        input_video_metadata.resolution.width,
        factor=IMAGE_FACTOR,
        min_pixels=0,
        max_pixels=args.max_pixels,
    )

    shorter, longer = sorted([resized_height, resized_width])
    vr = PyVideoReader(
        tmp_video_path,
        filter=f"scale={resized_width}:{resized_height}:flags=bicubic+full_chroma_int+accurate_rnd",
        resize_shorter_side=shorter,
        resize_longer_side=longer,
    )
    output_video_metadata = VideoMetadata(
        fps=args.output_fps,
        total_frames=output_nframes,
        resolution=VideoResolution(height=resized_height, width=resized_width),
    )

    stream_metadata = StreamMetadata(
        input_video=input_video_metadata,
        output_video=output_video_metadata,
        output_frames_per_chunk=args.frames_per_chunk,
    )

    def process_chunk(index: int):
        chunk_duration_start = index * stream_metadata.chunk_duration
        chunk_duration_end = (index + 1) * stream_metadata.chunk_duration
        input_range_start = chunk_duration_start * stream_metadata.input_video.fps
        # last input frame non-inclusive
        input_range_end = chunk_duration_end * stream_metadata.input_video.fps
        input_range_end = min(input_range_end, stream_metadata.input_video.total_frames)

        idx = (
            torch.arange(
                input_range_start,
                input_range_end,
                stream_metadata.fps_ratio,
            )
            .round()
            .long()
            .numpy()
        )
        # TODO: Fix there can be fucking off by one errors if fps_ratio is barely smaller than an integer.
        assert len(idx) <= stream_metadata.output_frames_per_chunk + 1, (
            f"idx[{idx}] should have length less than or equal to "
            f"output_frames_per_chunk[{stream_metadata.output_frames_per_chunk}]"
        )
        idx = idx[0 : stream_metadata.output_frames_per_chunk]

        assert np.all(idx < stream_metadata.input_video.total_frames), (
            f"idx[{idx}] should be less than "
            f"input_video.total_frames[{stream_metadata.input_video.total_frames}]"
        )

        # Important! Need to set `with_fallback=True` to avoid random EOF errors.
        video = vr.get_batch(idx, True)
        assert isinstance(video, np.ndarray), (
            f"video should be a numpy array, got {type(video)}"
        )
        video = torch.tensor(video).permute(0, 3, 1, 2)
        assert video.shape[1:] == (
            3,
            stream_metadata.output_video.resolution.height,
            stream_metadata.output_video.resolution.width,
        ), (
            f"video shape should be (T, C, H, W), got {video.shape[1:]} "
            f"for chunk {index} with resolution "
            f"{stream_metadata.output_video.resolution}"
        )
        return video

    def video_generator() -> Generator[torch.Tensor, None, None]:
        for i in range(stream_metadata.total_num_chunks):
            yield process_chunk(i)

    return stream_metadata, video_generator()
