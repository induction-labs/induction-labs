from __future__ import annotations

import math
from collections.abc import Generator

import torch
from pydantic import BaseModel
from smart_open import open as smart_open
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

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
    frames_per_chunk: int

    @property
    def fps_ratio(self) -> float:
        return int(self.input_video.fps / self.output_video.fps)

    # @model_validator(mode="after")
    # def validate_fps_ratio(self) -> StreamMetadata:
    #     assert self.input_video.fps % self.output_video.fps == 0, (
    #         f"input_video.fps[{self.input_video.fps}] should be divisible by "
    #         f"output_video.fps[{self.output_video.fps}]"
    #     )
    #     return self

    @property
    def total_num_chunks(self) -> int:
        return math.ceil(self.output_video.total_frames / self.frames_per_chunk)


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
    import decord

    with smart_open(args.video_path, "rb") as src, open(tmp_video_path, "wb") as dst:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            dst.write(chunk)
        dst.flush()
    vr = decord.VideoReader(tmp_video_path, ctx=decord.cpu(0))

    total_frames, video_fps = len(vr), vr.get_avg_fps()
    first_frame: torch.Tensor = vr.next()
    height, width, _ = first_frame.shape
    input_video_metadata = VideoMetadata(
        fps=video_fps,
        total_frames=total_frames,
        resolution=VideoResolution(height=height, width=width),
    )

    # This can be odd as well, shouldn't matter. We can clip to even number of frames if needed later.
    output_nframes = math.floor(args.output_fps * input_video_metadata.duration)
    # clipped_total_frames = output_nframes * fps_ratio
    # assert clipped_total_frames <= total_frames

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=0,
        max_pixels=args.max_pixels,
    )
    output_video_metadata = VideoMetadata(
        fps=args.output_fps,
        total_frames=output_nframes,
        resolution=VideoResolution(height=resized_height, width=resized_width),
    )

    stream_metadata = StreamMetadata(
        input_video=input_video_metadata,
        output_video=output_video_metadata,
        frames_per_chunk=args.frames_per_chunk,
    )

    def process_chunk(index: int):
        input_range_start = (
            index * stream_metadata.frames_per_chunk * stream_metadata.fps_ratio
        )
        # last input frame non-inclusive
        input_range_end = (
            input_range_start
            + stream_metadata.frames_per_chunk * stream_metadata.fps_ratio
        )
        input_range_end = min(input_range_end, stream_metadata.input_video.total_frames)
        assert input_range_end <= stream_metadata.input_video.total_frames, (
            f"last_input_frame[{input_range_end}] should be less than "
            f"input_video.total_frames[{stream_metadata.input_video.total_frames}]"
        )
        idx = (
            torch.arange(
                input_range_start,
                input_range_end,
                stream_metadata.fps_ratio,
            )
            .round()
            .long()
            .tolist()
        )
        video = vr.get_batch(idx).asnumpy()
        video = torch.tensor(video).permute(0, 3, 1, 2)
        video = F.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return video

    def video_generator() -> Generator[torch.Tensor, None, None]:
        for i in range(stream_metadata.total_num_chunks):
            yield process_chunk(i)

    return stream_metadata, video_generator()
