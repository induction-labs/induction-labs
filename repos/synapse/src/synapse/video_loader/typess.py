from __future__ import annotations

import math
from fractions import Fraction

from pydantic import BaseModel


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
    time_base: Fraction
    start_time: float
    end_time: float

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


class VideoResolution(BaseModel):
    width: int
    height: int

    def __str__(self):
        return f"{self.width}x{self.height}"

    def __repr__(self):
        return self.__str__()

    def pixels(self):
        return self.width * self.height


resolution_480p = VideoResolution(width=854, height=480)


class VideoProcessArgs(BaseModel):
    video_path: str
    max_frame_pixels: int
    output_fps: float
    output_path: str
    frames_per_chunk: int = 32
