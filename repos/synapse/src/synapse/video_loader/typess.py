from __future__ import annotations

import math
from fractions import Fraction

from pydantic import BaseModel


class StreamVideoArgs(BaseModel):
    output_fps: Fraction
    video_path: str
    max_pixels: int
    frames_per_chunk: int


class VideoMetadata(BaseModel):
    fps: Fraction
    total_frames: int
    resolution: VideoResolution


class StreamMetadata(BaseModel):
    time_base: Fraction
    start_pts: int
    end_pts: int

    input_video: VideoMetadata
    output_video: VideoMetadata
    output_frames_per_chunk: int

    @property
    def start_time(self) -> Fraction:
        """Start time of the video in seconds."""
        return self.start_pts * self.time_base

    @property
    def end_time(self) -> Fraction:
        """End time of the video in seconds."""
        return self.end_pts * self.time_base

    @property
    def fps_ratio(self) -> Fraction:
        return self.input_video.fps / self.output_video.fps

    @property
    def input_frames_per_chunk(self) -> Fraction:
        """Number of frames in the input video per chunk."""
        return self.output_frames_per_chunk * self.fps_ratio

    @property
    def chunk_duration(self) -> Fraction:
        """Duration of each chunk in seconds."""
        return self.output_frames_per_chunk / self.output_video.fps

    @property
    def output_chunk_pts(self) -> int:
        return self.output_pts_per_frame * self.output_frames_per_chunk

    @property
    def output_pts_per_frame(self) -> int:
        pts_per_frame = 1 / (self.time_base * self.output_video.fps)
        assert pts_per_frame.denominator == 1, (
            f"{self.output_video.fps=}, f{self.time_base=}"
        )
        return pts_per_frame.numerator

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
    output_fps: Fraction
    output_path: str
    frames_per_chunk: int = 32
