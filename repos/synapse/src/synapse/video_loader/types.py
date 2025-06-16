from __future__ import annotations

from pydantic import BaseModel


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
