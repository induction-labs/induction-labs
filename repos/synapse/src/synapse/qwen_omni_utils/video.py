from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Generator

import av
import numpy as np
import torch
from pydantic import BaseModel
from smart_open import open as smart_open

from synapse.video_loader.typess import (
    StreamMetadata,
    StreamVideoArgs,
    VideoMetadata,
    VideoResolution,
)

from .video_process import IMAGE_FACTOR, smart_resize


class VideoReaderMetadata(BaseModel):
    width: int
    height: int
    duration: float
    fps: float
    frame_count: int


def stream_video_to_tensors(
    args: StreamVideoArgs,
    tmp_video_path: str,
) -> tuple[StreamMetadata, Generator[tuple[torch.Tensor, torch.Tensor], None, None]]:
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
    # import decord

    # Download to temporary file first, then repair with pyav
    with tempfile.NamedTemporaryFile(delete=True, suffix=".tmp") as tmp_path2:
        with smart_open(args.video_path, "rb") as src:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                tmp_path2.write(chunk)
            tmp_path2.flush()

        cmd = [
            "ffmpeg",
            "-err_detect",
            "ignore_err",
            "-i",
            tmp_path2.name,
            "-c",
            "copy",
            "-y",
            tmp_video_path,
        ]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Use pyav for metadata extraction
    metadata_container = av.open(tmp_video_path)
    metadata_video_stream = metadata_container.streams.video[0]

    assert metadata_video_stream.width is not None, "Video width is None"
    assert metadata_video_stream.height is not None, "Video height is None"
    assert metadata_video_stream.duration is not None, "Video duration is None"
    assert metadata_video_stream.time_base is not None, "Video time_base is None"
    assert metadata_video_stream.average_rate is not None, "Video average_rate is None"
    assert metadata_video_stream.frames is not None, "Video frames count is None"
    time_base = metadata_video_stream.time_base
    time_start = metadata_video_stream.start_time or 0

    input_video_metadata = VideoMetadata(
        fps=float(metadata_video_stream.average_rate),
        total_frames=metadata_video_stream.frames,
        resolution=VideoResolution(
            height=metadata_video_stream.height,
            width=metadata_video_stream.width,
        ),
    )
    metadata_container.close()

    # This can be odd as well, shouldn't matter. We can clip to even number of frames if needed later.
    output_nframes = math.floor(args.output_fps * input_video_metadata.duration)

    resized_height, resized_width = smart_resize(
        input_video_metadata.resolution.height,
        input_video_metadata.resolution.width,
        factor=IMAGE_FACTOR,
        min_pixels=0,
        max_pixels=args.max_pixels,
    )

    container = av.open(tmp_video_path)
    output_video_metadata = VideoMetadata(
        fps=args.output_fps,
        total_frames=output_nframes,
        resolution=VideoResolution(height=resized_height, width=resized_width),
    )

    stream_metadata = StreamMetadata(
        time_base=time_base,
        start_time=time_start,
        end_time=time_start + input_video_metadata.duration,
        input_video=input_video_metadata,
        output_video=output_video_metadata,
        output_frames_per_chunk=args.frames_per_chunk,
    )

    def process_chunk(index: int):
        chunk_duration_start = (
            index * stream_metadata.chunk_duration + stream_metadata.start_time
        )
        chunk_duration_end = min(
            chunk_duration_start + stream_metadata.chunk_duration,
            stream_metadata.end_time,
        )
        output_timesteps = torch.arange(
            chunk_duration_start,
            chunk_duration_end,
            step=1 / stream_metadata.output_video.fps,
        ).numpy()
        # TODO: Fix there can be fucking off by one errors if fps_ratio is barely smaller than an integer.
        output_timesteps = output_timesteps[0 : stream_metadata.output_frames_per_chunk]

        # Use pyav to extract frames at specific indices
        frames: list[np.ndarray] = []
        frame_timestamps: list[float] = []
        video_stream = container.streams.video[0]
        assert video_stream.time_base is not None, (
            "Video stream time_base cannot be None"
        )

        # Create a filter graph for resizing on demand
        filter_graph = av.filter.Graph()  # type: ignore  # noqa: PGH003
        buffer_src = filter_graph.add_buffer(template=video_stream)
        buffer_sink = filter_graph.add("buffersink")
        scale_filter = filter_graph.add(
            "scale",
            f"{resized_width}:{resized_height}:flags=bicubic+full_chroma_int+accurate_rnd",
        )
        buffer_src.link_to(scale_filter)
        scale_filter.link_to(buffer_sink)
        filter_graph.configure()

        # Get the first frame's timestamp to use as offset
        video_start_time = video_stream.start_time or 0

        for output_timestamp in output_timesteps:
            # Seek to the exact timestamp
            timestamp = int(output_timestamp / video_stream.time_base)
            container.seek(timestamp, stream=video_stream)

            # Read frames until we get the one we want
            frame_found = False
            for packet in container.demux(video_stream):
                for frame in packet.decode():
                    assert video_stream.time_base is not None
                    # Calculate frame timestamp relative to video start
                    frame_time = (frame.pts - video_start_time) * video_stream.time_base
                    if (
                        frame_time
                        >= output_timestamp - video_start_time * video_stream.time_base
                    ):
                        # Apply filtering for resizing
                        buffer_src.push(frame)
                        filtered_frame = buffer_sink.pull()
                        frame_array = filtered_frame.to_ndarray(format="rgb24")
                        frames.append(frame_array)
                        frame_timestamps.append(float(frame_time))
                        frame_found = True
                        break
                if frame_found:
                    break

        video = np.stack(frames, axis=0)
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
        return video, torch.tensor(frame_timestamps, dtype=torch.float32)

    def video_generator() -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        try:
            for i in range(stream_metadata.total_num_chunks):
                yield process_chunk(i)
        finally:
            container.close()

    return stream_metadata, video_generator()
