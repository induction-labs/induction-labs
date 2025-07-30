from __future__ import annotations

import asyncio
import math
import subprocess
import tempfile
from collections.abc import Callable, Generator
from contextlib import _GeneratorContextManager, contextmanager
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np
import torch
from av.container import InputContainer
from av.filter import Graph

if TYPE_CHECKING:
    from av.filter import FilterContext, Graph
    from av.video.format import VideoFormat
import gcsfs
from smart_open import open as smart_open

from synapse.elapsed_timer.elapsed_timer import elapsed_timer
from synapse.video_loader.typess import (
    FramesMetadata,
    StreamMetadata,
    StreamVideoArgs,
    VideoMetadata,
    VideoResolution,
)

from .video_utils import IMAGE_FACTOR, smart_resize

fs = gcsfs.GCSFileSystem(project="induction-labs")


class VideoMetadataFetcher:
    @staticmethod
    def get_video_metadata(video_stream: av.VideoStream) -> VideoMetadata:
        if video_stream.duration is None:
            raise ValueError("Video duration is None")

        if video_stream.average_rate is None:
            raise ValueError("Video average rate (fps) is None")

        if video_stream.frames is None:
            raise ValueError("Video frames count is None")
        if video_stream.time_base is None:
            raise ValueError("Video time base is None")

        metadata = VideoMetadata(
            fps=video_stream.average_rate,
            total_frames=video_stream.frames,
            resolution=VideoResolution(
                height=video_stream.height,
                width=video_stream.width,
            ),
            start_pts=video_stream.start_time or 0,
            duration=video_stream.duration,
            time_base=video_stream.time_base,
        )
        return metadata

    @staticmethod
    async def from_path(video_path: str | Path) -> tuple[VideoMetadata, VideoFormat]:
        """Get metadata from a video file using PyAV."""

        def fetch_data(video_path: str | Path) -> tuple[VideoMetadata, VideoFormat]:
            """Fetch video metadata using PyAV."""
            with tempfile.NamedTemporaryFile(delete=True, suffix=".tmp") as tmp_path2:
                with smart_open(video_path, "rb") as src:
                    for chunk in iter(lambda: src.read(1 << 20), b""):
                        tmp_path2.write(chunk)
                    tmp_path2.flush()
                container = av.open(tmp_path2.name)
                metadata = VideoMetadataFetcher.get_video_metadata(
                    container.streams.video[0]
                )
                video_format = container.streams.video[0].format
                container.close()
                return metadata, video_format

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fetch_data, video_path)


def process_chunk(
    index: int,
    frame_iter: Generator[av.VideoFrame, None, None],
    stream_metadata: StreamMetadata,
    buffer_src: FilterContext,
    buffer_sink: FilterContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_frame_index = index * stream_metadata.output_frames_per_chunk
    last_frame_index = min(
        first_frame_index + stream_metadata.output_frames_per_chunk,
        stream_metadata.output_video.total_frames,
    )

    frames: list[np.ndarray] = []
    frame_pts: list[int] = []

    # Process each output frame for this chunk
    for frame_index in range(first_frame_index, last_frame_index):
        output_frame_pts = (
            frame_index * stream_metadata.output_pts_per_frame
            + stream_metadata.input_video.start_pts
        )
        assert output_frame_pts <= stream_metadata.input_video.end_pts, (
            f"{output_frame_pts=}, {stream_metadata=}"
        )

        # Advance through frames until we find one with pts >= output_frame_pts
        frame_found = False

        # Check if current_frame already meets our criteria

        # Advance to the next suitable frame
        try:
            while True:
                current_frame = next(frame_iter)
                # print(
                #     current_frame.pts,
                #     output_frame_pts,
                # )
                assert current_frame.pts <= stream_metadata.input_video.end_pts, (
                    f"Current frame PTS {current_frame.pts} exceeds end PTS "
                    f"{stream_metadata.input_video.end_pts}"
                )

                if current_frame.pts >= output_frame_pts:
                    buffer_src.push(current_frame)
                    filtered_frame = buffer_sink.pull()
                    frame_array = filtered_frame.to_ndarray(format="rgb24")  # type: ignore[name-defined]
                    frames.append(frame_array)
                    frame_pts.append(current_frame.pts)
                    frame_found = True
                    break
        except StopIteration:
            # End of stream reached
            print(
                f"End of stream reached while processing chunk {index}, "
                f"frame index {frame_index}. No more frames available."
            )
            break

        if not frame_found:
            print(
                f"Could not find a suitable frame for chunk {index}, "
                f"frame index {frame_index}. Stopping processing."
            )
            # If we couldn't find a suitable frame, we've reached the end
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
    return video, torch.tensor(frame_pts, dtype=torch.uint64)


async def get_stream_metadata(
    args: StreamVideoArgs,
) -> tuple[StreamMetadata, VideoFormat]:
    # Use pyav for metadata extraction
    input_video_metadata, video_format = await VideoMetadataFetcher.from_path(
        args.video_path
    )

    # This can be odd as well, shouldn't matter. We can clip to even number of frames if needed later.

    resized_height, resized_width = smart_resize(
        input_video_metadata.resolution.height,
        input_video_metadata.resolution.width,
        factor=IMAGE_FACTOR,
        min_pixels=0,
        max_pixels=args.max_pixels,
    )

    output_nframes = math.floor(args.output_fps * input_video_metadata.duration_seconds)

    output_video_metadata = FramesMetadata(
        fps=args.output_fps,
        total_frames=output_nframes,
        resolution=VideoResolution(height=resized_height, width=resized_width),
    )

    stream_metadata = StreamMetadata(
        input_video=input_video_metadata,
        output_video=output_video_metadata,
        output_frames_per_chunk=args.frames_per_chunk,
    )
    print(f"Stream metadata: {stream_metadata.model_dump_json(indent=2)}")

    return stream_metadata, video_format


VideoStreamContext = Callable[
    [],
    _GeneratorContextManager[Generator[av.VideoFrame, None, None], None, None],
]


def download_video_with_ffmpeg_copy(source_path: str, dest_path: str):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp_path2:
        with smart_open(source_path, "rb") as src:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                tmp_path2.write(chunk)
            tmp_path2.flush()
        cmd = [
            "ffmpeg",
            "-err_detect",
            "ignore_err",
            "-copyts",
            "-copytb",
            "1",
            "-avoid_negative_ts",
            "make_zero",
            "-i",
            tmp_path2.name,
            "-c",
            "copy",
            "-y",
            dest_path,
        ]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        container1 = av.open(dest_path)
        container2 = av.open(tmp_path2)
        stream1 = container1.streams.video[0]
        stream2 = container2.streams.video[0]
        assert stream1.start_time == stream2.start_time
        container1.close()
        container2.close()


async def configure_video_stream(
    args: StreamVideoArgs,
) -> tuple[StreamMetadata, VideoFormat, VideoStreamContext]:
    stream_metadata, video_format = await get_stream_metadata(args)

    @contextmanager
    def video_frame_stream():
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp_path:
            download_video_with_ffmpeg_copy(args.video_path, tmp_path.name)
            container = av.open(tmp_path.name)
            video_stream = container.streams.video[0]
            try:
                assert video_stream.start_time is not None
                frame_iter: Generator[av.VideoFrame, None, None] = (
                    (frame)  # Include PTS in the yielded tuple
                    for packet in container.demux(video_stream)
                    for frame in packet.decode()
                    if frame.pts is not None
                )  #  type: ignore[name-defined]
                yield frame_iter
            finally:
                container.close()

    return stream_metadata, video_format, video_frame_stream


async def get_folder_stream_metadata(
    args: StreamVideoArgs,
) -> tuple[StreamMetadata, VideoFormat, list[str]]:
    # Use pyav for metadata extraction
    from synapse.video_loader.file_utils import get_mp4_files

    video_paths = get_mp4_files(args.video_path)
    with elapsed_timer("get_video_metadatas"):
        input_video_datas = await asyncio.gather(
            *(VideoMetadataFetcher.from_path(video_path) for video_path in video_paths)
        )
    assert len(input_video_datas) > 0, (
        f"No video files found in {args.video_path}. "
        "Please provide a valid path with .mp4 files."
    )
    input_video_metadatas = [metadata for metadata, _ in input_video_datas]
    video_format = input_video_datas[0][1]

    # Reduce over the video metadatas list to get a single metadata
    def combine_metadata(acc: VideoMetadata, curr: VideoMetadata) -> VideoMetadata:
        # Check resolution and fps match
        assert acc.start_pts <= curr.start_pts, (
            f"Start PTS must be in order: {input_video_metadatas=}"
        )
        assert acc.end_pts <= curr.end_pts, (
            f"End PTS must be in order: {input_video_metadatas=}"
        )

        if acc.resolution != curr.resolution:
            raise AssertionError(
                f"Resolution mismatch: {acc.resolution} vs {curr.resolution}"
            )
        if acc.fps != curr.fps:
            raise AssertionError(f"FPS mismatch: {acc.fps} vs {curr.fps}")
        assert acc.time_base == curr.time_base, (
            f"Time base mismatch: {acc.time_base} vs {curr.time_base}"
        )

        # Check for gaps
        # Combine metadata
        return VideoMetadata(
            fps=acc.fps,
            total_frames=acc.total_frames + curr.total_frames,
            resolution=acc.resolution,
            start_pts=acc.start_pts,
            duration=curr.end_pts - acc.start_pts,
            time_base=acc.time_base,
        )

    input_video_metadata = reduce(combine_metadata, input_video_metadatas)

    # This can be odd as well, shouldn't matter. We can clip to even number of frames if needed later.

    resized_height, resized_width = smart_resize(
        input_video_metadata.resolution.height,
        input_video_metadata.resolution.width,
        factor=IMAGE_FACTOR,
        min_pixels=0,
        max_pixels=args.max_pixels,
    )

    output_nframes = math.floor(args.output_fps * input_video_metadata.duration_seconds)

    output_video_metadata = FramesMetadata(
        fps=args.output_fps,
        total_frames=output_nframes,
        resolution=VideoResolution(height=resized_height, width=resized_width),
    )

    stream_metadata = StreamMetadata(
        input_video=input_video_metadata,
        output_video=output_video_metadata,
        output_frames_per_chunk=args.frames_per_chunk,
    )
    print(f"Stream metadata: {stream_metadata.model_dump_json(indent=2)}")

    return stream_metadata, video_format, video_paths


async def configure_video_folder_stream(
    args: StreamVideoArgs,
) -> tuple[StreamMetadata, VideoFormat, VideoStreamContext]:
    stream_metadata, video_format, video_paths = await get_folder_stream_metadata(args)

    @contextmanager
    def folder_frame_stream():
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp_path:
            container: InputContainer | None = None
            try:

                def folder_frame_generator() -> Generator[av.VideoFrame, None, None]:
                    nonlocal container
                    for video_path in video_paths:
                        download_video_with_ffmpeg_copy(video_path, tmp_path.name)
                        container = av.open(tmp_path.name)
                        video_stream = container.streams.video[0]
                        assert video_stream.start_time is not None
                        frame_iter: Generator[av.VideoFrame, None, None] = (
                            (frame)
                            for packet in container.demux(video_stream)
                            for frame in packet.decode()
                            if frame.pts is not None
                        )  # type: ignore[name-defined]
                        yield from frame_iter
                        container.close()

                yield folder_frame_generator()
            finally:
                assert container is not None
                container.close()

    return stream_metadata, video_format, folder_frame_stream


def process_stream_tensors(
    frame_iter: Generator[av.VideoFrame, None, None],
    video_format: VideoFormat,
    stream_metadata: StreamMetadata,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
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

    # Create filter graph for resizing (shared across chunks)

    filter_graph = Graph()  # type: ignore  # noqa: PGH003
    # https://pyav.org/docs/stable/api/filter.html#av.filter.graph.Graph
    buffer_src = filter_graph.add_buffer(
        width=stream_metadata.input_video.resolution.width,
        height=stream_metadata.input_video.resolution.height,
        format=video_format,
        name="src",
        time_base=stream_metadata.input_video.time_base,
    )

    buffer_sink = filter_graph.add("buffersink")
    scale_filter = filter_graph.add(
        "scale",
        f"{stream_metadata.output_video.resolution.width}:{stream_metadata.output_video.resolution.height}:flags=bicubic+full_chroma_int+accurate_rnd",
    )
    buffer_src.link_to(scale_filter)
    scale_filter.link_to(buffer_sink)
    filter_graph.configure()

    for i in range(stream_metadata.total_num_chunks):
        yield process_chunk(i, frame_iter, stream_metadata, buffer_src, buffer_sink)
