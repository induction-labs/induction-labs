from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Generator

import av
import numpy as np
import torch
from smart_open import open as smart_open

from synapse.video_loader.typess import (
    FramesMetadata,
    StreamMetadata,
    StreamVideoArgs,
    VideoMetadata,
    VideoResolution,
)

from .video_utils import IMAGE_FACTOR, smart_resize


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Get metadata from a video file using PyAV."""
    container = av.open(video_path)
    video_stream = container.streams.video[0]

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
    container.close()
    return metadata


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
    input_video_metadata = get_video_metadata(tmp_video_path)

    # This can be odd as well, shouldn't matter. We can clip to even number of frames if needed later.

    resized_height, resized_width = smart_resize(
        input_video_metadata.resolution.height,
        input_video_metadata.resolution.width,
        factor=IMAGE_FACTOR,
        min_pixels=0,
        max_pixels=args.max_pixels,
    )

    container = av.open(tmp_video_path)
    video_stream = container.streams.video[0]

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

    def process_chunk(
        index: int,
        frame_iter: Generator[av.VideoFrame, None, None],
        buffer_src: av.filter.Buffer,  # type: ignore[name-defined]
        buffer_sink: av.filter.Buffersink,  # type: ignore[name-defined]
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
                    if current_frame.pts >= output_frame_pts:
                        buffer_src.push(current_frame)
                        filtered_frame = buffer_sink.pull()
                        frame_array = filtered_frame.to_ndarray(format="rgb24")
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

    def video_generator() -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        nonlocal container, video_stream, stream_metadata, resized_width, resized_height
        try:
            frame_iter: Generator[av.VideoFrame, None, None] = (
                frame
                for packet in container.demux(video_stream)
                for frame in packet.decode()
                if frame.pts is not None
            )  # type: ignore[name-defined]

            # Create filter graph for resizing (shared across chunks)
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

            for i in range(stream_metadata.total_num_chunks):
                yield process_chunk(i, frame_iter, buffer_src, buffer_sink)
        finally:
            container.close()

    return stream_metadata, video_generator()
