from __future__ import annotations

import tensorstore as ts
from synapse.elapsed_timer import elapsed_timer
from synapse.qwen_omni_utils.video import StreamVideoArgs, stream_video_to_tensors
from synapse.utils.logging import configure_logging

# from zarr import Array
from tqdm import tqdm

from .types import VideoProcessArgs
from .zarr_utils import (
    ZarrArrayAttributes,
    append_batch,
    create_zarr_array,
)

logger = configure_logging(__name__)


async def process_video(
    args: VideoProcessArgs,
):
    logger.debug(f"Processing video with args: {args}")
    with elapsed_timer("process_video") as timer:
        # root = get_zarr_root(fs, args.output_path)
        stream_args = StreamVideoArgs(
            output_fps=args.output_fps,
            video_path=args.video_path,
            max_pixels=args.max_frame_pixels,
            frames_per_chunk=args.frames_per_chunk,
        )
        stream_metadata, video_frames = stream_video_to_tensors(stream_args)

        # Create the Zarr array if it does not exist
        zarr_array = await create_zarr_array(
            ZarrArrayAttributes(
                chunk_shape=(
                    args.frames_per_chunk,
                    3,
                    stream_metadata.output_video.resolution.width,
                    stream_metadata.output_video.resolution.height,
                ),
                shape=(
                    0,
                    3,
                    stream_metadata.output_video.resolution.width,
                    stream_metadata.output_video.resolution.height,
                ),  # Start with 0 frames
                dtype=ts.uint8,
                path=args.output_path,
            ),
        )
        # assert isinstance(zarr_array, Array)
        # logger.debug(f"Created Zarr array with attributes: {zarr_array.attrs}")

        # Process and append frames to the Zarr array
        for i, frames in enumerate(
            tqdm(
                video_frames,
                desc="Processing video frames",
                total=stream_metadata.total_num_chunks,
            )
        ):
            logger.debug(f"Processing chunk {i + 1}/{stream_metadata.total_num_chunks}")
            chunk_start = i * stream_metadata.frames_per_chunk
            with elapsed_timer("append_batch"):
                await append_batch(zarr_array, frames, chunk_start)
            logger.debug(f"Appended chunk {i + 1} to Zarr array")
        logger.info(
            f"Processed {stream_metadata.total_num_chunks} chunks in {timer.elapsed:.2f} seconds"
        )
        return stream_metadata, zarr_array
