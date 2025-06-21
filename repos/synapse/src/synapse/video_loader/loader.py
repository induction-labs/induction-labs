from __future__ import annotations

import tempfile

import tensorstore as ts
from tqdm import tqdm

from synapse.elapsed_timer import elapsed_timer
from synapse.qwen_omni_utils.video import StreamVideoArgs, stream_video_to_tensors
from synapse.utils.logging import configure_logging

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
    logger.debug("Processing video with args: %r", args)
    with elapsed_timer("process_video") as timer:
        # root = get_zarr_root(fs, args.output_path)
        stream_args = StreamVideoArgs(
            output_fps=args.output_fps,
            video_path=args.video_path,
            max_pixels=args.max_frame_pixels,
            frames_per_chunk=args.frames_per_chunk,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            stream_metadata, video_frames = stream_video_to_tensors(
                stream_args, tmp.name
            )

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
                    metadata={
                        "stream": stream_metadata.model_dump(),
                    },
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
                logger.debug(
                    "Processing chunk %d/%d", i + 1, stream_metadata.total_num_chunks
                )
                chunk_start = i * stream_metadata.frames_per_chunk
                with elapsed_timer("append_batch"):
                    await append_batch(zarr_array, frames, chunk_start)
                logger.debug("Appended chunk %d to Zarr array", i + 1)
            logger.info(
                "Processed %d chunks in %.2f seconds",
                stream_metadata.total_num_chunks,
                timer.elapsed,
            )
            return stream_metadata, zarr_array
