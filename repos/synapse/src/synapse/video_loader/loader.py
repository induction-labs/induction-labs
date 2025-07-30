from __future__ import annotations

import tensorstore as ts
from tqdm import tqdm

from synapse.elapsed_timer.elapsed_timer import elapsed_timer
from synapse.utils.logging import configure_logging
from synapse.video_loader.video import (
    StreamVideoArgs,
    configure_video_folder_stream,
    configure_video_stream,
    process_stream_tensors,
)

from .typess import VideoProcessArgs
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
        configure_fn = (
            configure_video_folder_stream
            if args.video_path.endswith("/")
            else configure_video_stream
        )
        stream_metadata, video_format, stream_context = await configure_fn(
            stream_args,
        )

        # Create the Zarr array if it does not exist
        zarr_array = await create_zarr_array(
            ZarrArrayAttributes(
                chunk_shape=(
                    args.frames_per_chunk,
                    3,
                    stream_metadata.output_video.resolution.height,
                    stream_metadata.output_video.resolution.width,
                ),
                shape=(
                    stream_metadata.output_video.total_frames,
                    3,
                    stream_metadata.output_video.resolution.height,
                    stream_metadata.output_video.resolution.width,
                ),  # Start with 0 frames
                dtype=ts.uint8,
                path=args.output_path,
                metadata={
                    "stream": stream_metadata.model_dump(),
                },
            ),
        )
        timestamps_array = await create_zarr_array(
            ZarrArrayAttributes(
                chunk_shape=(stream_metadata.output_video.total_frames,),
                shape=(
                    stream_metadata.output_video.total_frames,
                ),  # Start with 0 timestamps
                dtype=ts.uint64,
                path=args.output_path + "/timestamps",
            ),
        )

        # Process and append frames to the Zarr array
        with stream_context() as stream_generator:
            for i, (frames, timestamps) in enumerate(
                tqdm(
                    process_stream_tensors(
                        stream_generator,
                        video_format,
                        stream_metadata,
                    ),
                    desc="Processing video frames",
                    total=stream_metadata.total_num_chunks,
                )
            ):
                logger.debug(
                    "Processing chunk %d/%d", i + 1, stream_metadata.total_num_chunks
                )
                chunk_start = i * stream_metadata.output_frames_per_chunk
                with elapsed_timer("append_batch"):
                    from asyncio import gather

                    await gather(
                        append_batch(zarr_array, frames, chunk_start),
                        append_batch(timestamps_array, timestamps, chunk_start),
                    )
                    del frames, timestamps
                    # await append_batch(timestamps_array, timestamps, chunk_start)
                    # await append_batch(zarr_array, frames, chunk_start)
                logger.debug("Appended chunk %d to Zarr array", i + 1)
            logger.info(
                "Processed %d chunks in %.2f seconds",
                stream_metadata.total_num_chunks,
                timer.elapsed,
            )
        return stream_metadata, zarr_array
