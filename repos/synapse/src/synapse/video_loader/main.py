from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from fractions import Fraction
from functools import wraps
from pathlib import Path

import gcsfs
import typer

from synapse.elapsed_timer import elapsed_timer
from synapse.redis.main import (
    GCSVideoQueue,
    get_redis_client,
    get_video_next_unprocessed,
    put_video_in_queue,
)


class AsyncTyper(typer.Typer):
    def async_command(self, *args, **kwargs):
        def decorator(async_func):
            @wraps(async_func)
            def sync_func(*_args, **_kwargs):
                return asyncio.run(async_func(*_args, **_kwargs))

            self.command(*args, **kwargs)(sync_func)
            return async_func

        return decorator


app = AsyncTyper()


@app.async_command()
async def test():
    from .loader import VideoProcessArgs, process_video
    from .typess import resolution_480p

    output_processor = "gs://induction-labs/jeffrey/test_video.zarr"
    # First remove the output file if it exists
    fs = gcsfs.GCSFileSystem()
    if fs.exists(output_processor):
        print(f"Removing existing output file: {output_processor}")
        fs.rm(output_processor, recursive=True)

    video_args = VideoProcessArgs(
        # video_path="gs://induction-labs/jeffrey/test_data/test_video.mp4",
        video_path="gs://induction-labs/youtube/CGvIVbISOxY/output.mp4",  # short video
        # video_path="gs://induction-labs/youtube/X5T3gN09oEg/output.mp4",  # Long one
        max_frame_pixels=resolution_480p.pixels(),
        output_fps=Fraction(4),
        output_path=output_processor,
        frames_per_chunk=128,
    )
    with elapsed_timer("main") as timer:
        await process_video(video_args)
    timer.print_timing_tree()


INPUT_PREFIX = Path("induction-labs/youtube/")
OUTPUT_PREFIX = Path("induction-labs/youtube-output-2/")
fs = gcsfs.GCSFileSystem()


def check_input_path_exists(input_path: Path) -> bool:
    """
    Check if the input path exists in Google Cloud Storage.
    """
    return fs.exists("gs://" + input_path.as_posix())


@app.async_command()
async def run(
    # config_path: str = typer.Argument(
    #     ..., help="Path to experiment configuration toml file"
    # ),
    # extra_args: str = typer.Option("", help="Additional arguments for the module"),
):
    r = get_redis_client()
    while True:
        video_data = get_video_next_unprocessed(r)
        if video_data is None:
            print("No more videos to process.")
            break
        try:
            print(f"Processing video: {video_data.video_id}")
            from .loader import VideoProcessArgs, process_video
            from .typess import resolution_480p

            input_path = INPUT_PREFIX / video_data.video_id / "output.mp4"
            output_path = OUTPUT_PREFIX / f"{video_data.video_id}.zarr"
            # check if the input video exists
            if not check_input_path_exists(input_path):
                print(f"Input video {input_path} does not exist in GCS, skipping.")
                put_video_in_queue(r, video_data, GCSVideoQueue.ERROR)
                continue
            if check_input_path_exists(output_path):
                print(f"Output video {output_path} already exists, skipping.")
                put_video_in_queue(r, video_data, GCSVideoQueue.ERROR)
                continue
            video_data.time_started_processing = datetime.now(UTC)
            # Wait idk why i used a queue here thats kinda dumb
            put_video_in_queue(r, video_data, GCSVideoQueue.IN_PROGRESS)

            video_args = VideoProcessArgs(
                video_path="gs://" + input_path.as_posix(),
                max_frame_pixels=resolution_480p.pixels(),
                output_fps=Fraction(4),
                output_path="gs://" + output_path.as_posix(),
                frames_per_chunk=128,
            )
            with elapsed_timer("main") as timer:
                await process_video(video_args)
            video_data.time_finished_processing = datetime.now(UTC)
            put_video_in_queue(r, video_data, GCSVideoQueue.DONE)
            timer.print_timing_tree()

        except Exception as e:
            print(f"Error processing video {video_data.video_id}: {e}")
            # Optionally, you can log the error or update the video status in Redis
            # For example, you could push the video back to a different queue for retrying
            video_data.time_finished_processing = datetime.now(UTC)
            video_data.time_started_processing = None
            put_video_in_queue(r, video_data, GCSVideoQueue.ERROR)
            continue
            # raise e


if __name__ == "__main__":
    app()
