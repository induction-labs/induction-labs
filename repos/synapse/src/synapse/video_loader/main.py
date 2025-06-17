from __future__ import annotations

import asyncio
from functools import wraps

import typer
from synapse.elapsed_timer import elapsed_timer


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
    from .types import resolution_480p

    video_args = VideoProcessArgs(
        # video_path="gs://induction-labs/jeffrey/test_data/test_video.mp4",
        video_path="gs://induction-labs/youtube/-_-4RGKdhsc/output.mp4",
        max_frame_pixels=resolution_480p.pixels(),
        output_fps=2.0,
        output_path="jeffrey/test_vid6.zarr",
        frames_per_chunk=128,
    )
    with elapsed_timer("main") as timer:
        await process_video(video_args)
    timer.print_timing_tree()


@app.async_command()
async def run(
    # config_path: str = typer.Argument(
    #     ..., help="Path to experiment configuration toml file"
    # ),
    # extra_args: str = typer.Option("", help="Additional arguments for the module"),
):
    pass


if __name__ == "__main__":
    app()
