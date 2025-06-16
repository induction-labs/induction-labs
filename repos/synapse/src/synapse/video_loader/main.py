from __future__ import annotations

import asyncio
from functools import wraps

import typer


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
async def run(
    # config_path: str = typer.Argument(
    #     ..., help="Path to experiment configuration toml file"
    # ),
    # extra_args: str = typer.Option("", help="Additional arguments for the module"),
):
    from .loader import VideoProcessArgs, process_video
    from .types import resolution_480p

    video_args = VideoProcessArgs(
        video_path="gs://induction-labs/jeffrey/test_data/test_video.mp4",
        max_frame_pixels=resolution_480p.pixels(),
        output_fps=2.0,
        output_path="jeffrey/test_vid5.zarr",
        frames_per_chunk=32,
    )

    await process_video(video_args)


@app.async_command()
async def run2(
    # config_path: str = typer.Argument(
    #     ..., help="Path to experiment configuration toml file"
    # ),
    # extra_args: str = typer.Option("", help="Additional arguments for the module"),
):
    from .loader import VideoProcessArgs, process_video
    from .types import resolution_480p

    video_args = VideoProcessArgs(
        video_path="path/to/video.mp4",
        max_frame_pixels=resolution_480p.pixels(),
        output_fps=2.0,
        output_path="induction-labs/jeffrey/test_vid5.zarr",
        frames_per_chunk=32,
    )

    await process_video(video_args)


if __name__ == "__main__":
    app()
