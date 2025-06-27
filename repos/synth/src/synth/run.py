from __future__ import annotations

import asyncio
import random
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts
import torch
from matplotlib import cm
from PIL import Image, ImageDraw
from synapse import Cubic
from synapse.video_loader.typess import (
    FramesMetadata,
    StreamMetadata,
    VideoMetadata,
    VideoResolution,
)
from synapse.video_loader.video import smart_resize
from synapse.video_loader.zarr_utils import (
    ZarrArrayAttributes,
    append_batch,
    create_zarr_array,
)
from tqdm.asyncio import tqdm_asyncio

OVERLAY = Image.open(
    Path(__file__).parent / ".." / ".." / "assets" / "default.png"
).convert("RGBA")


def sample_cubics(n: int, delta: float) -> tuple[float, list[Cubic]]:
    num_cubics = n
    start_position = random.uniform(0, 1)
    current_position = start_position
    current_cubics = []
    for _ in range(num_cubics):
        permissible_range = [-current_position, 1 - current_position]

        def generate_cubic():
            return Cubic(
                m=random.uniform(-delta, delta),
                n=random.uniform(-delta, delta),
                a=random.uniform(*permissible_range),  # noqa: B023
            )

        cubic = generate_cubic()
        results = cubic(np.linspace(0, 1, 10))
        min_results = np.min(results)
        max_results = np.max(results)

        while min_results + current_position < 0 or max_results + current_position > 1:
            cubic = generate_cubic()
            results = cubic(np.linspace(0, 1, 10))
            min_results = np.min(results)
            max_results = np.max(results)

        current_position += cubic(1)
        current_cubics.append(cubic)

    return start_position, current_cubics


def cubics_to_points(
    x_start: float,
    y_start: float,
    x_cubics: list[Cubic],
    y_cubics: list[Cubic],
    fps=2,
    x_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_points is None:
        x_points = np.linspace(0, 1, 400)

    previous_last = (x_start, y_start)

    all_poly_x = []
    all_poly_y = []

    for xc, yc in zip(x_cubics, y_cubics, strict=False):
        x_dense = x_points
        y_cubic_of_xcoord = xc(x_dense)
        y_cubic_of_ycoord = yc(x_dense)

        all_poly_x.extend(y_cubic_of_xcoord + previous_last[0])
        all_poly_y.extend(y_cubic_of_ycoord + previous_last[1])

        previous_last = (xc(1) + previous_last[0], yc(1) + previous_last[1])

    # t = list(range(len(all_poly_x)))
    t = np.arange(0, len(all_poly_x) / fps, 1 / fps)

    return t, np.array(all_poly_x), np.array(all_poly_y)


def generate_image_from_segments(
    t: np.ndarray, x_norm: np.ndarray, y_norm: np.ndarray, screen_size: tuple[int, int]
) -> Image.Image:
    x = x_norm * screen_size[0]
    y = y_norm * screen_size[1]
    norm = plt.Normalize(t.min(), t.max())
    colors = cm.viridis(norm(t))

    # pick a colormap and normalize t to [0,1]
    norm = (t - t.min()) / (t.max() - t.min())
    colors = (cm.viridis(norm)[:, :3] * 255).astype(np.uint8)

    # make a blank RGBA image
    scale_factor = 8
    upsampled = (int(screen_size[0] * scale_factor), int(screen_size[1] * scale_factor))
    img = Image.new("RGBA", upsampled, (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    for i in range(len(x) - 1):
        p0 = (x[i] * scale_factor, y[i] * scale_factor)
        p1 = (x[i + 1] * scale_factor, y[i + 1] * scale_factor)
        draw.line([p0, p1], fill=tuple(colors[i]), width=int(4.3 * scale_factor))

    img_small = img.resize(screen_size, Image.LANCZOS)
    return img_small


def create_video_array(
    x_start: float, y_start: float, x_cubics: list[Cubic], y_cubics: list[Cubic]
):
    ts, all_poly_x, all_poly_y = cubics_to_points(x_start, y_start, x_cubics, y_cubics)
    base_image = generate_image_from_segments(ts, all_poly_x, all_poly_y, SCREEN_SIZE)

    ts, all_poly_x, all_poly_y = cubics_to_points(
        x_start, y_start, x_cubics, y_cubics, fps=4, x_points=np.array([0, 0.5])
    )
    imgs = []
    timestamps = []
    for time, x, y in zip(ts, all_poly_x, all_poly_y, strict=False):
        new_img = base_image.copy()
        new_img.paste(
            OVERLAY, (int(x * SCREEN_SIZE[0]), int(y * SCREEN_SIZE[1])), OVERLAY
        )
        imgs.append(new_img)
        timestamps.append(time)

    img_array = np.array([imgs[0], *imgs])[:, :, :, :3]
    time_array = np.array(timestamps)

    return time_array, img_array, x_cubics, y_cubics


SCREEN_SIZE = smart_resize(854, 480, factor=28, min_pixels=0, max_pixels=854 * 480)


def create_video(
    n_segments: int, delta: float
) -> tuple[np.ndarray, np.ndarray, list[Cubic], list[Cubic]]:
    (x_start, x_cubics), (y_start, y_cubics) = (
        sample_cubics(n_segments, delta),
        sample_cubics(n_segments, delta),
    )
    time, imgs, x_cubics, y_cubics = create_video_array(
        x_start, y_start, x_cubics, y_cubics
    )

    return time, imgs, x_cubics, y_cubics


async def upload_sample(
    time: np.ndarray,
    imgs: np.ndarray,
    x_cubics: list[Cubic],
    y_cubics: list[Cubic],
    output_path: str,
):
    output_meta = FramesMetadata(
        fps=Fraction(4),
        total_frames=imgs.shape[0],
        resolution=VideoResolution(
            width=SCREEN_SIZE[0],
            height=SCREEN_SIZE[1],
        ),
    )
    input_meta = VideoMetadata(
        start_pts=0,
        duration=imgs.shape[0],
        time_base=Fraction(1, 4),
        **output_meta.model_dump(),
    )
    stream_metadata = StreamMetadata(
        input_video=input_meta,
        output_video=output_meta,
        output_frames_per_chunk=imgs.shape[0],
    )

    shape = (
        output_meta.total_frames,
        3,
        output_meta.resolution.height,
        output_meta.resolution.width,
    )
    zarr_array = await create_zarr_array(
        ZarrArrayAttributes(
            chunk_shape=shape,
            shape=shape,
            dtype=ts.uint8,
            path=output_path,
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
            path=output_path + "/timestamps",
        ),
    )

    assert len(x_cubics) == len(y_cubics), (
        "x_cubics and y_cubics must have the same length"
    )

    cursor_actions_array = await create_zarr_array(
        ZarrArrayAttributes(
            chunk_shape=(len(x_cubics), 2, 3),
            shape=(
                len(x_cubics),
                2,
                3,
            ),
            dtype=ts.float32,
            path=output_path + "/cursor_action",
            metadata={
                "frames_per_action_step": 2,
            },
        ),
    )

    cursor_actions = torch.Tensor(
        np.array(
            [
                [x.to_ndarray(), y.to_ndarray()]
                for x, y in zip(x_cubics, y_cubics, strict=False)
            ]
        )
    )

    imgs_zarr_trans = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    await asyncio.gather(
        *[
            append_batch(zarr_array, imgs_zarr_trans, 0),
            append_batch(
                timestamps_array, torch.arange(imgs.shape[0]).to(torch.uint64), 0
            ),
            append_batch(cursor_actions_array, cursor_actions, 0),
        ]
    )


async def create_sample(sem: asyncio.Semaphore, i: int, path_template: str):
    async with sem:
        segments = random.randrange(5, 12)
        delta = random.uniform(0.1, 1)
        await upload_sample(
            *create_video(n_segments=segments, delta=delta), path_template.format(i=i)
        )


async def create_samples(num_samples: int, path_template: str):
    sem = asyncio.Semaphore(10)
    await tqdm_asyncio.gather(
        *[create_sample(sem, i, path_template) for i in range(num_samples)]
    )


if __name__ == "__main__":
    asyncio.run(
        create_samples(
            1000,
            "gs://induction-labs/jonathan/synth/cursor_follow_v0/sample_{i}.zarr",
        )
    )
