from __future__ import annotations

import asyncio
import io
import math
import random
from fractions import Fraction
from functools import partial
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np
import tensorstore as ts
import torch
from matplotlib import cm
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

try:
    import cv2

    HAS_SCIPY_SKIMAGE = True
except ImportError:
    HAS_SCIPY_SKIMAGE = False
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
from tqdm.auto import tqdm  # one bar in the main process

OVERLAY = Image.open(
    Path(__file__).parent / ".." / ".." / "assets" / "default.png"
).convert("RGBA")


def sample_cubics_with_jitter(n: int, delta: float) -> tuple[float, list[Cubic]]:
    """Sample cubic splines with control-point jitter for more natural ink-like strokes"""
    num_cubics = n
    start_position = random.uniform(0, 1)
    current_position = start_position
    current_cubics = []

    # Add tremor/jitter parameters
    jitter_strength = random.uniform(0.005, 0.02)  # Small jitter for natural look
    tremor_frequency = random.uniform(0.1, 0.3)  # Low frequency for hand tremor

    for i in range(num_cubics):
        permissible_range = [-current_position, 1 - current_position]

        def generate_cubic(i):
            # Base cubic parameters
            m = random.uniform(-delta, delta)
            n = random.uniform(-delta, delta)
            a = random.uniform(*permissible_range)  # noqa: B023

            # Add control-point jitter (tremor simulation)
            tremor_phase = i * tremor_frequency * 2 * math.pi
            jitter_m = jitter_strength * math.sin(
                tremor_phase + random.uniform(0, 2 * math.pi)
            )
            jitter_n = jitter_strength * math.cos(
                tremor_phase + random.uniform(0, 2 * math.pi)
            )

            # Apply Gaussian smoothing to avoid noisy corners
            jitter_m *= random.gauss(1.0, 0.2)
            jitter_n *= random.gauss(1.0, 0.2)

            return Cubic(
                m=m + jitter_m,
                n=n + jitter_n,
                a=a,
            )

        cubic = generate_cubic(i)
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


# Keep the original function for backward compatibility
def sample_cubics(n: int, delta: float) -> tuple[float, list[Cubic]]:
    return sample_cubics_with_jitter(n, delta)


def create_paper_texture(size: tuple[int, int]) -> np.ndarray:
    """Generate a simple paper texture pattern"""
    height, width = size

    # Create base texture using multiple noise layers
    texture = np.random.uniform(0.95, 1.05, (height, width))

    # Add paper fibers (vertical streaks)
    for _ in range(random.randint(20, 50)):
        x = random.randint(0, width - 1)
        thickness = random.randint(1, 3)
        intensity = random.uniform(0.92, 1.08)
        for t in range(thickness):
            if x + t < width:
                texture[:, x + t] *= intensity

    # Add horizontal texture (less pronounced)
    for _ in range(random.randint(10, 30)):
        y = random.randint(0, height - 1)
        thickness = random.randint(1, 2)
        intensity = random.uniform(0.96, 1.04)
        for t in range(thickness):
            if y + t < height:
                texture[y + t, :] *= intensity

    return np.clip(texture, 0.9, 1.1)


def apply_directional_blur(
    img_array: np.ndarray, angle: float, length: int
) -> np.ndarray:
    """Apply directional blur to simulate scanner carriage vibration"""
    if not HAS_SCIPY_SKIMAGE:
        return img_array

    # Create directional kernel
    kernel = np.zeros((length, length))
    center = length // 2

    # Create line kernel at specified angle
    for i in range(length):
        x = int(center + (i - center) * math.cos(math.radians(angle)))
        y = int(center + (i - center) * math.sin(math.radians(angle)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0

    kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel

    # Apply to each channel
    result = np.zeros_like(img_array)
    for c in range(img_array.shape[2]):
        result[:, :, c] = cv2.filter2D(img_array[:, :, c], -1, kernel)

    return result


def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression artifacts"""
    # Save to bytes buffer with specified quality
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Load back from compressed data
    compressed_img = Image.open(buffer)
    return compressed_img.convert("RGB")


def apply_data_augmentations(img: Image.Image) -> Image.Image:
    """Apply various data augmentations to simulate ink signatures and scanning artifacts"""
    # Convert to numpy for some operations
    img_array = np.array(img)
    height, width = img_array.shape[:2]

    # Apply augmentations with random probability
    augmented_img = img.copy()

    # === PAPER TEXTURE EFFECTS ===

    # Paper texture overlay (60% chance)
    if random.random() < 0.6:
        texture = create_paper_texture((height, width))
        img_array = np.array(augmented_img).astype(np.float32)

        # Apply texture using multiply blend mode
        for c in range(img_array.shape[2]):
            img_array[:, :, c] *= texture

        augmented_img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    # Varying background tint (40% chance)
    if random.random() < 0.4:
        # Create subtle background tint
        tint_color = random.choice(
            [
                (255, 252, 240),  # Warm white
                (252, 248, 245),  # Cream
                (248, 245, 240),  # Light beige
                (250, 250, 245),  # Cool white
            ]
        )

        # Create tint layer
        tint_img = Image.new("RGB", augmented_img.size, tint_color)
        augmented_img = Image.blend(augmented_img.convert("RGB"), tint_img, alpha=0.1)

    # Edge vignette/shadow (35% chance)
    if random.random() < 0.35:
        img_array = np.array(augmented_img).astype(np.float32)

        # Create radial gradient for vignette
        center_x, center_y = width // 2, height // 2
        y_coords, x_coords = np.ogrid[:height, :width]

        # Calculate distance from center
        max_dist = math.sqrt(center_x**2 + center_y**2)
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

        # Create vignette mask
        vignette_strength = random.uniform(0.1, 0.3)
        vignette = 1.0 - (distances / max_dist) * vignette_strength
        vignette = np.clip(vignette, 0.7, 1.0)

        # Apply vignette
        for c in range(img_array.shape[2]):
            img_array[:, :, c] *= vignette

        augmented_img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    # === SCANNING ARTIFACTS ===

    # Sub-pixel mis-alignment blur (50% chance)
    if random.random() < 0.5:
        blur_sigma = random.uniform(0.4, 0.8)
        if HAS_SCIPY_SKIMAGE:
            img_array = np.array(augmented_img).astype(np.float32)
            for c in range(img_array.shape[2]):
                img_array[:, :, c] = cv2.GaussianBlur(
                    img_array[:, :, c], (0, 0), blur_sigma
                )
            augmented_img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        else:
            # Fallback to PIL blur
            augmented_img = augmented_img.filter(
                ImageFilter.GaussianBlur(radius=blur_sigma)
            )

    # Directional scanner blur (30% chance)
    if HAS_SCIPY_SKIMAGE and random.random() < 0.3:
        blur_angle = random.uniform(-2, 2)  # Small angle variation
        blur_length = random.randint(3, 7)
        img_array = apply_directional_blur(
            np.array(augmented_img), blur_angle, blur_length
        )
        augmented_img = Image.fromarray(img_array.astype(np.uint8))

    # Noise injection (40% chance)
    if random.random() < 0.4:
        noise_sigma = random.uniform(3, 8) / 255.0
        img_array = np.array(augmented_img).astype(np.float32) / 255.0

        # Add Gaussian noise
        noise = np.random.normal(0, noise_sigma, img_array.shape)
        noisy_array = img_array + noise

        augmented_img = Image.fromarray(
            np.clip(noisy_array * 255, 0, 255).astype(np.uint8)
        )

    # JPEG compression artifacts (35% chance)
    if random.random() < 0.35:
        quality = random.randint(25, 60)
        augmented_img = apply_jpeg_compression(augmented_img, quality)

    # Low-frequency intensity banding (25% chance)
    if random.random() < 0.25:
        img_array = np.array(augmented_img).astype(np.float32)

        # Create sine wave banding
        period = random.randint(80, 200)
        amplitude = random.uniform(0.02, 0.08)

        # Choose direction (horizontal or vertical)
        if random.random() < 0.5:
            # Horizontal banding
            banding = 1.0 + amplitude * np.sin(2 * np.pi * np.arange(height) / period)
            banding = banding.reshape(-1, 1, 1)
        else:
            # Vertical banding
            banding = 1.0 + amplitude * np.sin(2 * np.pi * np.arange(width) / period)
            banding = banding.reshape(1, -1, 1)

        img_array *= banding
        augmented_img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    # Moiré/scanner pattern (15% chance)
    if random.random() < 0.15:
        img_array = np.array(augmented_img).astype(np.float32)

        # Create checkerboard pattern
        pattern_size = random.randint(2, 4)
        pattern_intensity = random.uniform(0.005, 0.01)

        # Create pattern
        y_coords, x_coords = np.ogrid[:height, :width]
        pattern = np.sin(2 * np.pi * x_coords / pattern_size) * np.sin(
            2 * np.pi * y_coords / pattern_size
        )
        pattern = pattern_intensity * pattern

        # Apply pattern
        for c in range(img_array.shape[2]):
            img_array[:, :, c] *= 1 + pattern

        augmented_img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    # === TRADITIONAL AUGMENTATIONS (reduced probabilities) ===

    # Brightness adjustment (20% chance)
    if random.random() < 0.2:
        brightness_factor = random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Brightness(augmented_img)
        augmented_img = enhancer.enhance(brightness_factor)

    # Contrast adjustment (25% chance)
    if random.random() < 0.25:
        contrast_factor = random.uniform(0.9, 1.2)
        enhancer = ImageEnhance.Contrast(augmented_img)
        augmented_img = enhancer.enhance(contrast_factor)

    # Slight sharpness adjustment (15% chance)
    if random.random() < 0.15:
        sharpness_factor = random.uniform(0.8, 1.3)
        enhancer = ImageEnhance.Sharpness(augmented_img)
        augmented_img = enhancer.enhance(sharpness_factor)

    return augmented_img


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


def generate_pressure_variation(t: np.ndarray, base_width: float) -> np.ndarray:
    """Generate pressure-based width modulation using smooth noise"""
    # Create smooth pressure variation using multiple sine waves
    pressure = np.ones_like(t)

    # Add multiple frequency components for natural pressure variation
    frequencies = [0.1, 0.3, 0.7, 1.2]  # Different frequency components
    amplitudes = [0.4, 0.2, 0.1, 0.05]  # Decreasing amplitudes

    for freq, amp in zip(frequencies, amplitudes, strict=False):
        phase = random.uniform(0, 2 * math.pi)
        pressure += amp * np.sin(freq * t * 2 * math.pi + phase)

    # Add some random noise
    noise_strength = 0.05
    pressure += noise_strength * np.random.normal(0, 1, len(t))

    # Normalize and apply to width
    pressure = np.clip(pressure, 0.3, 2.0)  # Limit pressure range
    return pressure * base_width


def add_path_noise(
    x: np.ndarray, y: np.ndarray, noise_strength: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Add small noise to the path before rasterization for more natural ink look"""
    # Add high-frequency noise for ink texture
    noise_x = noise_strength * np.random.normal(0, 1, len(x))
    noise_y = noise_strength * np.random.normal(0, 1, len(y))

    # Smooth the noise to avoid sharp corners
    if len(noise_x) > 3:
        # Simple moving average smoothing
        window = 3
        noise_x = np.convolve(noise_x, np.ones(window) / window, mode="same")
        noise_y = np.convolve(noise_y, np.ones(window) / window, mode="same")

    return x + noise_x, y + noise_y


def generate_image_from_segments(
    t: np.ndarray, x_norm: np.ndarray, y_norm: np.ndarray, screen_size: tuple[int, int]
) -> Image.Image:
    x = x_norm * screen_size[0]
    y = y_norm * screen_size[1]

    # Add path noise for more natural ink appearance
    noise_strength = random.uniform(0.2, 1.0)
    x, y = add_path_noise(x, y, noise_strength)

    # Variable background colors (favor white for ink-like appearance)
    bg_color = (
        (255, 255, 255, 255) if random.random() < 0.9 else (250, 248, 245, 255)
    )  # Off-white paper

    # Base stroke width for pressure variation
    stroke_width_base = random.uniform(1.5, 4.0)  # Thinner for ink-like appearance

    # Generate pressure-based width variation
    pressure_widths = generate_pressure_variation(t, stroke_width_base)

    # Ink-like colors (mostly black/dark blue)
    ink_colors = ["solid_black", "solid_dark_blue", "solid_dark_gray", "solid_brown"]
    line_color_mode = random.choice(
        [*ink_colors, "viridis", "plasma"]
    )  # Mostly ink colors

    if line_color_mode == "solid_black":
        colors = np.full((len(t), 3), [0, 0, 0], dtype=np.uint8)
    elif line_color_mode == "solid_dark_blue":
        colors = np.full((len(t), 3), [0, 0, 80], dtype=np.uint8)
    elif line_color_mode == "solid_dark_gray":
        colors = np.full((len(t), 3), [40, 40, 40], dtype=np.uint8)
    elif line_color_mode == "solid_brown":
        colors = np.full((len(t), 3), [60, 30, 0], dtype=np.uint8)
    else:
        # Use matplotlib colormaps for variety
        norm = (t - t.min()) / (t.max() - t.min())
        if line_color_mode == "viridis":
            colors = (cm.viridis(norm)[:, :3] * 255).astype(np.uint8)
        elif line_color_mode == "plasma":
            colors = (cm.plasma(norm)[:, :3] * 255).astype(np.uint8)

    # make a blank RGBA image
    scale_factor = 8
    upsampled = (int(screen_size[0] * scale_factor), int(screen_size[1] * scale_factor))
    img = Image.new("RGBA", upsampled, bg_color)
    draw = ImageDraw.Draw(img)

    # Draw with pressure-based width variation
    for i in range(len(x) - 1):
        p0 = (x[i] * scale_factor, y[i] * scale_factor)
        p1 = (x[i + 1] * scale_factor, y[i + 1] * scale_factor)

        # Use pressure-based width
        stroke_width = max(1, int(pressure_widths[i] * scale_factor))

        # Add slight color variation for ink bleeding effect
        color = colors[i].copy()
        if random.random() < 0.3:  # 30% chance for slight color variation
            color = color.astype(np.int16)
            variation = random.randint(-10, 10)
            color = np.clip(color + variation, 0, 255).astype(np.uint8)

        draw.line([p0, p1], fill=tuple(color), width=stroke_width)

    img_small = img.resize(screen_size, Image.LANCZOS)
    return img_small


def create_video_array(
    x_start: float,
    y_start: float,
    x_cubics: list[Cubic],
    y_cubics: list[Cubic],
    garbage: bool = False,
):
    ts, all_poly_x, all_poly_y = cubics_to_points(x_start, y_start, x_cubics, y_cubics)
    base_image = generate_image_from_segments(ts, all_poly_x, all_poly_y, SCREEN_SIZE)

    # Apply data augmentations to the base image
    base_image = apply_data_augmentations(base_image)

    ts, all_poly_x, all_poly_y = cubics_to_points(
        x_start, y_start, x_cubics, y_cubics, fps=4, x_points=np.array([0, 0.5])
    )
    imgs = []
    timestamps = []
    for time, _x, _y in zip(ts, all_poly_x, all_poly_y, strict=False):
        new_img = base_image.copy()
        # Cursor overlay removed
        imgs.append(new_img)
        timestamps.append(time)

    img_array = np.array([imgs[0], *imgs])[:, :, :, :3]
    time_array = np.array(timestamps)

    if garbage:
        img_array = np.zeros_like(img_array)
        # actually generate just noise
        noise = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)
        img_array = noise

    return time_array, img_array, x_cubics, y_cubics


SCREEN_SIZE = smart_resize(854, 480, factor=28, min_pixels=0, max_pixels=854 * 480)


def create_video(
    n_segments: int, delta: float, garbage: bool = False
) -> tuple[np.ndarray, np.ndarray, list[Cubic], list[Cubic]]:
    (x_start, x_cubics), (y_start, y_cubics) = (
        sample_cubics(n_segments, delta),
        sample_cubics(n_segments, delta),
    )

    time, imgs, x_cubics, y_cubics = create_video_array(
        x_start, y_start, x_cubics, y_cubics, garbage=garbage
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


async def create_sample(
    sem: asyncio.Semaphore, i: int, path_template: str, garbage: bool = False
):
    async with sem:
        segments = random.randrange(5, 12)
        delta = random.uniform(0.1, 1)
        await upload_sample(
            *create_video(n_segments=segments, delta=delta, garbage=garbage),
            path_template.format(i=i),
        )


async def _batch(start: int, tpl: str, garbage: bool = False, width: int = 4):
    sem = asyncio.Semaphore(width)  # 4 concurrent tasks *inside* one process
    await asyncio.gather(
        *(
            create_sample(sem, i, tpl, garbage=garbage)
            for i in range(start, start + width)
        )
    )


def worker(start: int, tpl: str, garbage: bool = False, width: int = 4):
    asyncio.run(
        _batch(start, tpl, garbage=garbage, width=width)
    )  # each process owns its own loop


# ── driver ───────────────────────────────────────────────────────────
def run_all(
    total: int, tpl: str, n_proc: int = 6, garbage: bool = False, width: int = 4
):
    starts = range(0, total, width)  # one “batch” every <width> indices
    with Pool(n_proc) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(worker, tpl=tpl, garbage=garbage, width=width), starts
            ),
            total=len(starts),
        ):
            pass


# ── entry-point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import contextlib

    with contextlib.suppress(
        RuntimeError
    ):  # makes the script work on Windows & macOS ≥3.8 too
        set_start_method("spawn")

    run_all(
        15000,
        "gs://induction-labs/jonathan/synth/cursor_follow_augmented/sample_{i}.zarr",
        n_proc=12,
        garbage=False,
        width=2,
    )
