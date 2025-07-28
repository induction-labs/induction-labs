from __future__ import annotations

import asyncio
import json
import random
from fractions import Fraction
from functools import partial
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np
import tensorstore as ts
import torch
from google.cloud import storage
from PIL import Image, ImageDraw, ImageFont
from synapse import Cubic
from synapse.actions.keyboard_press import keys_to_tokens
from synapse.actions.keyboard_tokenizer import Tokenizer
from synapse.actions.models import Action
from synapse.actions.mouse_movements import (
    cubics_to_points,
    generate_image_from_segments,
)
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
from tqdm.auto import tqdm

# QWERTY proximity map for realistic inter-key timing
_POS = {
    **{c.lower(): (i, 0) for i, c in enumerate("QWERTYUIOP")},
    **{c.lower(): (i, 1) for i, c in enumerate("ASDFGHJKL")},
    **{c.lower(): (i, 2) for i, c in enumerate("ZXCVBNM")},
}
_MAX_DIST = (9**2 + 2**2) ** 0.5


def _key_dist(a, b):
    a, b = a.lower(), b.lower()
    if a in _POS and b in _POS:
        dx = _POS[a][0] - _POS[b][0]
        dy = _POS[a][1] - _POS[b][1]
        return ((dx * dx + dy * dy) ** 0.5) / _MAX_DIST
    return 1.0


def _map_key(ch):
    return "space" if ch == " " else ch


# symbols that require shift and their unshifted equivalents
_SHIFT_KEY_MAP = {
    "!": "1",
    "@": "2",
    "#": "3",
    "$": "4",
    "%": "5",
    "^": "6",
    "&": "7",
    "*": "8",
    "(": "9",
    ")": "0",
    "_": "-",
    "+": "=",
    "{": "[",
    "}": "]",
    "|": "\\",
    ":": ";",
    '"': "'",
    "<": ",",
    ">": ".",
    "?": "/",
}

OVERLAY = Image.open(
    Path(__file__).parent / ".." / ".." / "assets" / "default.png"
).convert("RGBA")

SCREEN_SIZE = smart_resize(854, 480, factor=28, min_pixels=0, max_pixels=854 * 480)

tokenizer = Tokenizer.load("gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json")


def sample_cubics(n: int, delta: float) -> tuple[float, list[Cubic]]:
    num_cubics = n
    start_position = random.uniform(0, 1)
    current_position = start_position
    current_cubics = []
    for _ in range(num_cubics):
        permissible_range = [-current_position, 1 - current_position]

        def generate_cubic(perm_range=permissible_range):
            return Cubic(
                m=random.uniform(-delta, delta),
                n=random.uniform(-delta, delta),
                a=random.uniform(perm_range[0], perm_range[1]),
            )

        cubic = generate_cubic()
        test_vals = np.linspace(0, 1, 10)
        results = np.array([cubic(val) for val in test_vals])
        min_results = np.min(results)
        max_results = np.max(results)

        while min_results + current_position < 0 or max_results + current_position > 1:
            cubic = generate_cubic()
            results = np.array([cubic(val) for val in test_vals])
            min_results = np.min(results)
            max_results = np.max(results)

        current_position += float(cubic(1))
        current_cubics.append(cubic)

    return start_position, current_cubics


def generate_typing_video_with_mouse_paths(
    text: str,
    fps: int = 4,
    speed: float = 3.0,
    frame_size=(480, 854),
    font_path=None,
    font_size=24,
    random_position: bool = True,
    seed=None,
    n_mouse_segments: int | None = None,
    mouse_delta: float | None = None,
    # timing parameters (mu, sigma for log-normal)
    hold_mu=-2.0,
    hold_sigma=0.5,
    space_hold_mu=-2.5,
    space_hold_sigma=0.4,
    gap_mu=-1.5,
    gap_sigma=0.6,
    post_space_gap=2.0,
    min_gap_factor=0.5,
):
    """
    Generate a video with keyboard typing and mouse movement paths overlaid.

    Returns:
      video: np.ndarray of shape (n_frames, H, W, 3)
      key_logs: list of {"action": {...}, "timestamp": ...} dicts
      frame_times: list of frame timestamps
      x_cubics: list of Cubic objects for x mouse movements
      y_cubics: list of Cubic objects for y mouse movements
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    H, W = frame_size
    font = (
        ImageFont.truetype(font_path, font_size)
        if font_path
        else ImageFont.load_default()
    )

    # Measure full-text bounding box
    tmp = Image.new("RGB", (W, H))
    dtmp = ImageDraw.Draw(tmp)
    bx = dtmp.textbbox((0, 0), text, font=font)
    text_w, text_h = bx[2] - bx[0], bx[3] - bx[1]

    # Choose random origin so text stays fully inside
    if random_position:
        x0 = random.randint(0, W - text_w)
        y0 = random.randint(0, H - text_h)
    else:
        x0, y0 = 10, 10

    # 1) Simulate keyboard press/release events
    raw_events = []
    release_times = []
    t = 0.0
    prev = None

    for ch in text:
        key_name = _map_key(ch)
        needs_shift = ch.isupper() or ch in _SHIFT_KEY_MAP

        if needs_shift:
            raw_events.append((t, "shift", True))
            raw_events.append((t, ch, True))
        else:
            raw_events.append((t, key_name, True))

        mu, sig = (
            (space_hold_mu, space_hold_sigma) if ch == " " else (hold_mu, hold_sigma)
        )
        raw = random.lognormvariate(mu, sig)
        hold = max(0.02, min(raw, 1.0)) / speed
        key_release = t + hold

        if needs_shift:
            if random.random() < 0.5:
                shift_rel = t + hold * 0.5
                raw_events.append((shift_rel, "shift", False))
                rel_key = ch.lower() if ch.isalpha() else _SHIFT_KEY_MAP[ch]
                raw_events.append((key_release, rel_key, False))
                release_times.append(key_release)
            else:
                raw_events.append((key_release - 0.005, key_name, False))
                raw_events.append((key_release, "shift", False))
                release_times.append(key_release)
        else:
            raw_events.append((key_release, key_name, False))
            release_times.append(key_release)

        t += hold

        if ch != text[-1]:
            base = random.lognormvariate(gap_mu, gap_sigma)
            if ch == " ":
                base *= post_space_gap
            base = max(0.03, min(base, 1.5)) / speed
            factor = min_gap_factor + (1 - min_gap_factor) * _key_dist(prev or ch, ch)
            t += base * factor

        prev = ch

    total_time = t

    # 2) Generate mouse movement data
    if n_mouse_segments is None:
        n_mouse_segments = random.randrange(5, 12)
    if mouse_delta is None:
        mouse_delta = random.uniform(0.1, 1)

    (x_start, x_cubics), (y_start, y_cubics) = (
        sample_cubics(n_mouse_segments, mouse_delta),
        sample_cubics(n_mouse_segments, mouse_delta),
    )

    # Generate mouse path image
    ts_path, all_poly_x_path, all_poly_y_path = cubics_to_points(
        x_start, y_start, x_cubics, y_cubics
    )
    mouse_path_image = generate_image_from_segments(
        all_poly_x_path, all_poly_y_path, (W, H)
    ).convert("RGB")

    # Generate cursor positions for each frame
    ts_cursor, all_poly_x_cursor, all_poly_y_cursor = cubics_to_points(
        x_start, y_start, x_cubics, y_cubics, fps=fps, x_points=np.array([0, 0.5])
    )

    # 3) Extend mouse timeline to match keyboard timeline if needed
    keyboard_frame_times = list(np.arange(0, total_time, 1 / fps))
    if keyboard_frame_times[-1] < total_time:
        keyboard_frame_times.append(total_time)

    # Extend cursor positions if keyboard is longer than mouse movement
    extended_ts_cursor = list(ts_cursor)
    extended_x_cursor = list(all_poly_x_cursor)
    extended_y_cursor = list(all_poly_y_cursor)

    if len(keyboard_frame_times) > len(ts_cursor):
        # Pad with stationary cursor at last position
        last_x = all_poly_x_cursor[-1] if len(all_poly_x_cursor) > 0 else 0.5
        last_y = all_poly_y_cursor[-1] if len(all_poly_y_cursor) > 0 else 0.5

        for i in range(len(ts_cursor), len(keyboard_frame_times)):
            extended_ts_cursor.append(keyboard_frame_times[i])
            extended_x_cursor.append(last_x)
            extended_y_cursor.append(last_y)

    # Use the extended timeline
    frame_times = extended_ts_cursor

    # 4) Generate frames like the original code
    cursor_frames = []
    for time, x, y in zip(
        extended_ts_cursor, extended_x_cursor, extended_y_cursor, strict=False
    ):
        img = mouse_path_image.copy()
        draw = ImageDraw.Draw(img)

        # Calculate how much text should be typed based on time
        typed = sum(1 for rt in release_times if rt <= time)

        # Draw text directly on the image - simple black text
        # Draw full text in light gray
        draw.text((x0, y0), text, font=font, fill=(150, 150, 150))

        # Draw typed portion in black
        if typed > 0:
            draw.text((x0, y0), text[:typed], font=font, fill=(0, 0, 0))

        # Add cursor at current position
        cursor_x = int(x * W)
        cursor_y = int(y * H)
        img.paste(OVERLAY, (cursor_x, cursor_y), OVERLAY)

        cursor_frames.append(img)

    # Convert to numpy arrays like original: [first_frame, *frames][:, :, :, :3]
    frames = [np.array(frame) for frame in cursor_frames]
    video_frames = np.array([frames[0], *frames])[:, :, :, :3]

    # Frame times with doubled first frame
    frame_times = [0.0, *frame_times]

    # 5) Format key logs
    key_logs = []
    for t, key, is_down in sorted(raw_events, key=lambda e: e[0]):
        key_logs.append(
            {
                "action": {"action": "key_button", "key": key, "is_down": is_down},
                "timestamp": float(t),
            }
        )

    # Pad mouse cubic actions with zero coefficients if needed
    number_of_actions = video_frames.shape[0] // 2
    if number_of_actions > len(x_cubics):
        # Add zero coefficient cubics for the extended timeline
        from synapse import Cubic

        zero_cubic = Cubic(m=0.0, n=0.0, a=0.0)
        padded_x_cubics = list(x_cubics)
        padded_y_cubics = list(y_cubics)

        padding_needed = number_of_actions - len(x_cubics)
        for _ in range(padding_needed):
            padded_x_cubics.append(zero_cubic)
            padded_y_cubics.append(zero_cubic)
    else:
        padded_x_cubics = x_cubics
        padded_y_cubics = y_cubics

    return video_frames, key_logs, frame_times, padded_x_cubics, padded_y_cubics


async def upload_sample(
    imgs: np.ndarray,
    actions: list[dict],
    output_path: str,
    frame_time: list[float],
    x_cubics: list[Cubic],
    y_cubics: list[Cubic],
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
            shape=(stream_metadata.output_video.total_frames,),
            dtype=ts.uint64,
            path=output_path + "/timestamps",
        ),
    )

    # Keyboard tokens
    MAX_TOKENS = 24
    tokens_shape = (stream_metadata.output_video.total_frames // 2, MAX_TOKENS)
    tokens_array = await create_zarr_array(
        ZarrArrayAttributes(
            chunk_shape=tokens_shape,
            shape=tokens_shape,
            dtype=ts.uint16,
            path=output_path + "/keyboard_tokens",
        ),
    )
    zarr_pad_mask_array = await create_zarr_array(
        ZarrArrayAttributes(
            chunk_shape=tokens_shape,
            shape=tokens_shape,
            dtype=ts.bool,
            path=output_path + "/keyboard_tokens_mask",
        ),
    )

    # Mouse cursor actions
    assert len(x_cubics) == len(y_cubics)
    cursor_actions_array = await create_zarr_array(
        ZarrArrayAttributes(
            chunk_shape=(len(x_cubics), 2, 3),
            shape=(len(x_cubics), 2, 3),
            dtype=ts.float32,
            path=output_path + "/cursor_action",
            metadata={
                "frames_per_action_step": 2,
            },
        ),
    )

    # Upload key actions as JSONL
    key_actions_jsonl_path = output_path + "/key_actions.jsonl"
    key_actions_jsonl = "\n".join(json.dumps(action) for action in actions)
    storage.Client().bucket(key_actions_jsonl_path[5:].split("/", 1)[0]).blob(
        key_actions_jsonl_path[5:].split("/", 1)[1]
    ).upload_from_string(key_actions_jsonl, content_type="application/json")

    # Process keyboard tokens
    tokens = [Action(**action) for action in actions]
    result = keys_to_tokens(
        tokens,
        frame_time[1::2],
        tokenizer,
        clock_tick_len=0.5 / MAX_TOKENS,
        time_per_segment=0.5,
        press_threshold=0.15,
    )

    def pad_tokens(tokens, max_len):
        return tokens + [tokenizer.mappings["[pad]"]] * (max_len - len(tokens))

    result_padded = [pad_tokens(line[0], MAX_TOKENS) for line in result]
    pad_mask = [
        [1] * len(line[0]) + [0] * (MAX_TOKENS - len(line[0])) for line in result
    ]
    tokenized_tensors = torch.tensor(result_padded, dtype=torch.uint16)
    pad_mask_array = torch.tensor(pad_mask, dtype=torch.bool)

    # Process cursor actions
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
            append_batch(tokens_array, tokenized_tensors, 0),
            append_batch(zarr_pad_mask_array, pad_mask_array, 0),
            append_batch(cursor_actions_array, cursor_actions, 0),
        ]
    )


async def generate_sample(sem: asyncio.Semaphore, text: str, output_path: str):
    async with sem:
        vid, keys, frame_time, x_cubics, y_cubics = (
            generate_typing_video_with_mouse_paths(
                text,
                fps=4,
                font_path="/home/jonathan_inductionlabs_com/induction-labs/repos/synth/assets/arial.ttf",
                speed=3,
                random_position=True,
                frame_size=(SCREEN_SIZE[1], SCREEN_SIZE[0]),
            )
        )
        await upload_sample(
            imgs=vid,
            actions=keys,
            output_path=output_path,
            frame_time=frame_time,
            x_cubics=x_cubics,
            y_cubics=y_cubics,
        )


with open("texts.txt") as f:
    TEXTS = f.read().splitlines()


async def _batch(start: int, tpl: str, width: int = 4):
    sem = asyncio.Semaphore(width)
    await asyncio.gather(
        *(
            generate_sample(sem, TEXTS[i], tpl.format(i=i))
            for i in range(start, start + width)
        )
    )


def worker(start: int, tpl: str, width: int = 4):
    asyncio.run(_batch(start, tpl, width))


def run_all(total: int, tpl: str, n_proc: int = 6, width: int = 4):
    starts = range(0, total, width)
    with Pool(n_proc) as pool:
        for _ in tqdm(
            pool.imap_unordered(partial(worker, tpl=tpl, width=width), starts),
            total=len(starts),
        ):
            pass


if __name__ == "__main__":
    import contextlib

    with contextlib.suppress(RuntimeError):
        set_start_method("spawn")

    run_all(
        len(TEXTS) // 4,
        "gs://induction-labs/jonathan/synth/typing_with_mouse_paths_v0/sample_{i}.zarr",
        n_proc=16,
        width=4,
    )
