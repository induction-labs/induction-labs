from __future__ import annotations

import asyncio
import json
import random
from fractions import Fraction
from functools import partial
from multiprocessing import Pool, set_start_method

import numpy as np
import tensorstore as ts
import torch
from google.cloud import storage
from PIL import Image, ImageDraw, ImageFont
from synapse.actions.keyboard_press import keys_to_tokens
from synapse.actions.keyboard_tokenizer import Tokenizer
from synapse.actions.models import Action
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


# map space for logging
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


def generate_typing_video(
    text: str,
    fps: int = 10,
    speed: float = 3.0,
    frame_size=(480, 854),
    font_path=None,
    font_size=24,
    random_position: bool = True,
    seed=None,
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
    Returns:
      video: np.ndarray of shape (n_frames, H, W, 3)
      key_logs: list of {"action": {...}, "timestamp": ...} dicts
      frame_times: list of frame timestamps
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

    # 1) Simulate press/release events
    raw_events = []  # (time, key, is_down)
    release_times = []
    t = 0.0
    prev = None

    for ch in text:
        key_name = _map_key(ch)
        needs_shift = ch.isupper() or ch in _SHIFT_KEY_MAP

        # -- press phase --
        if needs_shift:
            raw_events.append((t, "shift", True))
            raw_events.append((t, ch, True))
        else:
            raw_events.append((t, key_name, True))

        # -- compute hold duration --
        mu, sig = (
            (space_hold_mu, space_hold_sigma) if ch == " " else (hold_mu, hold_sigma)
        )
        raw = random.lognormvariate(mu, sig)
        hold = max(0.02, min(raw, 1.0)) / speed
        key_release = t + hold

        # -- release phase: shift always released mid-hold, then key up --
        if needs_shift:
            if random.random() < 0.5:
                # release shift halfway through hold
                shift_rel = t + hold * 0.5
                raw_events.append((shift_rel, "shift", False))
                # then release the character key (lowercase for letters, mapped for symbols)
                rel_key = ch.lower() if ch.isalpha() else _SHIFT_KEY_MAP[ch]
                raw_events.append((key_release, rel_key, False))
                release_times.append(key_release)
            else:
                # release character key first, then shift
                # this means we don't need to map the character key
                raw_events.append((key_release - 0.005, key_name, False))
                raw_events.append((key_release, "shift", False))
                release_times.append(key_release)
        else:
            raw_events.append((key_release, key_name, False))
            release_times.append(key_release)

        t += hold

        # -- inter-key gap --
        if ch != text[-1]:
            base = random.lognormvariate(gap_mu, gap_sigma)
            if ch == " ":
                base *= post_space_gap
            base = max(0.03, min(base, 1.5)) / speed
            factor = min_gap_factor + (1 - min_gap_factor) * _key_dist(prev or ch, ch)
            t += base * factor

        prev = ch

    total_time = t

    # 2) Build frame timestamps (include final moment)
    frame_times = list(np.arange(0, total_time, 1 / fps))
    if frame_times[-1] < total_time:
        frame_times.append(total_time)

    # 3) Render frames
    frames = []
    for tf in frame_times:
        typed = sum(1 for rt in release_times if rt <= tf)
        img = Image.new("RGB", (W, H), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        # draw full text faded
        draw.text((x0, y0), text, font=font, fill=(100, 100, 100))
        # overlay typed portion
        if typed > 0:
            draw.text((x0, y0), text[:typed], font=font, fill=(200, 200, 200))
        frames.append(np.array(img))

    # copy the first frame once
    frames.insert(0, frames[0].copy())

    video = np.stack(frames, 0)  # shape (n_frames, H, W, 3)

    frame_times = [0.0, *frame_times]  # start with 0 timestamp

    # 4) Format key logs
    key_logs = []
    for t, key, is_down in sorted(raw_events, key=lambda e: e[0]):
        key_logs.append(
            {
                "action": {"action": "key_button", "key": key, "is_down": is_down},
                "timestamp": float(t),
            }
        )

    return video, key_logs, frame_times


with open("texts.txt") as f:
    texts = f.read().splitlines()

SCREEN_SIZE = smart_resize(854, 480, factor=28, min_pixels=0, max_pixels=854 * 480)

tokenizer = Tokenizer.load("gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json")


async def upload_sample(
    imgs: np.ndarray, actions: list[dict], output_path: str, frame_time: list[float]
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

    MAX_TOKENS = 24
    tokens_shape = (
        stream_metadata.output_video.total_frames // 2,
        MAX_TOKENS,  # Assuming max 32 tokens per frame
    )
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

    key_actions_jsonl_path = output_path + "/key_actions.jsonl"
    key_actions_jsonl = "\n".join(json.dumps(action) for action in actions)
    storage.Client().bucket(key_actions_jsonl_path[5:].split("/", 1)[0]).blob(
        key_actions_jsonl_path[5:].split("/", 1)[1]
    ).upload_from_string(key_actions_jsonl, content_type="application/json")

    tokens = [Action(**action) for action in actions]
    # print(frame_time[::2])
    result = keys_to_tokens(
        tokens,
        frame_time[1::2],
        tokenizer,
        clock_tick_len=0.5 / MAX_TOKENS,
        time_per_segment=0.5,
        press_threshold=0.15,
    )

    # print(result)
    # print("\n\n".join([debug_actions(line[0], tokenizer) for line in result]))
    # print(len(frame_time))
    def pad_tokens(tokens, max_len):
        return tokens + [tokenizer.mappings["[pad]"]] * (max_len - len(tokens))

    result_padded = [pad_tokens(line[0], MAX_TOKENS) for line in result]
    pad_mask = [
        [1] * len(line[0]) + [0] * (MAX_TOKENS - len(line[0])) for line in result
    ]
    tokenized_tensors = torch.tensor(result_padded, dtype=torch.uint16)
    pad_mask_array = torch.tensor(pad_mask, dtype=torch.bool)
    # print("\n\n".join([debug_actions(line, tokenizer) for line in tokenized_tensors]))

    imgs_zarr_trans = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    await asyncio.gather(
        *[
            append_batch(zarr_array, imgs_zarr_trans, 0),
            append_batch(
                timestamps_array, torch.arange(imgs.shape[0]).to(torch.uint64), 0
            ),
            append_batch(tokens_array, tokenized_tensors, 0),
            append_batch(zarr_pad_mask_array, pad_mask_array, 0),
        ]
    )


async def generate_sample(sem: asyncio.Semaphore, text: str, output_path: str):
    async with sem:
        vid, keys, frame_time = generate_typing_video(
            text,
            fps=4,
            font_path="/home/jonathan_inductionlabs_com/induction-labs/repos/synth/assets/arial.ttf",
            speed=3,
            random_position=True,
            frame_size=(SCREEN_SIZE[1], SCREEN_SIZE[0]),
        )
        await upload_sample(
            imgs=vid,
            actions=keys,
            output_path=output_path,
            frame_time=frame_time,
        )


with open("texts.txt") as f:
    TEXTS = f.read().splitlines()


async def _batch(start: int, tpl: str, width: int = 4):
    sem = asyncio.Semaphore(width)  # 4 concurrent tasks *inside* one process
    await asyncio.gather(
        *(
            generate_sample(sem, TEXTS[i], tpl.format(i=i))
            for i in range(start, start + width)
        )
    )


def worker(start: int, tpl: str, width: int = 4):
    asyncio.run(_batch(start, tpl, width))  # each process owns its own loop


# ── driver ───────────────────────────────────────────────────────────
def run_all(total: int, tpl: str, n_proc: int = 6, width: int = 4):
    starts = range(0, total, width)  # one “batch” every <width> indices
    with Pool(n_proc) as pool:
        for _ in tqdm(
            pool.imap_unordered(partial(worker, tpl=tpl, width=width), starts),
            total=len(starts),
        ):
            pass


# ── entry-point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import contextlib

    # asyncio.run(_batch(
    #     start=0,
    #     tpl="gs://induction-labs/jonathan/synth/typing_v0/sample_{i}.zarr",
    # ))

    with contextlib.suppress(
        RuntimeError
    ):  # makes the script work on Windows & macOS ≥3.8 too
        set_start_method("spawn")

    run_all(
        len(TEXTS),
        "gs://induction-labs/jonathan/synth/typing_with_keyboard_v4_24/sample_{i}.zarr",
        n_proc=16,
        width=4,
    )
    # print(TEXTS[0])
    # import random
    # value = random.randrange(10000)
    # asyncio.run(generate_sample(
    #     sem=asyncio.Semaphore(4),
    #     text=TEXTS[0],
    #     output_path=f"gs://induction-labs/jonathan/synth/typing_test/sample_{value}.zarr"
    # ))
