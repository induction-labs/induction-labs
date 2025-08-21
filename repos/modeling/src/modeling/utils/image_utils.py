from __future__ import annotations

import base64
import io
import os
import re
from io import BytesIO

import requests
from PIL import Image, ImageFile, ImageOps, features


def pil_open_from_url(
    url: str,
    timeout: float = 15.0,
    allow_truncated: bool = False,
    extra_headers: dict[str, str] | None = None,
) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0"}
    if extra_headers:
        headers.update(extra_headers)

    r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
    r.raise_for_status()
    data = r.content

    if allow_truncated:
        # Helps with slightly corrupted/partial files
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # global flag

    ctype = (r.headers.get("Content-Type") or "").lower()
    # Many CDNs use octet-stream; allow it. Block obvious non-image types.
    if not (ctype.startswith("image/") or ctype == "application/octet-stream"):
        raise ValueError(
            f"URL did not return an image. Content-Type={ctype!r}, "
            f"status={r.status_code}, final_url={r.url}"
        )

    # Common gotcha: WebP without Pillow support
    sig = data[:12]
    if sig[:4] == b"RIFF" and sig[8:12] == b"WEBP" and not features.check("webp"):
        raise RuntimeError(
            "This file is WebP but your Pillow lacks WebP support. "
            "Upgrade to a recent Pillow wheel (e.g. `pip install -U pillow`)."
        )

    try:
        im = Image.open(BytesIO(data))
        im.load()  # fully decode before returning
        return im
    except Image.UnidentifiedImageError as e:
        # Add helpful context for debugging
        preview = data[:32].hex()
        raise Image.UnidentifiedImageError(
            f"{e}. Cannot identify image. status={r.status_code}, "
            f"content-type={ctype}, len={len(data)}, final_url={r.url}, "
            f"first_32_bytes=0x{preview}"
        ) from e


def transform_image(
    image: Image.Image, image_dimensions: tuple[int, int] | None
) -> bytes:
    im = ImageOps.exif_transpose(image)
    if image_dimensions is not None:
        assert im.size == image_dimensions, (
            f"Image dimensions mismatch: expected {image_dimensions}, got {im.size}"
        )

    # Convert to PNG bytes
    # Use RGBA for broad compatibility; Pillow will drop alpha if not needed.
    converted = im.convert("RGBA")
    buf = io.BytesIO()
    converted.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    return png_bytes


def get_base64_from_image_path(
    image_path: str, image_dimensions: tuple[int, int] | None = None
) -> str | None:
    """
    Load an image from a URL, filesystem path, or base64 string (optionally a data URL),
    convert it to PNG, optionally assert its dimensions, and return a data URL:
    'data:image/png;base64,<...>'.

    Supports PNG/JPEG/WebP inputs.
    Prefers assertion to silent failure when dimensions are provided.
    """
    data_url_prefix = "data:image/png;base64,"

    def _read_bytes_from_input(s: str) -> Image.Image | bytes:
        s = s.strip()

        # Case 1: data URL like "data:image/xxx;base64,<b64>"
        m = re.match(r"^data:image/[^;]+;base64,(?P<payload>[A-Za-z0-9+/=\n\r]+)$", s)
        if m:
            return base64.b64decode(m.group("payload"), validate=True)

        # Case 2: looks like raw base64 (no filesystem path, no scheme)
        looks_like_path = ("://" in s) or os.path.exists(s)
        if not looks_like_path:
            try:
                return base64.b64decode(s, validate=True)
            except Exception:
                pass  # fall through to other options

        # Case 3: URL (http/https)
        if s.startswith(("http://", "https://")):
            return pil_open_from_url(s)

        # Case 4: filesystem path
        if os.path.exists(s):
            with open(s, "rb") as f:
                return f.read()

        raise ValueError("Input is not a valid URL, file path, or base64 image.")

    try:
        raw_bytes = _read_bytes_from_input(image_path)
        if isinstance(raw_bytes, Image.Image):
            # If we got an Image, convert it to bytes
            png_bytes = transform_image(raw_bytes, image_dimensions)
        elif isinstance(raw_bytes, bytes):
            # If we got raw bytes, convert them to an Image and then to PNG bytes
            with Image.open(io.BytesIO(raw_bytes)) as image:
                png_bytes = transform_image(image, image_dimensions)
        else:
            raise ValueError("Input must be an image or valid base64 string.")

        # Open with Pillow, apply EXIF-aware orientation, then (optionally) assert size.

        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"{data_url_prefix}{b64}"
    except Exception as e:
        print(f"Failed to convert image: {e} {image_path=} {image_dimensions=}")
        return None
