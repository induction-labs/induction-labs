import base64
import io
import os
import re
import urllib.request

from PIL import Image, ImageOps


def get_base64_from_image_path(
    image_path: str, image_dimensions: tuple[int, int] | None = None
) -> str:
    """
    Load an image from a URL, filesystem path, or base64 string (optionally a data URL),
    convert it to PNG, optionally assert its dimensions, and return a data URL:
    'data:image/png;base64,<...>'.

    Supports PNG/JPEG/WebP inputs.
    Prefers assertion to silent failure when dimensions are provided.
    """
    data_url_prefix = "data:image/png;base64,"

    def _read_bytes_from_input(s: str) -> bytes:
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
            with urllib.request.urlopen(s, timeout=30) as resp:
                return resp.read()

        # Case 4: filesystem path
        if os.path.exists(s):
            with open(s, "rb") as f:
                return f.read()

        raise ValueError("Input is not a valid URL, file path, or base64 image.")

    raw_bytes = _read_bytes_from_input(image_path)

    # Open with Pillow, apply EXIF-aware orientation, then (optionally) assert size.
    with Image.open(io.BytesIO(raw_bytes)) as im:
        im = ImageOps.exif_transpose(im)
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

    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"{data_url_prefix}{b64}"
