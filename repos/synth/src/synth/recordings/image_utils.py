from __future__ import annotations

from io import BytesIO

from PIL import Image


def pil_to_bytes(img: Image.Image, format="JPEG", **save_kwargs) -> bytes:
    """
    Return encoded image bytes (PNG/JPEG/etc).
    Example: pil_to_bytes(img, "JPEG", quality=90, optimize=True)
    """
    buf = BytesIO()
    # JPEG needs RGB (no alpha)
    if format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buf, format=format, **save_kwargs)
    return buf.getvalue()
