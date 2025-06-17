from __future__ import annotations

import datetime
import os
from urllib.parse import urlparse

from google.cloud import storage


def upload_to_gcs_and_delete(from_path: str, to_path: str):
    if not os.path.isfile(from_path):
        raise FileNotFoundError(f"Local file not found: {from_path}")

    # print(f"Uploading {from_path} to {to_path}...")

    parsed = urlparse(to_path)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Invalid GCS URI: {to_path}")

    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(from_path)

    os.remove(from_path)

    # print(f"Uploaded {from_path} → gs://{bucket_name}/{blob_name} and deleted local file.")


def recording_metadata(username: str, video_segment_buffer_length: float) -> dict:
    from pynput.screen import ScreenInfo

    physical_pixel_width, physical_pixel_height = ScreenInfo.get_screen_dimensions()
    logical_pixel_ratio = ScreenInfo.physical_to_logical_ratio()
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "username": username,
        "screen_info": {
            "video_width": int(physical_pixel_width),
            "video_height": int(physical_pixel_height),
            "logical_pixel_ratio": logical_pixel_ratio,
            # e.g., the space that mouse actions are recorded in
            "logical_pixel_width": int(physical_pixel_width / logical_pixel_ratio),
            "logical_pixel_height": int(physical_pixel_height / logical_pixel_ratio),
        },
        "video_segment_buffer_length": video_segment_buffer_length,
    }
