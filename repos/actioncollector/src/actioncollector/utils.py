from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version


def get_bundled_credentials_path() -> str:
    """Get path to bundled service account credentials"""
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle
        bundle_dir = Path(sys._MEIPASS)
        credentials_path = bundle_dir / "credentials" / "service-account-key.json"
        if credentials_path.exists():
            return str(credentials_path)

    # Fallback to local file for development
    return "service-account-key.json"


def get_gcs_client():
    """Get authenticated Google Cloud Storage client using bundled credentials"""
    credentials_path = get_bundled_credentials_path()

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"Service account credentials not found at: {credentials_path}"
        )

    # Set the credentials environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    return storage.Client()


def upload_to_gcs_and_delete(from_path: str, to_path: str):
    if not os.path.isfile(from_path):
        raise FileNotFoundError(f"Local file not found: {from_path}")

    parsed = urlparse(to_path)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Invalid GCS URI: {to_path}")

    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(from_path)

    os.remove(from_path)

    # print(f"Uploaded {from_path} → gs://{bucket_name}/{blob_name} and deleted local file.")


def recording_metadata(
    username: str, video_segment_buffer_length: float, gs_file_path: str, framerate: int
) -> dict:
    from pynput.screen import ScreenInfo

    physical_pixel_width, physical_pixel_height = ScreenInfo.get_screen_dimensions()
    logical_pixel_ratio = ScreenInfo.physical_to_logical_ratio()

    # Get the action collector version
    try:
        action_collector_version = version("actioncollector")
    except Exception:
        action_collector_version = "unknown"

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "username": username,
        "action_collector_version": action_collector_version,
        "gs_file_path": gs_file_path,
        "framerate": framerate,
        "screen_info": {
            "video_width": int(physical_pixel_width),
            "video_height": int(physical_pixel_height),
            "logical_pixel_ratio": logical_pixel_ratio,
            # e.g., the space that mouse actions are recorded in
            "logical_pixel_width": int(physical_pixel_width / logical_pixel_ratio),
            "logical_pixel_height": int(physical_pixel_height / logical_pixel_ratio),
        },
        "video_segment_buffer_length": video_segment_buffer_length,
        "platform": sys.platform,
    }
