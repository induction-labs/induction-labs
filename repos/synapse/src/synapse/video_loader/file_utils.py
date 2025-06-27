from __future__ import annotations

import os
import re


def get_mp4_files(path: str) -> list[str]:
    """
    Gets all .mp4 files from a path (local or GCS).

    Args:
        path: Path to directory containing video files (local or GCS path)

    Returns:
        List of .mp4 file paths
    """
    if path.startswith("gs://"):
        # Handle GCS paths
        from google.cloud import storage

        # Parse GCS path
        path_parts = path.replace("gs://", "").split("/", 1)
        bucket_name = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ""
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # List all mp4 blobs
        blobs = list(bucket.list_blobs(prefix=prefix))
        return [
            f"gs://{bucket_name}/{blob.name}"
            for blob in blobs
            if blob.name.endswith(".mp4")
        ]
    else:
        # Handle local paths
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

        all_files = os.listdir(path)
        return [os.path.join(path, f) for f in all_files if f.endswith(".mp4")]


def validate_sequential_files(file_paths: list[str], prefix: str) -> list[str]:
    """
    Validates that files follow sequential numbering pattern and returns sorted list.

    Args:
        file_paths: List of file paths to validate
        prefix: Expected filename prefix (e.g., 'screen_capture_')

    Returns:
        List of file paths sorted by index

    Raises:
        AssertionError: If any indexes are missing in the sequence
    """
    pattern = rf"{re.escape(prefix)}(\d{{6}})\.mp4"
    indexed_files = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        match = re.match(pattern, filename)
        if match:
            index = int(match.group(1))
            indexed_files.append((index, file_path))

    # Sort by index
    indexed_files.sort(key=lambda x: x[0])

    # Validate sequence - check for missing indices
    if indexed_files:
        indices = [item[0] for item in indexed_files]
        expected_indices = list(range(indices[0], indices[-1] + 1))

        missing_indices = set(expected_indices) - set(indices)
        if missing_indices:
            raise AssertionError(
                f"Missing video file indices: {sorted(missing_indices)}"
            )

    return [item[1] for item in indexed_files]


def list_video_files(path: str) -> list[str]:
    """
    Lists all video files in a path with .mp4 extensions, numbered sequentially.

    Args:
        path: Path to directory containing video files (local or GCS path)

    Returns:
        List of video file paths sorted by index

    Raises:
        AssertionError: If any indexes are missing in the sequence
    """
    mp4_files = get_mp4_files(path)
    return validate_sequential_files(mp4_files, "screen_capture_")
