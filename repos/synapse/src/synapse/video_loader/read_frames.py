from __future__ import annotations

import numpy as np
import tensorstore as ts

from synapse.video_loader.zarr_utils import get_kvstore_config

from .typess import StreamMetadata


async def fetch_frames_from_zarr(
    zarr_path: str, frame_range: tuple[int, int]
) -> np.ndarray:
    """
    Load frames from a specific range from a tensorstore zarr file.

    Args:
        zarr_path: Path to the tensorstore zarr file (supports local paths, gs://, and s3://)
        frame_range: Tuple of (start_frame, end_frame) indices (0-based, end_frame is exclusive)

    Returns:
        Numpy array of shape (T, C, H, W) containing the frames
    """
    # Get the appropriate kvstore configuration
    kvstore_config = get_kvstore_config(zarr_path)

    # Open the tensorstore zarr array
    zarr_array = await ts.open({"driver": "zarr3", "kvstore": kvstore_config})

    start_frame, end_frame = frame_range

    # Read the specified frame range
    frames = await zarr_array[start_frame:end_frame].read()

    return frames


async def fetch_metadata_from_zarr(zarr_path: str) -> StreamMetadata:
    """
    Fetch metadata from a tensorstore zarr file.

    Args:
        zarr_path: Path to the tensorstore zarr file (supports local paths, gs://, and s3://)

    Returns:
        Metadata dictionary containing shape, dtype, and other attributes.
    """
    # Get the appropriate kvstore configuration
    kvstore_config = get_kvstore_config(zarr_path)

    # Open the tensorstore zarr array
    zarr_array = await ts.open({"driver": "zarr3", "kvstore": kvstore_config})

    # Fetch metadata
    full_spec = zarr_array.spec()
    # This is so dumb https://google.github.io/tensorstore/python/api/tensorstore.Spec.html
    try:
        metadata = StreamMetadata.model_validate(
            full_spec.to_json()["metadata"]["attributes"]["stream"]
        )
        return metadata
    except KeyError as e:
        # If metadata is not present, return an empty StreamMetadata
        raise KeyError(
            f"Metadata not found in the zarr file at {zarr_path}. Ensure the file contains metadata."
        ) from e


async def get_frame_at_timestamp(zarr_path: str, timestamp: float) -> np.ndarray:
    """
    Get the frame at a specific timestamp from a tensorstore zarr file.

    Args:
        zarr_path: Path to the tensorstore zarr file (supports local paths, gs://, and s3://)
        timestamp: Timestamp in seconds

    Returns:
        Numpy array of shape (C, H, W) containing the frame
    """
    # Read the timestamp array to find the closest frame
    stream_metadata = await fetch_metadata_from_zarr(zarr_path)
    stream_start = stream_metadata.start_time
    stream_end = stream_metadata.end_time
    assert timestamp >= stream_start and timestamp <= stream_end, (
        f"Timestamp {timestamp} is out of bounds for the stream: "
        f"start={stream_start}, end={stream_end}"
    )
    timestamp = timestamp - stream_start  # Normalize to start of stream
    print(stream_start)
    timestamps_path = zarr_path + "/timestamps"
    timestamps_kvstore_config = get_kvstore_config(timestamps_path)

    timestamps_array = await ts.open(
        {"driver": "zarr3", "kvstore": timestamps_kvstore_config}
    )
    timestamps = await timestamps_array.read()

    # Find the closest timestamp
    # print(timestamps)
    closest_idx = int(np.argmin(np.abs(timestamps - timestamp)))
    print(
        f"Closest index for timestamp {timestamp + stream_start} is {closest_idx}, value: {timestamps[closest_idx] + stream_start}"
    )

    # Get the frame at that index
    frames = await fetch_frames_from_zarr(zarr_path, (closest_idx, closest_idx + 1))

    return frames[0]  # Return single frame with shape (C, H, W)
