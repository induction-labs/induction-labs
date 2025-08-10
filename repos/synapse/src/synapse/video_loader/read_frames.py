from __future__ import annotations

from fractions import Fraction

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


type PTSArray = np.ndarray[tuple[int], np.dtype[np.integer]]
type TimestampsArray = np.ndarray[tuple[int], np.dtype[np.floating]]


async def get_frame_pts_array(
    zarr_path: str,
) -> PTSArray:
    """
    Get the timestamps array from a tensorstore zarr file.

    Args:
        zarr_path: Path to the tensorstore zarr file (supports local paths, gs://, and s3://)

    Returns:
        Numpy array of timestamps
    """
    timestamps_path = zarr_path + "/timestamps"
    timestamps_kvstore_config = get_kvstore_config(timestamps_path)

    pts_zarr = await ts.open({"driver": "zarr3", "kvstore": timestamps_kvstore_config})

    pts_array: np.ndarray = await pts_zarr.read()
    assert pts_array.ndim == 1, (
        f"Expected 1D array for timestamps, got {pts_array.ndim}D array with shape {pts_array.shape}"
    )
    assert pts_array.dtype == np.uint64, (
        f"Expected timestamps to be in uint64 format, got {pts_array.dtype}"
    )
    return pts_array


async def get_frame_cursor_array(
    zarr_path: str,
) -> tuple[np.ndarray, int]:
    """
    Get the timestamps array from a tensorstore zarr file.

    Args:
        zarr_path: Path to the tensorstore zarr file (supports local paths, gs://, and s3://)

    Returns:
        Numpy array of timestamps
    """
    timestamps_path = zarr_path + "/cursor_action"
    timestamps_kvstore_config = get_kvstore_config(timestamps_path)

    pts_zarr = await ts.open({"driver": "zarr3", "kvstore": timestamps_kvstore_config})
    full_spec = pts_zarr.spec()
    frames_per_action = full_spec.to_json()["metadata"]["attributes"][
        "frames_per_action_step"
    ]
    assert isinstance(frames_per_action, int), (
        f"Expected frames_per_action_step to be an integer, got {type(frames_per_action)}"
    )

    pts_array: np.ndarray = await pts_zarr.read()
    assert pts_array.ndim == 3, (
        f"Expected 1D array for timestamps, got {pts_array.ndim}D array with shape {pts_array.shape}"
    )
    assert pts_array.shape[1] == 2 and pts_array.shape[2] == 3, (
        f"Expected cursor action array to have shape (N, 2, 3), got {pts_array.shape}"
    )
    assert pts_array.dtype == np.float32, (
        f"Expected timestamps to be in uint64 format, got {pts_array.dtype}"
    )
    return pts_array, frames_per_action


async def get_frame_keyboard_tokens(
    zarr_path: str,
) -> np.ndarray:
    """
    Get the keyboard tokens array from a tensorstore zarr file.

    Args:
        zarr_path: Path to the tensorstore zarr file (supports local paths, gs://, and s3://)

    Returns:
        Numpy array of timestamps
    """
    timestamps_path = zarr_path + "/keyboard_tokens"
    timestamps_kvstore_config = get_kvstore_config(timestamps_path)

    pts_zarr = await ts.open({"driver": "zarr3", "kvstore": timestamps_kvstore_config})

    keyboard_token_array: np.ndarray = await pts_zarr.read()
    assert keyboard_token_array.ndim == 2, (
        f"Expected 2D array for tokens, got {keyboard_token_array.ndim}D array with shape {keyboard_token_array.shape}"
    )
    assert keyboard_token_array.dtype == np.uint16, (
        f"Expected tokens to be in uint16 format, got {keyboard_token_array.dtype}"
    )
    return keyboard_token_array

async def get_frame_keyboard_mask(
    zarr_path: str,
) -> np.ndarray:
    """
    Get the keyboard mask array from a tensorstore zarr file.

    Args:
        zarr_path: Path to the tensorstore zarr file (supports local paths, gs://, and s3://)

    Returns:
        Numpy array of keyboard mask
    """
    timestamps_path = zarr_path + "/keyboard_tokens_mask"
    timestamps_kvstore_config = get_kvstore_config(timestamps_path)

    pts_zarr = await ts.open({"driver": "zarr3", "kvstore": timestamps_kvstore_config})

    keyboard_mask_array: np.ndarray = await pts_zarr.read()
    assert keyboard_mask_array.ndim == 2, (
        f"Expected 2D array for tokens, got {keyboard_mask_array.ndim}D array with shape {keyboard_mask_array.shape}"
    )
    assert keyboard_mask_array.dtype == np.bool, (
        f"Expected tokens to be in bool format, got {keyboard_mask_array.dtype}"
    )
    return keyboard_mask_array


def convert_pts_array_to_timestamps(
    pts_array: PTSArray, time_base: Fraction
) -> TimestampsArray:
    """
    Convert a PTS array to timestamps.

    Args:
        pts_array: PTS array (1D numpy array of int64)
        time_base: Time base in seconds

    Returns:
        Timestamps array (1D numpy array of float64)
    """

    timestamps = pts_array * time_base.numerator / time_base.denominator

    assert timestamps.ndim == 1, (
        f"Expected 1D array for timestamps, got {timestamps.ndim}D array with shape {timestamps.shape}"
    )
    return timestamps  # type: ignore[name-defined]


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
    timestamp_pts = int(timestamp / stream_metadata.input_video.time_base)
    assert (
        timestamp_pts >= stream_metadata.input_video.start_pts
        and timestamp_pts <= stream_metadata.input_video.end_pts
    ), (
        f"Timestamp {timestamp_pts} is out of bounds for the stream: "
        f"start={stream_metadata.input_video.start_pts}, end={stream_metadata.input_video.end_pts}"
    )

    stream_pts = await get_frame_pts_array(zarr_path)

    # Find the closest timestamp
    # print(timestamps)
    closest_idx = int(np.argmin(np.abs(stream_pts - timestamp_pts)))
    print(
        f"Closest index for timestamp {timestamp} with {timestamp_pts=} is {closest_idx}, value: {stream_pts[closest_idx]} ({float(stream_pts[closest_idx] * stream_metadata.input_video.time_base)}s)"
    )

    # Get the frame at that index
    frames = await fetch_frames_from_zarr(zarr_path, (closest_idx, closest_idx + 1))

    return frames[0]  # Return single frame with shape (C, H, W)
