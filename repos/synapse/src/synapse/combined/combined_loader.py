from __future__ import annotations

import numpy as np
from pydantic import BaseModel, field_validator
from synapse.actions.mouse_movements import (
    convert_cubic_to_np,
    fill_mouse_move_actions,
    get_all_action_logs,
    process_continuous_actions,
)
from synapse.video_loader.read_frames import (
    convert_pts_array_to_timestamps,
    fetch_frames_from_zarr,
    fetch_metadata_from_zarr,
    get_frame_pts_array,
)


class CombinedLoaderArgs(BaseModel):
    """
    Arguments for the CombinedLoader.
    """

    frames_zarr_path: str
    frame_range: tuple[int, int]
    action_logs_dir: str

    @field_validator("frame_range")
    @classmethod
    def validate_frame_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        """
        Validate that the frame range is a tuple of two integers.
        """
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("frame_range must be a tuple of two integers.")
        start, end = value
        assert 0 <= start <= end, (
            f"Invalid frame range: {value}. Start must be less than or equal to end."
        )

        return value


async def combined_loader(args: CombinedLoaderArgs):
    """
    Combined loader function that fetches frames and metadata from a zarr file.

    Args:
        args: CombinedLoaderArgs containing the zarr path, frame range, and action logs directory.

    Returns:
        A dictionary containing frames and metadata.
    """
    stream_metadata = await fetch_metadata_from_zarr(args.frames_zarr_path)
    actions = get_all_action_logs(args.action_logs_dir)
    range_start, range_end = args.frame_range
    assert range_end <= stream_metadata.output_video.total_frames, (
        f"Frame range {args.frame_range} exceeds total frames {stream_metadata.output_video.total_frames}."
    )
    frame_pts = await get_frame_pts_array(args.frames_zarr_path)
    assert len(frame_pts) == stream_metadata.output_video.total_frames, (
        f"Frame PTS array length {len(frame_pts)} does not match total frames {stream_metadata.output_video.total_frames}."
    )
    frame_pts = frame_pts[range_start:range_end]
    frame_timestamps = convert_pts_array_to_timestamps(
        frame_pts, stream_metadata.input_video.time_base
    )
    screen_size = (
        stream_metadata.input_video.resolution.width,
        stream_metadata.input_video.resolution.height,
    )
    filled_actions = fill_mouse_move_actions(actions)
    mouse_x_cubic, mouse_y_cubic = process_continuous_actions(
        frame_timestamps, filled_actions, screen_size
    )
    mouse_x_cubic_np, mouse_y_cubic_np = (
        convert_cubic_to_np(mouse_x_cubic),
        convert_cubic_to_np(mouse_y_cubic),
    )
    mouse_movements = np.stack(
        [mouse_x_cubic_np, mouse_y_cubic_np], axis=1
    )  # Shape: (n, [x, y], [m, n, a]) for x and y cubic coefficients
    frames = await fetch_frames_from_zarr(
        args.frames_zarr_path, (range_start, range_end)
    )
    assert len(frames) == len(mouse_movements) + 1 == range_end - range_start, (
        f"Number of frames {len(frames)=} does not match number of mouse movements {len(mouse_movements) + 1=} "
        f"for range {args.frame_range=}."
    )
    return frames, mouse_movements, stream_metadata
