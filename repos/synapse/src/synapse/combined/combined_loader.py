from __future__ import annotations

from pydantic import BaseModel, field_validator
from synapse.video_loader.read_frames import (
    fetch_frames_from_zarr,
    fetch_metadata_from_zarr,
    get_frame_cursor_array,
)


class CombinedLoaderArgs(BaseModel):
    """
    Arguments for the CombinedLoader.
    """

    frames_per_action: int = 2
    frames_zarr_path: str
    actions_range: tuple[int, int]

    @property
    def frame_range(self) -> tuple[int, int]:
        """
        Get the real frame range based on frames_per_action.
        """
        return (
            self.actions_range[0] * self.frames_per_action,
            self.actions_range[1] * self.frames_per_action,
        )

    @field_validator("actions_range")
    @classmethod
    def validate_actions_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        """
        Validate that the actions range is a tuple of two integers.
        """
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("actions_range must be a tuple of two integers.")
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
        frames: Numpy array of shape (seq, frames_per_action, 3, H, W) containing the frames.
        mouse_movements: Numpy array of shape (seq, [x, y], [m, n, a]) containing the mouse movements.
        stream_metadata: StreamMetadata object containing metadata
    """
    stream_metadata = await fetch_metadata_from_zarr(args.frames_zarr_path)

    # actions = get_all_action_logs(args.action_logs_dir)

    # We need `frames_end + 1` because we need an action for the last frame

    actions_array, frames_per_action = await get_frame_cursor_array(
        args.frames_zarr_path,
    )
    assert frames_per_action == args.frames_per_action, (
        f"Expected frames_per_action {args.frames_per_action}, got {frames_per_action}."
    )

    assert (
        len(actions_array)
        == stream_metadata.output_video.total_frames // frames_per_action
    ), (
        f"Actions array length {len(actions_array)} does not match expected length "
        f"{stream_metadata.output_video.total_frames // frames_per_action}."
    )
    frames_start, frames_end = args.frame_range
    actions_start, actions_end = args.actions_range

    assert frames_end <= stream_metadata.output_video.total_frames, (
        f"Frame range {args.actions_range} exceeds total frames {stream_metadata.output_video.total_frames}."
    )

    actions_array = actions_array[actions_start:actions_end]
    frames = await fetch_frames_from_zarr(
        args.frames_zarr_path, (frames_start, frames_end)
    )  # [seq * frames_per_action, 3, H, W]
    assert frames.shape[0] == (actions_end - actions_start) * args.frames_per_action, (
        f"Expected frames shape to be {(actions_end - actions_start) * args.frames_per_action, 3, stream_metadata.output_video.resolution.height, stream_metadata.output_video.resolution.width}, "
        f"got {frames.shape}."
    )
    frames = frames.reshape(
        actions_end - actions_start,
        args.frames_per_action,
        3,
        stream_metadata.output_video.resolution.height,
        stream_metadata.output_video.resolution.width,
    )
    assert len(frames) == len(actions_array), (
        f"Expected frames length {len(actions_array)}, got {len(frames)}."
    )

    return (frames, actions_array), stream_metadata
