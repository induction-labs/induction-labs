from __future__ import annotations

from fractions import Fraction

from pydantic import BaseModel
from synapse.utils.logging import configure_logging
from synapse.video_loader.typess import VideoResolution
from synth.recordings.parse_actions import (
    Action,
    Point,
)

logger = configure_logging(__name__)


class ScreenInfo(BaseModel):
    video_width: int
    video_height: int
    logical_pixel_ratio: float
    logical_pixel_width: int
    logical_pixel_height: int


class RecordingMetadata(BaseModel):
    timestamp: float
    username: str
    screen_info: ScreenInfo
    video_segment_buffer_length: int
    time_base: Fraction


def transform_coordinates(
    original_point: Point,
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> Point:
    """
    Transform coordinates from original video dimensions to target dimensions.

    Args:
        original_point: Point in original coordinates
        original_width: Original video width
        original_height: Original video height
        target_width: Target video width
        target_height: Target video height

    Returns:
        Point in target coordinates
    """
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    return Point(x=int(original_point.x * scale_x), y=int(original_point.y * scale_y))


def transform_action_coordinates(
    action: Action,
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> Action:
    """
    Transform action coordinates from original to target dimensions.

    Args:
        action: Action with coordinates to transform
        original_width: Original video width
        original_height: Original video height
        target_width: Target video width
        target_height: Target video height

    Returns:
        Action with transformed coordinates
    """
    from copy import deepcopy

    # Create a deep copy to avoid modifying the original action
    transformed_action = deepcopy(action)

    # Transform coordinates based on action type
    if hasattr(transformed_action.action, "point"):
        transformed_action.action.point = transform_coordinates(
            transformed_action.action.point,
            original_width,
            original_height,
            target_width,
            target_height,
        )

    # Handle drag actions with start_point and end_point
    if hasattr(transformed_action.action, "start_point"):
        transformed_action.action.start_point = transform_coordinates(
            transformed_action.action.start_point,
            original_width,
            original_height,
            target_width,
            target_height,
        )

    if hasattr(transformed_action.action, "end_point"):
        transformed_action.action.end_point = transform_coordinates(
            transformed_action.action.end_point,
            original_width,
            original_height,
            target_width,
            target_height,
        )

    # Handle scroll actions with displacement
    if hasattr(transformed_action.action, "displacement"):
        # Scale displacement but don't transform it as a point since it's relative
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        original_displacement = transformed_action.action.displacement
        transformed_action.action.displacement = (
            int(original_displacement[0] * scale_x),
            int(original_displacement[1] * scale_y),
        )

    return transformed_action


def filter_offscreen_actions(
    actions: list[Action],
    recording_metadata: RecordingMetadata,
) -> list[Action]:
    """
    Filter out sequences of actions where coordinates are outside the logical pixel bounds.

    When an action with coordinates outside the logical pixel area is found, it removes
    actions from that point until the next action that is back on screen.

    Args:
        actions: List of actions to filter
        recording_metadata: Recording metadata containing logical pixel dimensions

    Returns:
        Filtered list of actions with offscreen sequences removed
    """
    if not actions:
        return actions

    # Get logical pixel bounds from recording metadata
    max_x = recording_metadata.screen_info.logical_pixel_width
    max_y = recording_metadata.screen_info.logical_pixel_height

    def is_point_onscreen(point: Point) -> bool:
        """Check if a point is within the logical pixel bounds."""
        return 0 <= point.x < max_x and 0 <= point.y < max_y

    def has_onscreen_coordinates(action: Action) -> bool:
        """Check if an action has coordinates and they are on screen."""
        action_obj = action.action

        # Check for actions with a single point
        if hasattr(action_obj, "point"):
            return is_point_onscreen(action_obj.point)

        # Check for drag actions with start and end points
        if hasattr(action_obj, "start_point") and hasattr(action_obj, "end_point"):
            return is_point_onscreen(action_obj.start_point) and is_point_onscreen(
                action_obj.end_point
            )

        # Actions without coordinates (like hotkeys, type) are considered onscreen
        return True

    filtered_actions = []
    i = 0

    while i < len(actions):
        action = actions[i]

        if has_onscreen_coordinates(action):
            # Action is on screen, keep it
            filtered_actions.append(action)
            i += 1
        else:
            # Action is off screen, skip until we find an action back on screen
            logger.debug(
                f"Found offscreen action at index {i}: {action.dump_to_text()}"
            )

            # Skip all actions until we find one back on screen
            j = i + 1
            while j < len(actions):
                if has_onscreen_coordinates(actions[j]):
                    # Found an action back on screen
                    logger.debug(f"Resuming at index {j}: {actions[j].dump_to_text()}")
                    break
                j += 1

            # Log how many actions were skipped
            skipped_count = j - i
            if skipped_count > 0:
                logger.debug(
                    f"Skipped {skipped_count} offscreen actions from index {i} to {j - 1}"
                )

            # Continue from the next on-screen action (j)
            i = j

    initial_count = len(actions)
    final_count = len(filtered_actions)

    if final_count != initial_count:
        logger.info(
            f"Filtered out {initial_count - final_count} offscreen actions "
            f"(kept {final_count}/{initial_count})"
        )

    return filtered_actions


def transform_action_coords_list(
    actions: list[Action],
    target_resolution: VideoResolution,
    recording_metadata: RecordingMetadata,
) -> list[Action]:
    """
    Transform a list of actions to match the target resolution.

    Args:
        actions: List of Action objects to transform
        target_resolution: Target video resolution
        recording_metadata: RecordingMetadata object containing original resolution

    Returns:
        List of transformed Action objects
    """
    actions = filter_offscreen_actions(actions, recording_metadata)
    return [
        transform_action_coordinates(
            action,
            recording_metadata.screen_info.logical_pixel_width,
            recording_metadata.screen_info.logical_pixel_height,
            target_resolution.width,
            target_resolution.height,
        )
        for action in actions
    ]
