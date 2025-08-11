from __future__ import annotations

import base64
import json
import multiprocessing
import secrets
import string
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from multiprocessing import Manager, Process
from multiprocessing.managers import ListProxy
from queue import Empty, Queue

import av
import dotenv
import gcsfs
import pandas as pd
import PIL
import PIL.Image
import tqdm
from google import genai
from google.genai import types
from pydantic import BaseModel
from smart_open import open as smart_open
from synapse.utils.logging import configure_logging, logging
from synapse.video_loader.typess import VideoResolution, resolution_1080p
from synapse.video_loader.video import (
    VideoMetadataFetcher,
    download_video_with_ffmpeg_copy,
    get_resize_filter,
    smart_resize,
)
from synth.recordings.action_models import FinishedAction
from synth.recordings.image_utils import pil_to_bytes
from synth.recordings.parse_actions import Action, Point, ScrollAction, parse_actions
from synth.recordings.synth_captions_generated_samples import (
    PROMPT_WITHOUT_NEXT,
    extract_frames_by_pts_from_container,
    get_actions,
)

dotenv.load_dotenv()

logger = configure_logging(__file__, logging.INFO)

gem_client = genai.Client()


def is_close(a: Point, b: Point, tol: float = 10) -> bool:
    return abs(a.x - b.x) <= tol and abs(a.y - b.y) <= tol


def reverse_chunk_actions_to_length(
    actions: list[Action],
    length: int = 5,
) -> list[list[Action]]:
    if len(actions) < length - 1:
        return []
    actions, last_block = actions[: -length + 1], actions[-length + 1 :]
    last_block.append(
        Action(
            action=FinishedAction(),
            timestamp=last_block[-1].end_timestamp + SS_DELAY + BEFORE_ACTION_BUFFER,
            end_timestamp=last_block[-1].end_timestamp
            + SS_DELAY
            + BEFORE_ACTION_BUFFER,
        )
    )
    action_blocks = [last_block]
    for i in range(len(actions) - length + 1, -1, -length):
        block = actions[i : i + length]
        if len(block) == length:
            action_blocks.append(block)
    return action_blocks


def segment_actions_by_time_gaps(
    actions: list[Action],
    gap_threshold: float = 5.0,
) -> list[list[Action]]:
    """
    Segment actions into chunks based on time gaps between consecutive actions.

    Args:
        actions: List of actions to segment
        gap_threshold: Time threshold in seconds for creating new segment (default: 5.0)
        actions_per_segment: Maximum number of actions per segment (default: 5)

    Returns:
        List of action chunks, where each chunk is a list of consecutive actions
        without large time gaps
    """
    if not actions:
        return []

    # Sort actions by timestamp to ensure proper ordering
    sorted_actions = sorted(actions, key=lambda a: a.timestamp)

    segments = []
    current_segment = [sorted_actions[0]]

    for i in range(1, len(sorted_actions)):
        prev_action = sorted_actions[i - 1]
        current_action = sorted_actions[i]

        # Calculate gap between end of previous action and start of current action
        time_gap = current_action.timestamp - prev_action.end_timestamp

        if time_gap > gap_threshold:
            # Gap is too large, start a new segment
            segments.append(current_segment)
            current_segment = [current_action]
        else:
            # Continue the current segment
            current_segment.append(current_action)

    # Don't forget the last segment
    if current_segment:
        segments.append(current_segment)

    return segments


def combine_scroll_actions(actions: list[Action]) -> list[Action]:
    new_actions: list[Action] = []
    i = 0
    while i < len(actions):
        action = actions[i]
        i += 1
        if not isinstance(action.action, ScrollAction):
            new_actions.append(action)
            continue
        combined_scroll_action = action
        while (
            i < len(actions)
            and isinstance(actions[i].action, ScrollAction)
            and is_close(action.action.point, actions[i].action.point, tol=10)
        ):
            next_action = actions[i]
            combined_scroll_action.action.displacement = (
                combined_scroll_action.action.displacement[0]
                + next_action.action.displacement[0],
                combined_scroll_action.action.displacement[1]
                + next_action.action.displacement[1],
            )
            combined_scroll_action.end_timestamp = next_action.end_timestamp
            i += 1
        combined_scroll_action.action.direction = (
            "up" if combined_scroll_action.action.displacement[1] > 0 else "down"
        )
        new_actions.append(combined_scroll_action)
    return new_actions


SS_DELAY = 0.20
BEFORE_ACTION_BUFFER = 0.0


def get_timesteps_range(actions: list[Action]) -> tuple[float, list[float]]:
    timestamps: list[float] = []
    timestamps.append(actions[0].timestamp - SS_DELAY)
    for i, prev_action in enumerate(actions[:-1]):
        next_action = actions[i + 1]
        end_timestamp = min(
            prev_action.end_timestamp + SS_DELAY,
            next_action.timestamp - BEFORE_ACTION_BUFFER,
        )
        timestamps.append(end_timestamp)

    timestamps.append(actions[-1].end_timestamp + SS_DELAY)

    assert len(timestamps) >= 1, "No timestamps generated from actions"
    return (timestamps[0], timestamps[1:])


def get_video_instruction(
    first_frame: bytes,
    rest_frames: list[bytes],
    actions: list[Action],
) -> tuple[str, str] | None:
    """
    Generate an instruction based on the first frame and the actions.
    """

    image_action_pairs = []
    for i, action in enumerate(actions):
        image_action_pairs.append(action.dump_to_text())
        image_action_pairs.append(
            types.Part.from_bytes(
                data=rest_frames[i],
                mime_type="image/png",
            )
        )

    contents = [
        "The following are screenshots and the actions a user took from a video recording. Analyze what action the user took in the screenshots and then write a plausible instruction that would result in the behaviour shown in the screenshots.",
        types.Part.from_bytes(data=first_frame, mime_type="image/png"),
        *image_action_pairs,
        """After reviewing the screenshots and actions, end your response with:\n### Instruction: <instruction>""",
    ]
    # uploaded_file = gem_client.files.upload(file=image1_path)

    model_response = gem_client.models.generate_content(
        model="gemini-2.5-pro", contents=contents
    )
    assert model_response.candidates, "No candidates returned from model"
    assert model_response.candidates[0].content
    assert model_response.candidates[0].content.parts, "No content parts returned"
    model_response_text = model_response.candidates[0].content.parts[0].text
    if not model_response_text:
        print("Model response is empty")
        return None
    response_parts = model_response_text.split("### Instruction:")
    if len(response_parts) < 2:
        print(f"Model response did not contain instruction part: {model_response_text}")
        return None
    model_instruction = response_parts[-1].strip()
    if not model_instruction:
        print("Model instruction is empty")
        return None
    return model_response_text, model_instruction


class SaveAction(BaseModel):
    step: int
    image: str
    action: str
    text: str
    thinking: str
    frame_metadata: FrameMetadata


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


class TrainSample(BaseModel):
    actions: list[SaveAction]
    instruction: str
    metadata: dict | None = None


def get_thinking_texts(
    first_frame: bytes,
    rest_frames: list[bytes],
    actions: list[Action],
    instruction: str,
) -> list[str]:
    thinking_texts = []
    old_turns = []
    all_frames = [first_frame, *rest_frames]
    for i, action in enumerate(actions):
        text_prompt = PROMPT_WITHOUT_NEXT.format(
            instruction=instruction,
            old_turns=old_turns,
            new_action=action.dump_to_text(),
        )
        contents = [
            text_prompt,
            types.Part.from_bytes(data=all_frames[i], mime_type="image/png"),
            types.Part.from_bytes(data=all_frames[i + 1], mime_type="image/png"),
        ]
        # model = "gemini-2.5-flash"
        model = "gemini-2.5-pro"
        model_response = gem_client.models.generate_content(
            model=model, contents=contents
        )
        assert model_response.candidates, "No candidates returned from model"
        assert model_response.candidates[0].content
        assert model_response.candidates[0].content.parts, "No content parts returned"
        model_response_text = model_response.candidates[0].content.parts[0].text
        thinking_texts.append(model_response_text)
        logger.debug(f"Thinking text for action {i}: {model_response_text}")
        old_turns.append(f"{model_response_text}\n{action.dump_to_text()}")
    return thinking_texts


def process_actions_range(
    source_dir: str,
    recording_metadata: RecordingMetadata,
    target_resolution: VideoResolution,
    actions: list[Action],
) -> TrainSample | None:
    """
    Process a range of actions to create multiple TrainSamples.

    Args:
        source_dir: Source directory path (e.g., gs://bucket/folder)
        recording_metadata: RecordingMetadata object
        actions: List of actions to process

    Returns:
        List of TrainSamples created from the actions
    """
    first_t, rest_timestamps = get_timesteps_range(actions)
    timestamps = [first_t, *rest_timestamps]
    logger.debug(f"{timestamps=}")
    frames_with_metadata = get_frames_at_timestamps(
        source_dir,
        recording_metadata,
        timestamps,
        target_resolution=target_resolution,
    )
    logger.debug("Got frames")
    frames = [frame[0] for frame in frames_with_metadata]
    frame_metadatas = [frame[1] for frame in frames_with_metadata]
    frame_buffers = [pil_to_bytes(frame, format="PNG") for frame in frames]
    try:
        video_instruction = get_video_instruction(
            frame_buffers[0],
            frame_buffers[1:],
            actions,
        )
        logger.debug("Got video instruction")
        if video_instruction is None:
            logger.warning("Failed to generate video instruction ")
            return None
        instruction_text, instruction = video_instruction
    except Exception as e:
        logger.error(f"Error generating video instruction {e}", exc_info=True)
        return None
    try:
        thinking_texts = get_thinking_texts(
            frame_buffers[0],
            frame_buffers[1:],
            actions,
            instruction_text,
        )
        logger.debug("Got thinking texts")
    except Exception as e:
        logger.error(f"Error generating thinking texts for  {e}", exc_info=True)
        return None
    b64_images = [base64.b64encode(bytes).decode("utf-8") for bytes in frame_buffers]
    save_actions = [
        SaveAction(
            step=i,
            image=b64_images[i],
            action=action.dump_to_text(),
            text=f"{thinking_texts[i]}\nAction: {action.dump_to_text()}",
            thinking=thinking_texts[i],
            frame_metadata=frame_metadatas[i],
        )
        for i, action in enumerate(actions)
    ]
    save_actions.append(
        SaveAction(
            step=len(actions),
            image=b64_images[-1],
            action="finished()",
            text="Action: finished()",
            thinking="",
            frame_metadata=frame_metadatas[-1],  # Use the last frame metadata
        )
    )

    return TrainSample(
        actions=save_actions,
        instruction=instruction,
        metadata={
            "source_dir": source_dir,
            "start_time": float(frame_metadatas[0].timestamp),
            "end_time": float(frame_metadatas[-1].timestamp),
            "instruction_text": instruction_text,
            "original_resolution": {
                "width": recording_metadata.screen_info.video_width,
                "height": recording_metadata.screen_info.video_height,
            },
            "output_resolution": {
                "width": recording_metadata.screen_info.logical_pixel_width,
                "height": recording_metadata.screen_info.logical_pixel_height,
            },
            "logical_pixel_ratio": recording_metadata.screen_info.logical_pixel_ratio,
        },
    )


def get_actions_df(source_dir: str) -> pd.DataFrame:
    raw_actions = get_actions(source_dir)
    parsed_actions = parse_actions(raw_actions)
    combined_actions = combine_scroll_actions(parsed_actions)
    return pd.DataFrame(
        [
            {
                "action": a,
                "start_timestamp": a.timestamp,
                "end_timestamp": a.end_timestamp,
                "action_type": a.action.action_type,
            }
            for a in combined_actions
        ]
    )


def gen_id(length: int = 8) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def get_video_index(recording_metadata: RecordingMetadata, timestamp: float) -> int:
    """
    Get the video index from the recording metadata.

    Args:
        recording_metadata: RecordingMetadata object

    Returns:
        Video index as a string
    """
    rel_timestamp = timestamp - recording_metadata.timestamp
    if rel_timestamp < 0:
        logger.warning(
            f"Timestamp {timestamp} is before recording start time {recording_metadata.timestamp}. Returning index 0."
        )
        return 0
    video_segment_length = recording_metadata.video_segment_buffer_length
    video_index = int(rel_timestamp // video_segment_length)
    return video_index


class FrameMetadata(BaseModel):
    timestamp: float
    original_timestamp: float
    video_path: str


def get_target_resolution(
    recording_metadata: RecordingMetadata,
    max_pixels: int = resolution_1080p.pixels,
) -> VideoResolution:
    """
    Get the target resolution based on the recording metadata and optional target resolution.
    """
    resize_h, resize_w = smart_resize(
        recording_metadata.screen_info.video_height,
        recording_metadata.screen_info.video_width,
        min_pixels=0,
        max_pixels=max_pixels,
    )
    output_resolution = VideoResolution(width=resize_w, height=resize_h)
    return output_resolution


def get_frames_at_timestamps(
    source_dir: str,
    recording_metadata: RecordingMetadata,
    timestamps: list[float],
    target_resolution: VideoResolution,
) -> list[tuple[PIL.Image.Image, FrameMetadata]]:
    """
    Extract frames at specific timestamps from videos in a source directory.

    Args:
        source_dir: Source directory path (e.g., gs://bucket/folder)
        recording_metadata: RecordingMetadata object containing video segment info
        timestamps: List of sorted timestamps to extract frames from

    Returns:
        List of tuples (image, metadata) corresponding to the timestamps
    """
    if not timestamps:
        return []

    # Group timestamps by video index
    video_groups: dict[int, list[tuple[int, float]]] = {}
    for i, timestamp in enumerate(timestamps):
        video_index = get_video_index(recording_metadata, timestamp)
        if video_index not in video_groups:
            video_groups[video_index] = []
        video_groups[video_index].append((i, timestamp))

    # Initialize results list with correct size
    results: list[tuple[PIL.Image.Image, FrameMetadata] | None] = [None] * len(
        timestamps
    )

    # Set up resize filter once using RecordingMetadata screen info
    screen_info = recording_metadata.screen_info
    #
    buffer_src, buffer_sink, filter_graph = None, None, None
    input_resolution = VideoResolution(
        width=screen_info.video_width, height=screen_info.video_height
    )

    # Process each video index
    for video_index, timestamp_pairs in video_groups.items():
        video_path = f"{source_dir}/screen_capture_{video_index:06d}.mp4"

        # Extract just the timestamps for this video
        video_timestamps = [ts for _, ts in timestamp_pairs]

        try:
            # Download and process video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
                download_video_with_ffmpeg_copy(
                    source_path=video_path,
                    dest_path=f.name,
                    ffmpeg_path="/nix/store/q7j5awbg80d38p9my5b5zgn0xadgvbmb-ffmpeg-7.1.1-bin/bin/ffmpeg",
                )

                with av.open(f.name) as video_container:
                    # Use consistent filter based on RecordingMetadata
                    if target_resolution is not None:
                        buffer_src, buffer_sink, filter_graph = get_resize_filter(
                            video_container.streams.video[0].format,
                            input_resolution,
                            target_resolution,
                            recording_metadata.time_base,
                        )

                    # Extract frames at the relative timestamps
                    frame_tuples = extract_frames_by_pts_from_container(
                        video_container,
                        video_timestamps,
                        buffer_src,
                        buffer_sink,
                    )

                    # Store frames in the correct positions in results
                    for (frame_timestamp, image), (
                        original_index,
                        original_timestamp,
                    ) in zip(frame_tuples, timestamp_pairs, strict=True):
                        frame_metadata = FrameMetadata(
                            timestamp=frame_timestamp,
                            video_path=video_path,
                            original_timestamp=original_timestamp,
                        )
                        results[original_index] = (image, frame_metadata)

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}", exc_info=True)
            # Fill with None values for failed video
            for original_index, _ in timestamp_pairs:
                results[original_index] = None

    # Filter out None values and return only successful frames
    return [result for result in results if result is not None]


def load_metadata(source_dir: str) -> RecordingMetadata:
    """
    Load and validate metadata.json from a source directory.

    Args:
        source_dir: Source directory path (e.g., gs://bucket/folder)

    Returns:
        RecordingMetadata object with validated screen_info

    Raises:
        Exception: If metadata.json cannot be loaded or is invalid
    """
    fs = gcsfs.GCSFileSystem()

    try:
        # Remove gs:// prefix and construct metadata path
        gs_path = source_dir.replace("gs://", "")
        metadata_path = f"{gs_path}/metadata.json"

        with fs.open(metadata_path, "r") as f:
            metadata_dict = json.load(f)

        # Convert timestamp to
        metadata_dict["timestamp"] = 0
        # Validate using Pydantic

    except Exception as e:
        raise Exception(
            f"Failed to load or validate metadata from {source_dir}: {e}"
        ) from e
    first_video_path = f"{source_dir}/screen_capture_000000.mp4"
    try:
        # Validate first video
        # We need to do this because action recorder is not saving video timestamp in utc prior to 0.1.2 mac build
        with fs.open(first_video_path, "rb") as f:
            video_container = av.open(f)
            video_stream = video_container.streams.video[0]
            video_metadata = VideoMetadataFetcher.get_video_metadata(video_stream)
        first_video_start_time = float(
            video_metadata.start_pts * video_metadata.time_base
        )
        metadata_dict["timestamp"] = first_video_start_time
        metadata_dict["time_base"] = video_metadata.time_base

        metadata = RecordingMetadata(**metadata_dict)
        # Check if video resolution matches screen info
        if (
            video_metadata.resolution.width != metadata.screen_info.video_width
            or video_metadata.resolution.height != metadata.screen_info.video_height
        ):
            raise ValueError("Video resolution does not match screen info in metadata")
    except Exception as e:
        raise Exception(
            f"Failed to validate first video {first_video_path} in {source_dir}: {e}"
        ) from e
    return metadata


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


def chunkify(lst, n):
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def save_and_get_metadata(
    source_dir: str,
    output_dir: str,
    train_sample: TrainSample,
) -> dict:
    """
    Process a single video for multiprocessing.

    Args:
        args: Tuple containing (video_path, source_dir, actions_df, select_policy, output_dir, metadata)

    Returns:
        List of dictionaries with train sample metadata
    """

    train_sample_id = gen_id(12)

    # Save the train sample to file
    with smart_open(
        f"{output_dir}/metadata/{train_sample_id}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(train_sample.model_dump()["actions"], f, ensure_ascii=False, indent=0)
    with smart_open(
        f"{output_dir}/train_samples/{train_sample_id}.metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            train_sample.model_dump(exclude={"actions"}),
            f,
            ensure_ascii=False,
            indent=0,
        )

    # Add metadata for the main process to collect
    return {
        "attempt_id": train_sample_id,
        "eval_task_id": train_sample_id,
        "actions": [s.action for s in train_sample.actions],
        "thinking": [s.thinking for s in train_sample.actions],
        "instruction": train_sample.instruction,
        "trajectory_length": len(train_sample.actions),
        "source_dir": source_dir,
        "image_turns_start": 0,
        "image_turns_end": len(train_sample.actions) - 1,
        "text_turns_start": 0,
        "text_turns_end": len(train_sample.actions) - 1,
        "unmask_last_only": False,
    }


def thread_worker(
    source_dir: str,
    output_dir: str,
    recording_metadata: RecordingMetadata,
    target_resolution: VideoResolution,
    actions: list[Action],
):
    """Thread worker function that processes a single video."""
    try:
        result = process_actions_range(
            source_dir, recording_metadata, target_resolution, actions
        )
        if result is None:
            return (None, None)
        metadata = save_and_get_metadata(source_dir, output_dir, result)
        logger.debug(f"Processed {len(actions)} actions from {source_dir}")
        return (metadata, None)  # video_path, results, error
    except Exception:
        return (None, traceback.format_exc())


def process_worker(
    video_args_chunk: list[
        tuple[str, str, RecordingMetadata, VideoResolution, list[Action]]
    ],
    threads_per_proc: int,
    update_queue: Queue[tuple[dict | None, str | None]],
):
    """
    Runs in each separate process. Spins up a thread pool to process its chunk of videos.
    Reports completion (with error or not) back via update_queue.
    """
    with ThreadPoolExecutor(max_workers=threads_per_proc) as executor:
        futures = {
            executor.submit(thread_worker, *video_args): video_args
            for video_args in video_args_chunk
        }
        for fut in as_completed(futures):
            logging.debug(f"Future completed: {fut}")
            save_action, err = fut.result()
            # send results so main can aggregate them
            update_queue.put((save_action, err))


def progress_listener[T](
    total: int,
    update_queue: Queue[tuple[T | None, str | None]],
    results_collector: ListProxy[T],
):
    """
    Runs in a thread in main process to consume updates and show progress bar.
    Also collects results from all processes.
    """
    pbar = tqdm.tqdm(total=total, desc="Processing videos", unit="video")
    completed = 0
    while completed < total:
        try:
            save_action, err = update_queue.get(timeout=1)
        except Empty:
            if completed >= total:
                break
            continue
        except Exception:
            import traceback

            tqdm.tqdm.write(
                f"[!] Error getting update from queue: {traceback.format_exc()}"
            )
            continue
        if err:
            tqdm.tqdm.write(f"[!] Error processing action range:\n{err}")
        else:
            if save_action is not None:
                results_collector.append(save_action)
        completed += 1
        pbar.update(1)
    pbar.close()


def process_videos(
    source_folders: list[str],
    output_dir: str,
    num_processes: int | None = None,
    threads_per_process: int = 12,
    max_video_files: int | None = None,
):
    """
    Process videos in parallel using multiprocessing + ThreadPoolExecutor from multiple source folders.

    Args:
        source_folders: List of directories containing source videos and actions
        output_dir: Directory to save processed results
        num_processes: Number of processes to use. If None, uses CPU count
        threads_per_process: Number of threads per process (default: 12)
        max_video_files: Maximum number of video files to process
    """
    # Discover all video files in the source folders

    raw_action_sets = {k: get_actions(k) for k in source_folders}
    video_metadatas = {k: load_metadata(k) for k in source_folders}
    target_resolutions = {
        k: get_target_resolution(v) for k, v in video_metadatas.items()
    }
    action_sets = {
        k: transform_action_coords_list(
            combine_scroll_actions(parse_actions(v)),
            target_resolutions[k],
            video_metadatas[k],
        )
        for k, v in raw_action_sets.items()
    }
    action_segments = [
        (source_dir, segment_actions_by_time_gaps(actions))
        for source_dir, actions in action_sets.items()
    ]

    unravelled_segments = [
        (source_dir, action_chunk)
        for source_dir, segments in action_segments
        for actions in segments
        for action_chunk in reverse_chunk_actions_to_length(actions, 10)
    ]
    action_segment_lens_df = pd.DataFrame(
        [
            {
                "source_dir": source_dir,
                "num_actions": len(actions),
            }
            for source_dir, actions in unravelled_segments
        ]
    )
    print("Action segments summary:")
    print(action_segment_lens_df.describe())

    if not unravelled_segments:
        print("No video files found in the specified folders")
        return
    if max_video_files is not None:
        unravelled_segments = unravelled_segments[:max_video_files]
        print(f"Limiting to first {max_video_files} video files")

    # Build mappings for efficiency

    # Prepare arguments for each video
    process_args = [
        (
            source_dir,
            output_dir,
            video_metadatas[source_dir],
            target_resolutions[source_dir],
            actions,
        )
        for source_dir, actions in unravelled_segments
    ]

    # Use all CPU cores if num_processes not specified
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(
        f"Processing {len(process_args)} videos using {num_processes} processes x {threads_per_process} threads..."
    )

    # Use multiprocessing + threading pattern
    manager = Manager()
    update_queue: Queue[tuple[SaveAction | None, str | None]] = manager.Queue()
    results_collector: ListProxy[SaveAction] = (
        manager.list()
    )  # shared list to collect results

    # Start progress listener thread
    listener = threading.Thread(
        target=progress_listener,
        args=(len(process_args), update_queue, results_collector),
        daemon=True,
    )
    listener.start()

    # Split video args among processes
    chunks = chunkify(process_args, num_processes)
    processes = []
    for chunk in chunks:
        if not chunk:
            continue
        p = Process(
            target=process_worker,
            args=(chunk, threads_per_process, update_queue),
        )
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Interrupted, terminating child processes...")
        for p in processes:
            p.terminate()

    # Wait until listener sees all updates
    listener.join()

    # Convert results to regular list and save
    results = list(results_collector)
    if results:
        train_samples_df = pd.DataFrame(results)
        train_samples_df.to_json(
            f"{output_dir}/samples.jsonl",
            orient="records",
            lines=True,
        )

    print(f"Saved {len(results)} train samples to {output_dir}/samples.jsonl")


def main() -> None:
    process_videos(
        [
            "gs://induction-labs-data-ext/action_capture/jeffrey/2025-08-10_133207_0V8HU",
            "gs://induction-labs-data-ext/action_capture/Jarry/2025-08-10_121140_Q4KI9",
            "gs://induction-labs-data-ext/action_capture/jonathan/2025-07-17_093647_KZ3CG",
            # "gs://induction-labs-data-ext/action_capture/aryan_91532/2025-07-07_170814_A2QD2",
            # "gs://induction-labs-data-ext/action_capture/aryan_91532/2025-07-07_143610_SBK20",
            # "gs://induction-labs-data-ext/action_capture/aryan_91532/2025-07-08_160952_VX5RU",
        ],
        "gs://induction-labs/passive_data/2025-08-11/actionblock_10",
        # max_video_files=10,
        # num_processes=1,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
