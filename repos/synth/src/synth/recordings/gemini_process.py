from __future__ import annotations

import base64
import json
import logging
import multiprocessing
import re
import secrets
import string
import tempfile
import threading
import traceback
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Protocol

import av
import dotenv
import fsspec as fs
import gcsfs
import pandas as pd
import tqdm
from google import genai
from google.genai import types
from pydantic import BaseModel
from smart_open import open
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

# Set up logging
logger = logging.getLogger(__name__)

gem_client = genai.Client()


# Make a file a ./a.txt


def load_actions(path: Path) -> list[dict]:
    """
    Load actions from a given path.
    """
    actions: list[dict] = []
    with fs.open(path, "r") as f:
        for line in f:
            action = json.loads(line)
            actions.append(action)
    return actions


# Example:
# s, e = get_video_time_bounds_pyav("example.mp4")
# print(s, e)
# Example:
# s, e = get_video_time_bounds("example.mp4")
# print(s, e)


def is_close(a: Point, b: Point, tol: float = 10) -> bool:
    return abs(a.x - b.x) <= tol and abs(a.y - b.y) <= tol


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


MIN_ACTIONS_THRESHOLD = 6
SS_DELAY = 0.2
BEFORE_ACTION_BUFFER = 0.05


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
    thinking: str = ""


class ScreenInfo(BaseModel):
    video_width: int
    video_height: int
    logical_pixel_ratio: float
    logical_pixel_width: int
    logical_pixel_height: int


class RecordingMetadata(BaseModel):
    timestamp: str
    username: str
    screen_info: ScreenInfo
    video_segment_buffer_length: int


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
        model_response = gem_client.models.generate_content(
            model="gemini-2.5-pro", contents=contents
        )
        assert model_response.candidates, "No candidates returned from model"
        assert model_response.candidates[0].content
        assert model_response.candidates[0].content.parts, "No content parts returned"
        model_response_text = model_response.candidates[0].content.parts[0].text
        thinking_texts.append(model_response_text)
        old_turns.append(f"{model_response_text}\n{action.dump_to_text()}")
    return thinking_texts


class SelectActionPolicy(Protocol):
    @abstractmethod
    def select_actions(self, actions: list[Action]) -> list[list[Action]]:
        pass


class DefaultSelectActionPolicy(SelectActionPolicy):
    def select_actions(self, actions: list[Action]) -> list[list[Action]]:
        """
        Select actions from the list of actions in ranges of 5.
        Returns multiple ranges of 5 actions until there are no more.
        """
        non_scroll_actions = [
            action for action in actions if action.action.action_type != "scroll"
        ]
        if len(non_scroll_actions) < MIN_ACTIONS_THRESHOLD:
            print(
                f"Not enough actions to process: {len(non_scroll_actions)} < {MIN_ACTIONS_THRESHOLD}"
            )
            # If there are not enough actions, return an empty list
            # or handle it as needed (e.g., return a default action)
            return []

        # Create ranges of 5 actions until there are no more
        action_ranges = []
        for i in range(0, len(actions), 5):
            action_range = actions[i : i + 5]
            if len(action_range) >= 5:  # Only add if we have enough actions
                action_ranges.append(action_range)

        return action_ranges


def process_single_action_range(
    video_container,
    video_metadata,
    selected_actions: list[Action],
    metadata: RecordingMetadata,
    video_path: str,
) -> TrainSample | None:
    """
    Process a single range of actions to create one TrainSample.

    Args:
        video_container: Open video container
        video_metadata: Video metadata
        selected_actions: List of actions to process (should be 5 actions)
        metadata: Recording metadata
        video_path: Path to the video file

    Returns:
        TrainSample or None if processing failed
    """

    before_timestamp, action_end_timestamps = get_timesteps_range(selected_actions)

    resize_h, resize_w = smart_resize(
        video_metadata.resolution.height,
        video_metadata.resolution.width,
        min_pixels=0,
        max_pixels=resolution_1080p.pixels,
    )
    output_resolution = VideoResolution(width=resize_w, height=resize_h)

    # Transform action coordinates from logical pixel dimensions to resized dimensions
    logical_width = metadata.screen_info.logical_pixel_width
    logical_height = metadata.screen_info.logical_pixel_height

    # Transform all actions to match the resized video dimensions

    actions_to_process = [
        transform_action_coordinates(
            action, logical_width, logical_height, resize_w, resize_h
        )
        for action in selected_actions
    ]

    buffer_src, buffer_sink, filter_graph = get_resize_filter(
        video_container.streams.video[0].format,
        video_metadata.resolution,
        output_resolution,
        video_metadata.time_base,
    )

    frames = extract_frames_by_pts_from_container(
        video_container,
        [before_timestamp, *action_end_timestamps],
        buffer_src,
        buffer_sink,
    )
    assert len(frames) == len(action_end_timestamps) + 1, (
        f"Expected {len(action_end_timestamps) + 1} frames, "
        f"but got {len(frames)} for video {video_path}"
    )

    frame_buffers = [pil_to_bytes(frame, format="PNG") for frame in frames]
    print(f"Extracted {len(frames)} frames for video {video_path}")

    try:
        video_instruction = get_video_instruction(
            frame_buffers[0],
            frame_buffers[1:],
            actions_to_process,
        )
        if video_instruction is None:
            logger.warning(f"Failed to generate video instruction for {video_path}")
            return None
        instruction_text, instruction = video_instruction
    except Exception as e:
        logger.error(
            f"Error generating video instruction for {video_path}: {e}", exc_info=True
        )
        return None

    try:
        # print(f"Generated instruction: {instruction_text}")
        thinking_texts = get_thinking_texts(
            frame_buffers[0],
            frame_buffers[1:],
            actions_to_process,
            instruction_text,
        )
    except Exception as e:
        logger.error(
            f"Error generating thinking texts for {video_path}: {e}", exc_info=True
        )
        return None
    b64_images = [base64.b64encode(bytes).decode("utf-8") for bytes in frame_buffers]
    save_actions = [
        SaveAction(
            step=i,
            image=b64_images[i],
            action=action.dump_to_text(),
            text=f"{thinking_texts[i]}\nAction: {action.dump_to_text()}",
            thinking=thinking_texts[i],
        )
        for i, action in enumerate(actions_to_process)
    ]
    save_actions.append(
        SaveAction(
            step=len(actions_to_process),
            image=b64_images[-1],
            action="finished()",
            text="Action: finished()",
            thinking="",
        )
    )

    train_sample = TrainSample(
        actions=save_actions,
        instruction=instruction,
        metadata={
            "video_path": video_path,
            "start_time": float(video_metadata.start_pts * video_metadata.time_base),
            "end_time": float(video_metadata.end_pts * video_metadata.time_base),
            "instruction_text": instruction_text,
            "original_resolution": {
                "width": video_metadata.resolution.width,
                "height": video_metadata.resolution.height,
            },
            "output_resolution": {
                "width": output_resolution.width,
                "height": output_resolution.height,
            },
            "logical_pixel_ratio": metadata.screen_info.logical_pixel_ratio,
        },
    )
    return train_sample


def video_process(
    video_path: str,
    actions_df: pd.DataFrame,
    select_policy: SelectActionPolicy,
    metadata: RecordingMetadata,
) -> list[TrainSample]:
    train_samples = []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        download_video_with_ffmpeg_copy(
            source_path=video_path,
            dest_path=f.name,
            ffmpeg_path="/nix/store/q7j5awbg80d38p9my5b5zgn0xadgvbmb-ffmpeg-7.1.1-bin/bin/ffmpeg",
        )
        with av.open(f.name) as video_container:
            video_metadata = VideoMetadataFetcher.get_video_metadata(
                video_container.streams.video[0]
            )
            start_timestamp = float(video_metadata.start_pts * video_metadata.time_base)
            end_timestamp = float(video_metadata.end_pts * video_metadata.time_base)
            filtered_actions_df = actions_df[
                (actions_df["start_timestamp"] >= start_timestamp)
                & (actions_df["end_timestamp"] <= end_timestamp)
            ]

            filtered_actions: list[Action] = filtered_actions_df["action"].tolist()
            selected_action_ranges = select_policy.select_actions(filtered_actions)
            if len(selected_action_ranges) == 0:
                return []

            print(
                f"Processing {len(selected_action_ranges)} action ranges for video {video_path}"
            )

            # Process each action range to create multiple train samples
            for i, selected_actions in enumerate(selected_action_ranges):
                import random

                # 25% chance
                if random.random() < 0.25:
                    selected_actions[-1] = Action(
                        action=FinishedAction(),
                        timestamp=selected_actions[-2].end_timestamp
                        + SS_DELAY
                        + BEFORE_ACTION_BUFFER,
                        end_timestamp=selected_actions[-2].end_timestamp
                        + SS_DELAY
                        + BEFORE_ACTION_BUFFER,
                    )
                print(f"Processing action range {i + 1}/{len(selected_action_ranges)}")

                try:
                    train_sample = process_single_action_range(
                        video_container,
                        video_metadata,
                        selected_actions,
                        metadata,
                        video_path,
                    )

                    if train_sample is not None:
                        train_samples.append(train_sample)
                    else:
                        logger.warning(
                            f"Action range {i + 1}/{len(selected_action_ranges)} returned None for video {video_path}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error processing action range {i + 1}/{len(selected_action_ranges)} for video {video_path}: {e}",
                        exc_info=True,
                    )
                    # Continue processing other action ranges

    return train_samples


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

        # Validate using Pydantic
        metadata = RecordingMetadata(**metadata_dict)
        return metadata

    except Exception as e:
        raise Exception(
            f"Failed to load or validate metadata from {source_dir}: {e}"
        ) from e


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


def discover_video_files(source_folders: list[str]) -> list[tuple[str, str]]:
    """
    Discover all video files in the given source folders.

    Args:
        source_folders: List of folder paths (e.g., gs://bucket/folder)

    Returns:
        List of tuples (video_path, source_dir) where video_path is the full path
        to the video file and source_dir is the folder containing it
    """
    video_files = []
    fs = gcsfs.GCSFileSystem()

    for source_dir in source_folders:
        print(f"Discovering videos in {source_dir}")

        try:
            # Remove gs:// prefix for gcsfs
            gs_path = source_dir.replace("gs://", "")

            # List all files in the directory
            all_files = fs.ls(gs_path)

            # Filter for screen_capture videos and sort them
            video_pattern = re.compile(r"/screen_capture_(\d+)\.mp4$")
            video_files_in_dir = []

            for f in all_files:
                match = video_pattern.search(f)
                if match:
                    video_files_in_dir.append(f)

            # Sort by video number
            def get_video_number(filename):
                match = video_pattern.search(filename)  # noqa: B023
                return int(match.group(1)) if match else 0

            video_files_in_dir.sort(key=get_video_number)

            for video_file in video_files_in_dir:
                # Convert back to gs:// format
                video_path = f"gs://{video_file}"
                video_files.append((video_path, source_dir))

        except Exception as e:
            print(f"Error discovering videos in {source_dir}: {e}")
            continue

    print(f"Found {len(video_files)} video files across {len(source_folders)} folders")
    return video_files


def process_single_video(
    args: tuple[str, str, pd.DataFrame, SelectActionPolicy, str, RecordingMetadata],
) -> list[dict]:
    """
    Process a single video for multiprocessing.

    Args:
        args: Tuple containing (video_path, source_dir, actions_df, select_policy, output_dir, metadata)

    Returns:
        List of dictionaries with train sample metadata
    """
    video_path, source_dir, actions_df, select_policy, output_dir, metadata = args

    print(f"Processing video: {video_path}")
    train_samples = video_process(video_path, actions_df, select_policy, metadata)

    if not train_samples:
        return []

    results = []

    # Process each train sample
    for train_sample in train_samples:
        train_sample_id = gen_id(12)

        # Save the train sample to file
        with open(
            f"{output_dir}/metadata/{train_sample_id}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                train_sample.model_dump()["actions"], f, ensure_ascii=False, indent=0
            )
        with open(
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
        results.append(
            {
                "attempt_id": train_sample_id,
                "eval_task_id": train_sample_id,
                "actions": [s.action for s in train_sample.actions],
                "thinking": [s.thinking for s in train_sample.actions],
                "instruction": train_sample.instruction,
                "video_path": video_path,
                "trajectory_length": len(train_sample.actions),
                "source_dir": source_dir,
                "image_turns_start": 0,
                "image_turns_end": len(train_sample.actions) - 1,
                "text_turns_start": 0,
                "text_turns_end": len(train_sample.actions) - 1,
                "unmask_last_only": False,
            }
        )

    print(f"Generated {len(results)} train samples from video {video_path}")
    return results


def chunkify(lst, n):
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def thread_worker(video_args, lock):
    """Thread worker function that processes a single video."""
    try:
        results = process_single_video(video_args)
        return (video_args[0], results, None)  # video_path, results, error
    except Exception:
        return (video_args[0], [], traceback.format_exc())


def process_worker(video_args_chunk, threads_per_proc, lock, update_queue):
    """
    Runs in each separate process. Spins up a thread pool to process its chunk of videos.
    Reports completion (with error or not) back via update_queue.
    """
    with ThreadPoolExecutor(max_workers=threads_per_proc) as executor:
        futures = {
            executor.submit(thread_worker, video_args, lock): video_args
            for video_args in video_args_chunk
        }
        for fut in as_completed(futures):
            video_path, results, err = fut.result()
            # send results so main can aggregate them
            update_queue.put((video_path, results, err))


def progress_listener(total, update_queue, results_collector):
    """
    Runs in a thread in main process to consume updates and show progress bar.
    Also collects results from all processes.
    """
    pbar = tqdm.tqdm(total=total, desc="Processing videos", unit="video")
    completed = 0
    while completed < total:
        try:
            video_path, results, err = update_queue.get(timeout=1)
        except Exception:
            continue
        if err:
            tqdm.tqdm.write(f"[!] Error processing {video_path}:\n{err}")
        else:
            tqdm.tqdm.write(f"Done processing {video_path} -> {len(results)} samples")
            results_collector.extend(results)
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
    video_files = discover_video_files(source_folders)

    if not video_files:
        print("No video files found in the specified folders")
        return
    if max_video_files is not None:
        video_files = video_files[:max_video_files]
        print(f"Limiting to first {max_video_files} video files")

    # Build mappings for efficiency
    actions_dfs = {}
    metadatas = {}
    for _, source_dir in video_files:
        if source_dir not in actions_dfs:
            print(f"Loading actions for {source_dir}")
            actions_dfs[source_dir] = get_actions_df(source_dir)
            print(f"Loading metadata for {source_dir}")
            metadatas[source_dir] = load_metadata(source_dir)

    select_policy = DefaultSelectActionPolicy()

    # Prepare arguments for each video
    video_args = [
        (
            video_path,
            source_dir,
            actions_dfs[source_dir],
            select_policy,
            output_dir,
            metadatas[source_dir],
        )
        for video_path, source_dir in video_files
    ]

    # Use all CPU cores if num_processes not specified
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(
        f"Processing {len(video_files)} videos using {num_processes} processes x {threads_per_process} threads..."
    )

    # Use multiprocessing + threading pattern
    manager = Manager()
    lock = manager.Lock()  # shared across processes
    update_queue = manager.Queue()
    results_collector = manager.list()  # shared list to collect results

    # Start progress listener thread
    listener = threading.Thread(
        target=progress_listener,
        args=(len(video_files), update_queue, results_collector),
        daemon=True,
    )
    listener.start()

    # Split video args among processes
    chunks = chunkify(video_args, num_processes)
    processes = []
    for chunk in chunks:
        if not chunk:
            continue
        p = Process(
            target=process_worker,
            args=(chunk, threads_per_process, lock, update_queue),
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
            f"{output_dir}/train_samples.jsonl",
            orient="records",
            lines=True,
        )

    print(f"Saved {len(results)} train samples to {output_dir}/train_samples.jsonl")


def main() -> None:
    process_videos(
        [
            "gs://induction-labs-data-ext/action_capture/Jarry/2025-07-07_002920_0SPCN",
            "gs://induction-labs-data-ext/action_capture/aryan_91532/2025-07-07_170814_A2QD2",
            "gs://induction-labs-data-ext/action_capture/aryan_91532/2025-07-07_143610_SBK20",
            "gs://induction-labs-data-ext/action_capture/aryan_91532/2025-07-08_160952_VX5RU",
        ],
        "gs://induction-labs/passive_data/2025-08-09/jarry_aryan_data_2",
        # max_video_files=10,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
