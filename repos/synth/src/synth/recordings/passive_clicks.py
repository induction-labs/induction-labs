from __future__ import annotations

import base64
import datetime
import io
import json
import multiprocessing
import re
import threading
import traceback
from collections.abc import Callable
from functools import wraps
from multiprocessing import Manager, Process
from multiprocessing.managers import ListProxy
from queue import Queue

import dotenv
import pandas as pd
import PIL
import PIL.Image
from PIL import Image
from pydantic import BaseModel
from smart_open import open as smart_open
from synapse.utils.logging import configure_logging, logging
from synth.recordings.action_models import (
    Action,
    ClickAction,
    DragAction,
    LeftDoubleAction,
    Point,
    RightSingleAction,
)
from synth.recordings.gemini_process import (
    DEFAULT_MODEL,
    RecordingMetadata,
    SaveAction,
    TrainSample,
    call_model,
    chunkify,
    combine_actions,
    draw_red_circle,
    get_actions,
    get_frames_at_timestamps,
    get_target_resolution,
    get_timesteps_range,
    image_content,
    load_metadata,
    parse_actions,
    pil_to_bytes,
    process_worker,
    progress_listener,
    pyd_model_to_response_format,
    save_and_get_metadata,
    segment_actions_by_time_gaps,
    text_content,
    transform_action_coords_list,
)

dotenv.load_dotenv()


def pil_from_b64(s: str) -> Image.Image:
    """
    Decode a base64-encoded image string (optionally a data: URL) into a PIL Image.
    - Strips any 'data:*;base64,' prefix
    - Removes whitespace/newlines
    - Fixes missing padding
    Raises:
        binascii.Error on invalid base64
        PIL.UnidentifiedImageError if decoded bytes aren't a valid image
    """
    # Strip data URL prefix if present
    if s.startswith("data:"):
        s = s.split(",", 1)[1]

    # Remove whitespace/newlines
    s = re.sub(r"\s+", "", s)

    # Fix missing '=' padding
    pad = (-len(s)) % 4
    if pad:
        s += "=" * pad

    # Decode (try strict, fall back to urlsafe)
    try:
        raw = base64.b64decode(s, validate=True)
    except Exception:
        raw = base64.urlsafe_b64decode(s)

    img = Image.open(io.BytesIO(raw))
    img.load()  # force decoding now (throws if corrupt)
    return img


logger = configure_logging(__file__, logging.INFO)

_COORD_RE = re.compile(
    r"start_box\s*=\s*['\"]?\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)['\"]?"
)


def parse_point(s: str) -> Point:
    """
    Extract the first '(x,y)' pair from a string like "start_box='(762,0)'" and
    return it as a Point. Raises ValueError if no pair is found.
    """
    m = _CORD_RE.search(s) if (_CORD_RE := _COORD_RE) else None  # local alias for speed
    if not m:
        raise ValueError(f"Could not find a coordinate like '(x,y)' in {s!r}")
    x, y = map(int, m.groups())
    return Point(x=x, y=y)


def load_item(source_dir: str, attempt_id: str, index: int) -> SaveAction:
    source_dir = source_dir.rstrip("/")
    with smart_open(f"{source_dir}/metadata/{attempt_id}.json", "r") as f:
        items = json.load(f)
        assert isinstance(items, list)
    return SaveAction.model_validate(items[index])


def reprocess_click_action(source_dir: str, df_row: dict) -> TrainSample | None:
    click_action = load_item(source_dir, df_row["attempt_id"], df_row["action_index"])
    frame = pil_from_b64(click_action.image)
    action_response = reprocess_mouse_action_gpt(
        frame,
        click_action,
    )
    if action_response is None:
        return None
    new_action, cost_info = action_response
    if not new_action.useful_action:
        logger.info(f"Skipping non-useful action: {click_action.action} {new_action=}")
        return None
    new_train_sample = TrainSample(
        instruction=new_action.action_instruction,
        actions=[
            SaveAction(
                action=click_action.action,
                thinking=new_action.action_reasoning,
                image=click_action.image,
                step=0,
                reward=0.0,
                frame_metadata=click_action.frame_metadata,
            )
        ],
        metadata={
            "old_metadata": df_row,
            "train_sample_id": "-".join(
                [str(df_row["attempt_id"]), str(df_row["action_index"])]
            ),
            "cost_info": cost_info,
        },
    )

    return new_train_sample


def thread_worker[T](process_fn: Callable[[str, T], TrainSample | None]):
    @wraps(process_fn)
    def worker_fn(source_dir: str, output_dir: str, df_row: T):
        """Thread worker function that processes a single video."""
        try:
            result = process_fn(source_dir, df_row)

        except Exception:
            return (None, traceback.format_exc())
        if result is None:
            return (None, None)
        metadata = save_and_get_metadata(source_dir, output_dir, result)
        return (metadata, None)  # video_path, results,

    return worker_fn


# @thread_worker
def process_click_action(
    source_dir: str, action_metadata: tuple[Action, RecordingMetadata]
) -> TrainSample | None:
    action, recording_metadata = action_metadata
    target_resolution = get_target_resolution(recording_metadata)
    frames = get_frames_at_timestamps(
        source_dir,
        timestamps=[action.timestamp, action.end_timestamp],
        recording_metadata=recording_metadata,
        target_resolution=target_resolution,
    )
    assert len(frames) == 2
    first_frame, first_frame_metadata = frames[0]
    second_frame, second_frame_metadata = frames[1]

    ui_element_identification = ui_element_identification_gpt(
        frame=second_frame,
        action=action,
        image_dimensions=(target_resolution.width, target_resolution.height),
    )
    if ui_element_identification is None:
        logger.error(
            f"Failed to identify UI element for action {action.action} at {action.timestamp}"
        )
        return None
    element_description, ui_cost_info = ui_element_identification
    if not element_description.is_distinct_element:
        logger.debug(
            f"Skipping non-distinct element for action {action.action} at {action.timestamp}"
        )
        return None

    assert element_description.element_description is not None, (
        f"Element description should not be None, got {element_description=}"
    )
    action_response = process_mouse_action_gpt(
        second_frame,
        action,
        image_dimensions=(target_resolution.width, target_resolution.height),
        element_description=element_description.element_description,
    )
    if action_response is None:
        return None
    new_action, cost_info = action_response
    b64_image = base64.b64encode(pil_to_bytes(first_frame, format="PNG")).decode(
        "utf-8"
    )
    new_train_sample = TrainSample(
        instruction=new_action.action_instruction,
        actions=[
            SaveAction(
                action=action.dump_to_text(),
                thinking=new_action.action_reasoning,
                image=b64_image,
                step=0,
                reward=0.0,
                frame_metadata=first_frame_metadata,
            )
        ],
        metadata={
            "instruction_cost_info": cost_info,
            "ui_element_identification_cost_info": ui_cost_info,
            "ui_element_description": element_description.element_description,
            "action_timestamp": action.timestamp,
            "action_end_timestamp": action.end_timestamp,
            "second_frame_metadata": second_frame_metadata,
        },
    )

    return new_train_sample


def click_action_thread_worker(
    source_dir: str, output_dir: str, df_row: tuple[Action, RecordingMetadata]
):
    """Thread worker function that processes a single video."""
    try:
        result = process_click_action(source_dir, df_row)

    except Exception:
        return (None, traceback.format_exc())
    if result is None:
        return (None, None)
    metadata = save_and_get_metadata(source_dir, output_dir, result)
    return (metadata, None)  # video_path, results,


class MouseActionResponse(BaseModel):
    useful_action: bool
    action_instruction: str
    action_reasoning: str


def crop_centered_at_cursor(
    img: Image.Image,
    cursor_xy: tuple[int, int],
    crop_size: tuple[int, int] = (854, 480),
) -> Image.Image:
    """
    Return a crop of size `crop_size` (default 854x480) centered at `cursor_xy`,
    clamped so the crop stays fully inside `img`.

    - img: 1080p PIL image (typically 1920x1080).
    - cursor_xy: (x, y) pixel coords in image space.
    - crop_size: (width, height), defaults to 480p 16:9 (854x480).
    """
    cw, ch = crop_size
    W, H = img.size
    x, y = cursor_xy

    if cw > W or ch > H:
        raise ValueError(f"Requested crop {cw}x{ch} larger than image {W}x{H}.")

    # Clamp center if it's outside the image
    x = max(0, min(W - 1, int(x)))
    y = max(0, min(H - 1, int(y)))

    # Compute top-left so that the crop is centered, then clamp to image bounds
    left = max(0, min(W - cw, x - cw // 2))
    top = max(0, min(H - ch, y - ch // 2))

    # Box: (left, upper, right, lower)
    box = (left, top, left + cw, top + ch)
    return img.crop(box)


class UIElementIdentification(BaseModel):
    is_distinct_element: bool
    element_description: str | None = None


def ui_element_identification_gpt(
    frame: PIL.Image.Image,
    action: Action,
    image_dimensions: tuple[int, int],
    model_name: str = DEFAULT_MODEL,
) -> tuple[UIElementIdentification, dict] | None:
    point = get_point_from_click_action(action)
    frame_with_circle = draw_red_circle(frame, point)
    cropped = crop_centered_at_cursor(
        frame_with_circle, (point.x, point.y), crop_size=(320, 240)
    )
    content = []
    content.append(
        text_content(
            """
# Instruction
You will be given a two versions of the same screenshot. You're job is to identify the UI element the cursor is hovering over in the screenshot. 
The first image is in grayscale with the cursor highlighted with a red circle. This image is to help you locate the element.
The second is the original with no red circle and full color. 


Your have three tasks:
1. Locate the cursor.
2. Determine if the mouse action interacts with a distinct, identifiable UI element.
3. If the cursor is on a UI element, describe the element and its location in the screenshot.

## Task 1: Locate the Cursor

First, use the first image to LOCATE THE CURSOR IN THE RED CIRCLE. Take note of what is around it so you can find it in the second image.
If you cannot find the cursor, simply return `{is_distinct_element: false}` and you are finished.

## Task 2: Decide whether there is a distinct, identifiable UI element.

Then, cross reference with the second image and the location information from the first image to determine what is the element in the red circle. Look for similar visual landmarks between the two images around the cursor. 
Note that the red circle is not part of the original screenshot, it is added to help you focus on the clicked point. DO NOT reference the red circle in your response.
Mark `is_distinct_element: true` **ONLY** when the cursor is clearly directed at a specific, identifiable UI element from the following list:

### Acceptable UI Elements
- Menus, including dropdown menus, context menus, bookmark menus and color pickers
- Icons of any kind
- Buttons, including text buttons, image buttons, and toggle buttons
- Sliders, including volume sliders, brightness sliders, and scrollbars
- NON-EMPTY Spreadsheet cells (only if they include content). 

Mark `is_distinct_element: false` in all other cases, including but not limited to:

### Unacceptable UI Elements
- Empty space, such as blank areas of the screen or whitespace
- Clicking a browser tab at the top of the screen
- Clicking any application in the bottom taskbar
- Any text that is not part of a button or menu
- Empty spreadsheet cells (cells with no content)

## Task 3: Describe the UI element.
If you marked the cursor as on a distinct UI element, write a clear, concise description of the element and its location in the screenshot.
For example:
`The "Kitchen" option from the Category dropdown in the top left corner of the screen.`""".strip()
        )
    )
    content.append(
        text_content(
            f"""## Content:
The mouse action is {action.dump_to_text()}. The screenshot dimensions are {image_dimensions[0]}x{image_dimensions[1]} pixels.

Image with red circle added:
    """.strip()
        )
    )
    content.append(image_content(pil_to_bytes(cropped, format="PNG")))
    content.append(text_content("\nOriginal screenshot with no red circle:"))
    content.append(image_content(pil_to_bytes(frame, format="PNG")))

    response_format = pyd_model_to_response_format(UIElementIdentification)
    try:
        model_response_text, cost_info = call_model(
            model_name=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            response_format=response_format,
        )

    except Exception as e:
        logger.error(f"Error generating UI element identification: {e}")
        return None

    response = UIElementIdentification.model_validate_json(model_response_text)
    return response, cost_info


def reprocess_mouse_action_gpt(
    frame: PIL.Image.Image,
    original_action: SaveAction,
    model_name: str = DEFAULT_MODEL,
) -> tuple[MouseActionResponse, dict] | None:
    """
    Generate an instruction based on the first frame and the actions.

    Returns:
        Tuple of (response_text, instruction, cost_info) or None if failed
    """
    point = parse_point(original_action.action)
    frame_with_circle = draw_red_circle(frame, point)

    # Build interleaved content with text and images
    content = []

    # Add final instruction
    content.append(
        {
            "type": "text",
            "text": """
# Instruction
You will be given a screenshot of a user's mouse action, along with their reasoning for taking this action.

You then have three tasks:
1. Decide whether the user is clicking on a distinct, identifiable UI element.
2. If the mouse action is a useful action, write a simple instruction that would result in the mouse action shown in the screenshot.
3. In addition, if the mouse action is useful, rewrite the user's reasoning to focus only on the present action.

## Task 1: Decide whether the mouse action is useful.
Mark `useful_action: true` **ONLY** when the user's click or drag action is clearly directed at a specific, identifiable UI element, such as:
- Clicking a button, link, or menu item
- Selecting a text field or checkbox
- Dragging an item to a specific location

## Task 2: Write a *simple* instruction that would result in the behaviour shown in the screenshots.
If you marked the action as useful, write a clear, concise instruction that an AI assistant could follow to replicate the observed behavior. Good instructions should be unambiguous and limited to a single sentence of a few words. The instruction should describe *what to do*, not *how to do it*, for example "Change the orientation to landscape" is better than "Click the button in the top right corner to change the orientation to landscape."

If the action is not useful, simply write `action_instruction: "NOT USEFUL."` followed by a short explanation of why it is not useful.


## Task 3: Rewrite the user's reasoning.
If you marked the action as useful, rewrite the user's reasoning to focus **only** on the present action. Remove any references to past actions or future plans. When referencing elements on the screen, include their position in the screenshot, e.g., "the button in the top right corner" or "the text field in the middle of the screen". 
The user's reasoning should loosely follow the format (feel free to reword, but keep the structure):

```
My goal is [briefly restate the goal of the action]. I notice that [describe the relevant elements on the screen, including their position]. Therefore, I [describe the action taken, e.g., "clicked the button in the top right corner"].

Here's an example of the style I'm looking for:
### Instruction: Change the calculation mode to "manual" in excel.>

### Reasoning:
My current goal is to select the "manual" calculation mode in Excel. I noticed that the dropdown menu for the calculation options is already open in the top right of the screen, and it includes the "Manual" option that I need. This is exactly what I was looking for, so I'll go ahead and click on it to switch to manual calculation mode.

If the action is not useful, simply write `action_reasoning: "NOT USEFUL."
`.
""".strip(),
        }
    )
    content.append(
        text_content(
            f"""# Content:
The user's original reasoning for their action is: 
{original_action.thinking}
The user's original action is: 
{original_action.action}

The screenshot of the user's action is below. Two versions are provided: the original and a version with a red circle around the clicked point. Note that the red circle is not part of the original screenshot, it is added to help you focus on the clicked point. DO NOT reference the red circle in your response.

Original screenshot:
"""
        )
    )
    content.append(image_content(pil_to_bytes(frame, format="PNG")))
    content.append(
        text_content("\nScreenshot with red circle around the clicked point:")
    )
    content.append(image_content(pil_to_bytes(frame_with_circle, format="PNG")))

    response_format = pyd_model_to_response_format(MouseActionResponse)
    try:
        model_response_text, cost_info = call_model(
            model_name=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            response_format=response_format,
        )

    except Exception as e:
        logger.error(f"Error generating video instruction: {e}")
        return None
    response = MouseActionResponse.model_validate_json(model_response_text)

    return response, cost_info


def get_point_from_click_action(action: Action) -> Point:
    assert isinstance(
        action.action, ClickAction | LeftDoubleAction | RightSingleAction | DragAction
    ), f"Unsupported action type: {action.action.__class__.__name__}"
    point: Point
    if isinstance(action.action, ClickAction | LeftDoubleAction | RightSingleAction):
        point = action.action.point
    else:
        assert isinstance(action.action, DragAction)
        point = action.action.start_point
    return point


class MouseInstructionResponse(BaseModel):
    action_instruction: str
    action_reasoning: str


def process_mouse_action_gpt(
    frame: PIL.Image.Image,
    action: Action,
    image_dimensions: tuple[int, int],
    element_description: str,
    model_name: str = DEFAULT_MODEL,
) -> tuple[MouseInstructionResponse, dict] | None:
    """
    Generate an instruction based on the first frame and the actions.

    Returns:
        Tuple of (response_text, instruction, cost_info) or None if failed
    """
    point = get_point_from_click_action(action)

    frame_with_circle = draw_red_circle(frame, point)

    # Build interleaved content with text and images
    content = []

    # Add final instruction
    content.append(
        {
            "type": "text",
            "text": """
# Instruction
You will be given a user's mouse action, along with the screenshot taken right before the action and a description of the UI element that the user is interacting with.

You then have two tasks:
1. Write a simple instruction that would result in the mouse action shown in the screenshot.
2. In addition, write the user's reasoning as they perform the action.


## Task 1: Write a *simple* instruction that would result in the behaviour shown in the screenshots.
Write a clear, concise instruction that an AI assistant could follow to replicate the observed behavior.
Good instructions should be unambiguous and limited to a single sentence of a few words. The instruction should describe *what to do*. Do NOT describe *how to do it* or *where the element is*.
For example "Change the orientation to landscape" is better than "Click the button in the top right corner to change the orientation to landscape", because the user needs to figure out how to complete the task themselves.

## Task 2: Write the user's reasoning.
Write the user's thought process as they use the instruction and information on screen to deduce the action they need to take. In a first-person, present-tense inner voice, explain why this specific action is the right move to progress or accomplish the task. Be concrete and insightful—reference on-screen cues. When referencing elements on the screen, include their position in the screenshot, e.g., "the button in the top right corner" or "the dropdown is open in the middle of the screen".

The user's reasoning should first briefly restate what they need to accomplish. Then describe the relevant elements on the screen, including their position. Finally, deduce which action to take based on the current state of the screen.

Here's an example of the style I'm looking for:
### Instruction: Change the calculation mode to "manual" in excel.

### Reasoning:
I need to select the "manual" calculation mode in Excel. I noticed that the dropdown menu for the calculation options is already open in the top right of the screen, and it includes the "Manual" option that I need. This is exactly what I was looking for, so I'll go ahead and click on it to switch to manual calculation mode.
""".strip(),
        }
    )
    content.append(
        text_content(
            f"""# Content:
The user's action is: 
{action.dump_to_text()}
The screenshot dimensions are {image_dimensions[0]}x{image_dimensions[1]} pixels.
The element description is:
{element_description}
The screenshot taken prior to the user's action is below. 
""".strip()
        )
    )
    content.append(image_content(pil_to_bytes(frame, format="PNG")))
    content.append(
        text_content(
            f"""
The next image is the same screenshot with the point ({point.x, point.y}) highlighted with a red circle to help you focus on the clicked point. Note that the red circle is not part of the original screenshot, it is added to help you focus on the clicked point. DO NOT reference the red circle in your response.
""".strip()
        )
    )

    content.append(image_content(pil_to_bytes(frame_with_circle, format="PNG")))

    response_format = pyd_model_to_response_format(MouseInstructionResponse)
    try:
        model_response_text, cost_info = call_model(
            model_name=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            response_format=response_format,
        )

    except Exception as e:
        logger.error(f"Error generating video instruction: {e}")
        return None
    response = MouseInstructionResponse.model_validate_json(model_response_text)

    return response, cost_info


def reprocess_clicks(
    source_folder: str,
    output_dir: str,
    num_processes: int | None = None,
    threads_per_process: int = 12,
    max_video_files: int | None = None,
):
    source_folder = source_folder.rstrip("/")
    df_only_click = pd.read_json(
        f"{source_folder}/samples_actions_explode_only_click.jsonl", lines=True
    )
    process_args = [
        (source_folder, output_dir, i) for i in df_only_click.to_dict(orient="records")
    ]
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    if max_video_files is not None:
        process_args = process_args[:max_video_files]
        logger.info(f"Limiting to {max_video_files} video files")

    print(
        f"Processing {len(process_args)} videos using {num_processes} processes x {threads_per_process} threads..."
    )

    # Use multiprocessing + threading pattern
    manager = Manager()
    update_queue: Queue[tuple[dict | None, str | None]] = manager.Queue()
    results_collector: ListProxy[dict] = (
        manager.list()
    )  # shared list to collect results

    # Start progress listener thread
    listener = threading.Thread(
        target=progress_listener,
        args=(len(process_args), update_queue, results_collector, output_dir),
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
            args=(reprocess_click_action, threads_per_process, update_queue, chunk),
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


def parse_actions_with_error_handling(
    raw_actions: list[dict], source_dir: str | None = None
) -> list[Action]:
    try:
        return parse_actions(raw_actions)
    except Exception as e:
        logger.error(f"Error parsing actions {source_dir=}: {e}")
        logger.error(traceback.format_exc())
        return []


def process_clicks_from_raw(
    source_folders: list[str],
    output_dir: str,
    num_processes: int | None = None,
    threads_per_process: int = 6,
    max_video_files: int | None = None,
):
    source_folders = [(source.rstrip("/")) for source in source_folders if source]

    raw_action_sets = {k: get_actions(k) for k in source_folders}

    video_metadatas = {k: load_metadata(k) for k in source_folders}
    target_resolutions = {
        k: get_target_resolution(v) for k, v in video_metadatas.items()
    }

    action_sets = {
        k: transform_action_coords_list(
            combine_actions(parse_actions_with_error_handling(v, k)),
            target_resolutions[k],
            video_metadatas[k],
        )
        for k, v in raw_action_sets.items()
    }
    action_segments = [
        (source_dir, actions)
        for source_dir, actions_lists in action_sets.items()
        for actions in segment_actions_by_time_gaps(actions_lists)
    ]

    for _, actions in action_segments:
        action_timestamps = get_timesteps_range(actions)[
            :-1
        ]  # exclude last end timestamp
        assert len(action_timestamps) == len(actions)
        for action, timestamp in zip(actions, action_timestamps, strict=True):
            original_start = action.timestamp
            action.timestamp = max(timestamp, original_start - 0.5)
            action.end_timestamp = original_start

    mouse_actions = (
        (source_dir, action)
        for source_dir, actions in action_segments
        for action in actions
        if isinstance(
            action.action,
            ClickAction | LeftDoubleAction | RightSingleAction | DragAction,
        )
    )

    process_args = [
        (source_folder, output_dir, (action, video_metadatas[source_folder]))
        for source_folder, action in mouse_actions
    ]
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Shuffle the process_args to ensure random distribution across processes
    import random

    random.seed(42)  # for reproducibility
    random.shuffle(process_args)
    if max_video_files is not None:
        process_args = process_args[:max_video_files]
        logger.info(f"Limiting to {max_video_files} video files")

    print(
        f"Processing {len(process_args)} videos using {num_processes} processes x {threads_per_process} threads..."
    )

    # Use multiprocessing + threading pattern
    manager = Manager()
    update_queue: Queue[tuple[dict | None, str | None]] = manager.Queue()
    results_collector: ListProxy[dict] = (
        manager.list()
    )  # shared list to collect results

    # Start progress listener thread
    listener = threading.Thread(
        target=progress_listener,
        args=(len(process_args), update_queue, results_collector, output_dir, 500),
        daemon=True,
    )
    listener.start()

    # Split video args among processes
    chunks = chunkify(process_args, num_processes)
    print("Starting processes...")
    print(f"Saving to {output_dir}/samples_*.jsonl files")

    processes = []
    for chunk in chunks:
        if not chunk:
            continue
        p = Process(
            target=process_worker,
            args=(click_action_thread_worker, threads_per_process, update_queue, chunk),
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
    print(
        f"Processing completed. Results saved incrementally in batches to {output_dir}/samples_*.jsonl files"
    )


def main():
    dataset_name = "reprocess_all_no_loading"
    output_dir = f"gs://induction-labs/passive_data/smooth_brain_clicks/{datetime.datetime.now(datetime.UTC):%Y-%m-%d}/{dataset_name}-{datetime.datetime.now(datetime.UTC):%H-%M-%S}"
    base_path = "gs://induction-labs/passive_data/2025-08-14/reprocess_all_no_loading-04-30-15/".rstrip(
        "/"
    )
    reprocess_clicks(
        source_folder=base_path,
        output_dir=output_dir,
    )
    print(f"Results saved to {output_dir}/samples.jsonl")


def main2():
    dataset_name = "clicks_prob_good_cropped"
    output_dir = f"gs://induction-labs/passive_data/smooth_brain_clicks/{datetime.datetime.now(datetime.UTC):%Y-%m-%d}/{dataset_name}-{datetime.datetime.now(datetime.UTC):%H-%M-%S}"
    source_folders = [
        # Kunal
        # "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-18_101735_MIELX",
        # "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-19_121221_3A4PN",
        # "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-18_133213_FN0VR",
        # "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-18_172629_2RBSA",
        # "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-19_125156_PQS8V",
        # "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-19_150406_A27L7",
        # "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-21_111019_IF3S0",
        # "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-21_172805_4KV69",
        # "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-21_195634_CHHBR",
        # # Mahdi_lumio
        # "gs://induction-labs-data-ext/action_capture/Mahdi_lumio/2025-07-19_202437_BSGP6",
        # "gs://induction-labs-data-ext/action_capture/Mahdi_lumio/2025-07-19_202545_R9AN9",
        # "gs://induction-labs-data-ext/action_capture/Mahdi_lumio/2025-07-21_145524_Y5HX2",
        # "gs://induction-labs-data-ext/action_capture/Mahdi_lumio/2025-07-24_141207_3TWWR",
        # "gs://induction-labs-data-ext/action_capture/Mahdi_lumio/2025-08-06_155356_WIGKJ",
        # Tirador
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-06_213528_7A3GZ/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-07_004523_YZCKT/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-07_151909_EM1ZD/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-07_181227_8C74U/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-07_204719_ST5T0/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-07_211120_94XGE/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-08_182312_Y56YF/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-08_214414_4TT3Z/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-09_142027_DED1W/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-09_144647_ECLHS/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-09_155550_ZMYHV/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-09_193418_SP3LZ/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-10_162405_26O9J/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-11_001850_5VUNA/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-13_195531_RSWMW/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-14_182906_WVWG5/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-14_220158_LBGGM/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-15_165347_KP6YW/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-15_195134_KWILK/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-16_212920_6DWYV/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-17_163921_9Z25E/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-19_144843_SYMA8/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-21_204350_XA49Z/",
        "gs://induction-labs-data-ext/action_capture/Tirador/2025-07-22_224422_XFW5C/",
        # Sand
        "gs://induction-labs-data-ext/action_capture/sand/2025-07-18_135456_RWJJN/",
        "gs://induction-labs-data-ext/action_capture/sand/2025-07-19_144144_8DI6M/",
        "gs://induction-labs-data-ext/action_capture/sand/2025-07-25_161125_1GJM0/",
        "gs://induction-labs-data-ext/action_capture/sand/2025-07-26_135348_88OVA/",
        "gs://induction-labs-data-ext/action_capture/sand/2025-07-26_144717_3IZP6/",
        "gs://induction-labs-data-ext/action_capture/sand/2025-07-27_203922_EPEL2/",
        "gs://induction-labs-data-ext/action_capture/sand/2025-07-28_195826_O03E3/",
        "gs://induction-labs-data-ext/action_capture/sand/2025-07-29_000123_NYUHB/",
        "gs://induction-labs-data-ext/action_capture/sand/2025-07-29_130733_029HN/",
        # Joyce
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-04_110139_B6VYF/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-06_112602_9ZEFH/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-09_160136_WVNHY/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-12_100706_2RKVJ/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-14_111643_P725G/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-14_162035_B9PM6/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-15_100419_MAESY/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-15_101847_913IO/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-15_102313_K967C/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-18_203324_YL5VM/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-27_192513_2FNA0/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-27_192852_E59D8/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-27_193551_TX1BD/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-29_193316_LSEM8/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-30_111301_8RVWD/",
        "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-31_200548_0Z5EH/",
        # Aryan
        "gs://induction-labs-data-ext/action_capture/aryan_91532/2025-07-07_170814_A2QD2",
        "gs://induction-labs-data-ext/action_capture/aryan_91532/2025-07-07_143610_SBK20",
        # Jarry
        "gs://induction-labs-data-ext/action_capture/Jarry/2025-07-07_002920_0SPCN",
        "gs://induction-labs-data-ext/action_capture/Jarry/2025-08-10_121140_Q4KI9",
        "gs://induction-labs-data-ext/action_capture/Jarry/2025-08-11_185116_OXFUY",
        "gs://induction-labs-data-ext/action_capture/Jarry/2025-08-11_184643_WU8RW",
        "gs://induction-labs-data-ext/action_capture/Jarry/2025-08-12_224139_V8X0T",
        # Excel guy
        # "gs://induction-labs-data-ext-passive-mangodesk/action_capture/JayeshJadhav-Lumio/2025-08-13_200130_64GMN",
    ]
    process_clicks_from_raw(
        source_folders=source_folders, output_dir=output_dir, max_video_files=None
    )
    print(f"Results saved to {output_dir}/samples.jsonl")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main2()
