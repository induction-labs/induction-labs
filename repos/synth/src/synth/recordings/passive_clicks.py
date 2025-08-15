from __future__ import annotations

import base64
import datetime
import io
import json
import multiprocessing
import re
import threading
import traceback
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
from synth.recordings.gemini_process import (
    DEFAULT_MODEL,
    SaveAction,
    TrainSample,
    call_model,
    chunkify,
    draw_red_circle,
    image_content,
    pil_to_bytes,
    process_worker,
    progress_listener,
    pyd_model_to_response_format,
    save_and_get_metadata,
    text_content,
)
from synth.recordings.parse_actions import (
    Point,
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
    action_response = get_mouse_action(
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
                thinking=new_action.new_reasoning,
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


class MouseActionResponse(BaseModel):
    useful_action: bool
    action_instruction: str
    new_reasoning: str


def get_mouse_action(
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
If you marked the action as useful, write a clear, concise instruction that an AI assistant could follow to replicate the observed behavior. 
Good instructions should be unambiguous and limited to a single sentence of a few words, for example "Change the orientation to landscape".

If the action is not useful, simply write `action_instruction: "NOT USEFUL."` followed by a short explanation of why it is not useful.


## Task 3: Rewrite the user's reasoning.
If you marked the action as useful, rewrite the user's reasoning to focus **only** on the present action. Remove any references to past actions or future plans. When referencing elements on the screen, include their position in the screenshot, e.g., "the button in the top right corner" or "the text field in the middle of the screen". 
The user's reasoning should loosely follow the format (feel free to reword, but keep the structure):

```
My goal is [briefly restate the goal of the action]. I notice that [describe the relevant elements on the screen, including their position]. Therefore, I [describe the action taken, e.g., "clicked the button in the top right corner"].

Here's an example of the style I'm looking for:
### Instruction: Change the calculation mode to "manual" in excel.>

### Reasoning:
My current goal is to select the "manual" calculation mode in Excel. I noticed that the dropdown menu for the calculation options is already open in the top right of the screen, and it conveniently includes the "Manual" option that I need. This is exactly what I was looking for, so I'll go ahead and click on it to switch to manual calculation mode.

If the action is not useful, simply write `new_reasoning: "NOT USEFUL."
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


def thread_worker(source_dir: str, output_dir: str, df_row: dict):
    """Thread worker function that processes a single video."""
    try:
        result = reprocess_click_action(source_dir, df_row)

    except Exception:
        return (None, traceback.format_exc())
    if result is None:
        return (None, None)
    metadata = save_and_get_metadata(source_dir, output_dir, result)
    return (metadata, None)  # video_path, results, error


def process_clicks(
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
            args=(thread_worker, threads_per_process, update_queue, chunk),
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


def main():
    dataset_name = "reprocess_all_no_loading"
    output_dir = f"gs://induction-labs/passive_data/smooth_brain_clicks/{datetime.datetime.now(datetime.UTC):%Y-%m-%d}/{dataset_name}-{datetime.datetime.now(datetime.UTC):%H-%M-%S}"
    base_path = "gs://induction-labs/passive_data/2025-08-14/reprocess_all_no_loading-04-30-15/".rstrip(
        "/"
    )
    process_clicks(
        source_folder=base_path,
        output_dir=output_dir,
    )
    print(f"Results saved to {output_dir}/samples.jsonl")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
