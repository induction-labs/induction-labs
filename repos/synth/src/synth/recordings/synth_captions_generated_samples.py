from __future__ import annotations

import json
import os
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import TYPE_CHECKING, cast

import av
import dotenv
import gcsfs
import pandas as pd
import PIL
import PIL.Image
from av.container import InputContainer
from google.cloud import storage
from openai import OpenAI
from synth.recordings.action_models import Action, WaitAction
from synth.recordings.repair_dir import fix_string
from tqdm import tqdm

dotenv.load_dotenv()

client = OpenAI()
OUTPUT_PATH = "/home/jonathan_inductionlabs_com/induction-labs/repos/synth/src/synth/recordings/synth_data_annotated"

COMMON_INSTRUCTION = """
## Instruction
I'm trying to simulate a user's internal monologue while they perform computer-related tasks. Please write, in the user's voice as an internal monologue, the reasoning that would lead the user to believe that the given action was the correct one to take to complete their task.
Write in the future tense (i.e., "I'll do X") and in a simple, concise tone. It should only be a few sentences long, no Markdown. It should be written like the user's internal monologue, ONLY for the action just taken, but can allude to next steps or refer previous steps.
You will also be provided the action that the user took at the given screenshot. The internal monologue should motivate why the given specific action is a good one, and be specific and provide insight into how to solve the given task.
Don't use fancy words or refer to objects in the physical world (such as "flicking a mouse" or "since I'm already using my mouse"). Include text they typed or keyboard shortcuts pressed in the reasoning.

Make sure to lead with reasoning, then the action. Don't just start with the action.

Here's an example of the style I'm looking for:
Let me see how to add a favorites folder. First, I need to open the settings menu in the browser. I noticed there's a three-dot icon in the upper right corner of the browser window, which is the entry point for the settings menu. I'll click on it to reveal more options.
""".strip()

PROMPT_WITH_NEXT = (
    COMMON_INSTRUCTION
    + "\n"
    + """## Context
Here's the task the user has been given:
{instruction}

Here's the reasoning the user has done so far:
{old_turns}

**Here's the new action the user took that you should write the monolouge for**:
{new_action}

Only for planning purposes, here is the next action they will take:
{next_action}

Attached are screenshots of the user's screen. The first is before they took the action, the second is right after they took the action, and the third is the action they took after that.
Use the third image only to make the planning more relevant to the task. The monologue should be focused on the action just taken. For example, if it isn't obvious what the contexts of the next image are, obviously don't use them to write the monologue."""
)

PROMPT_WITHOUT_NEXT = (
    COMMON_INSTRUCTION
    + "\n"
    + """## Context
Here's the task the user has been given:
{instruction}

Here's the reasoning the user has done so far:
{old_turns}

**Here's the new action the user took that you should write the monolouge for**:
{new_action}

Attached are screenshots of the user's screen. The first is before they took the action and the second is right after they took the action."""
)

_client = storage.Client()

DATA_PATH = "/home/jonathan_inductionlabs_com/induction-labs/repos/synth/src/synth/recordings/all_trajectories"


def get_next_thought(
    instruction: str,
    old_turns: str,
    new_action: str,
    next_action: str | None,
    old_image: str,
    new_image: str,
    next_image: str | None,
) -> str:
    PROMPT = PROMPT_WITHOUT_NEXT if next_action is None else PROMPT_WITH_NEXT

    args = {
        "instruction": instruction,
        "old_turns": old_turns,
        "new_action": new_action,
    }
    if next_action is not None:
        args["next_action"] = next_action

    response = client.responses.create(
        model="o3",
        reasoning={"effort": "high"},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT.format(**args)},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{old_image}",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{new_image}",
                    },
                ]
                + (
                    [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{next_image}",
                        },
                    ]
                    if next_action is not None
                    else []
                ),
            }  # pyright: ignore[reportArgumentType]
        ],
    )

    return response


def add_wait_time(actions, gap_time=5):
    # if there's > gap_time second gap then add a wait action
    new_array = []
    for i in range(1, len(actions)):
        new_array.append(actions[i - 1])
        if actions[i].timestamp - actions[i - 1].end_timestamp >= gap_time:
            num_waits = int(
                (actions[i].timestamp - actions[i - 1].end_timestamp) // gap_time
            )
            for wait_idx in range(num_waits):
                start_timestamp = actions[i - 1].end_timestamp + wait_idx * gap_time
                wait_action = Action(
                    action=WaitAction(),
                    timestamp=start_timestamp,
                    end_timestamp=start_timestamp + gap_time,
                )
                new_array.append(wait_action)

    new_array.append(actions[-1])

    return new_array


def format_cot_with_action(chain_of_thought_with_action):
    """
    Format the chain of thought with action into a string.
    """
    if not chain_of_thought_with_action:
        return "(No reasoning yet, since this is the first turn)"
    formatted = []
    for item in chain_of_thought_with_action:
        formatted.append(f"Thinking: {item['thinking']}\nAction: {item['action']}")
    return "\n\n".join(formatted)


def get_actions(gs_path: str):
    gs_path = gs_path.replace("gs://", "")
    fs = gcsfs.GCSFileSystem(project="induction-labs")
    all_files = fs.ls(gs_path)
    pat = re.compile(r"/action_capture_(\d+)\.jsonl$")
    ordered_action_files = sorted(
        (f for f in all_files if pat.search(f)),
        key=lambda f: int(pat.search(f).group(1)),
    )
    actions = []
    for file in ordered_action_files:
        with fs.open(file, "r") as f:
            for line in f:
                action = json.loads(line)
                actions.append(action)

    return actions


def download_to_tmp(gs_path: str, tmp_root: str = "/tmp") -> str:
    video_path = gs_path + "/screen_capture_000000.mp4"
    bucket_name, blob_path = video_path.replace("gs://", "").split("/", 1)
    local_path = os.path.join(tmp_root, blob_path)

    # Fast path: already downloaded.
    if os.path.exists(local_path):
        return local_path

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    bucket = _client.bucket(bucket_name)
    bucket.blob(blob_path).download_to_filename(local_path)

    return local_path


def extract_frames_by_pts(video_path, pts_list) -> list[tuple[float, PIL.Image.Image]]:
    """
    Extract frames from video at specific PTS timestamps.

    Args:
        video_path (str): Path to video file
        pts_list (list): List of PTS timestamps

    Returns:
        list: List of PIL Image objects
    """

    with av.open(video_path) as container:
        return extract_frames_by_pts_from_container(container, pts_list)


if TYPE_CHECKING:
    from av.filter import FilterContext


def extract_frames_by_pts_from_container(
    container: InputContainer,
    timestamps: list[float],
    buffer_src: FilterContext | None = None,
    buffer_sink: FilterContext | None = None,
) -> list[tuple[float, PIL.Image.Image]]:
    """
    Extract frames from video at specific PTS timestamps.

    Args:
        video_path (str): Path to video file
        pts_list (list): List of PTS timestamps

    Returns:
        list: List of PIL Image objects
    """
    frames: list[tuple[float, PIL.Image.Image]] = []

    stream = container.streams.video[0]
    time_base = stream.time_base
    assert time_base is not None, "Stream time base is None"

    for timestamp in sorted(timestamps):
        # Seek to timestamp
        pts = int(timestamp / time_base)

        container.seek(pts, stream=stream)

        # Get the closest frame
        last_frame = None
        frame_found = False

        for frame in container.decode(stream):
            last_frame = frame
            frame_timestamp = float(frame.pts * time_base)
            if frame.pts >= pts:
                # print(
                # f"Extracting frame at PTS {float(frame.pts * time_base)} (requested {timestamp})"
                # )
                if buffer_src and buffer_sink:
                    # If we have a filter graph, push the frame through it
                    buffer_src.push(frame)
                    filtered_frame: av.VideoFrame = cast(
                        av.VideoFrame, buffer_sink.pull()
                    )
                    assert frame.pts == filtered_frame.pts, (
                        f"{frame.pts=}, {filtered_frame.pts=}"
                    )
                    frame = filtered_frame
                frames.append(
                    (
                        frame_timestamp,
                        frame.to_image(),
                    )
                )
                frame_found = True
                break

        # If no frame was found (timestamp beyond video end), use the last available frame
        if not frame_found and last_frame is not None:
            # print(
            # f"Timestamp {timestamp} beyond video end, using last frame at PTS {float(last_frame.pts * time_base)}"
            # )
            if buffer_src and buffer_sink:
                # If we have a filter graph, push the last frame through it
                buffer_src.push(last_frame)
                filtered_frame: av.VideoFrame = cast(av.VideoFrame, buffer_sink.pull())
                assert last_frame.pts == filtered_frame.pts, (
                    f"{last_frame.pts=}, {filtered_frame.pts=}"
                )

            frames.append(
                (
                    float(last_frame.pts * time_base),
                    last_frame.to_image(),
                )
            )

    return frames


def process_sample(entry, file_lock):
    # entry_good_attempt = next(
    #     attempt
    #     for attempt in entry["attempts"]
    #     if attempt.get("approve_status") == "approved"
    # )
    # if len(entry_good_attempt["eval_dates"]) > 1:
    #     print(
    #         f"Skipping {entry_good_attempt['attempt_id']} since it has multiple eval dates for now."
    #     )
    #     return
    attempt_id = entry["attempt_id"]
    with open(f"{DATA_PATH}/metadata/{attempt_id}.json") as f:
        image_and_action_pairs = json.load(f)

    res_with_wait = [
        turn["text"].split("Action: ")[-1] for turn in image_and_action_pairs
    ]
    images = [turn["image"] for turn in image_and_action_pairs]

    # skip last turn (duplicated)
    res_with_wait = res_with_wait[:-1]
    images = images[:-1]

    chain_of_thought_with_action = []
    # print(entry["instruction"])
    for i in range(len(res_with_wait) - 1):
        original_image = images[i]
        action = res_with_wait[i]
        new_image = images[i + 1]
        next_action = res_with_wait[i + 1]
        next_next = images[i + 2] if (i + 2) < len(images) else new_image
        # display(images[i])
        # print(f"Action: {action}")
        # display(images[i + 1])
        tries = 3
        while tries > 0:
            tries -= 1
            try:
                response = get_next_thought(
                    instruction=entry["instruction"],
                    old_turns=format_cot_with_action(chain_of_thought_with_action),
                    new_action=action,
                    next_action=next_action,
                    old_image=original_image,
                    new_image=new_image,
                    next_image=next_next,
                )
                completion = response.output_text
                break
            except Exception as e:
                if tries == 0:
                    raise e

        # print(response.usage)
        chain_of_thought_with_action.append(
            {
                "action": action,
                "thinking": completion,
            }
        )
        # print(f"Action: {action}")
        # print(f"Thinking: {completion}")

    res = get_next_thought(
        instruction=entry["instruction"],
        old_turns=format_cot_with_action(chain_of_thought_with_action),
        new_action=res_with_wait[-1],
        next_action=None,
        old_image=images[-2],
        new_image=images[-1],
        next_image=None,
    )
    chain_of_thought_with_action.append(
        {
            "action": res_with_wait[-1],
            "thinking": res.output_text,
        }
    )

    final_block = []
    for i, (img, action_thinking) in enumerate(
        zip(images, chain_of_thought_with_action, strict=False)
    ):
        action = action_thinking["action"]
        thinking = fix_string(action_thinking["thinking"])
        final_block.append(
            {
                "step": i,
                "image": img,
                "action": action,
                "thinking": thinking,
                "text": f"Thought: {thinking}\nAction: {action}",
            }
        )

    attempt_id = entry["attempt_id"]
    with open(f"{OUTPUT_PATH}/metadata/{attempt_id}.json", "w") as f:
        json.dump(final_block, f)

    with file_lock, open(f"{OUTPUT_PATH}/samples.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "attempt_id": attempt_id,
                    "actions": [block["action"] for block in final_block],
                    "thinking": [block["thinking"] for block in final_block],
                    "instruction": entry["instruction"],
                    "eval_task_id": entry["eval_task_id"],
                    "trajectory_length": len(final_block),
                }
            )
        )
        f.write("\n")


def thread_worker(entry, lock):
    try:
        process_sample(entry, lock)  # your I/O-bound work
        return (entry, None)
    except Exception:
        return (entry, traceback.format_exc())


def process_worker(entries_chunk, threads_per_proc, lock, update_queue):
    """
    Runs in each separate process. Spins up a thread pool to process its chunk.
    Reports completion (with error or not) back via update_queue.
    """
    with ThreadPoolExecutor(max_workers=threads_per_proc) as executor:
        futures = {
            executor.submit(thread_worker, entry, lock): entry
            for entry in entries_chunk
        }
        for fut in as_completed(futures):
            entry, err = fut.result()
            # send a tuple so main can log / update progress
            update_queue.put((entry, err))
    # signal this process is done (optional; main can infer from counts)
    return


def progress_listener(total, update_queue):
    """
    Runs in a thread in main process to consume updates and show progress bar.
    """
    pbar = tqdm(total=total, desc="Processing samples", unit="item")
    completed = 0
    while completed < total:
        try:
            entry, err = update_queue.get(timeout=1)
        except Exception:
            continue
        if err:
            tqdm.write(f"[!] Error processing {entry['attempt_id']}:\n{err}")
        else:
            tqdm.write(f"Done processing {entry['attempt_id']}.")
        completed += 1
        pbar.update(1)
    pbar.close()


def chunkify(lst, n):
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def main(entries, num_processes=4, threads_per_process=5):
    manager = Manager()
    lock = manager.Lock()  # shared across processes
    update_queue: Queue = manager.Queue()

    total = len(entries)
    # start progress listener thread
    listener = threading.Thread(
        target=progress_listener, args=(total, update_queue), daemon=True
    )
    listener.start()

    # split entries among processes
    chunks = chunkify(entries, num_processes)
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
    # wait until listener sees all updates
    listener.join()


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        os.makedirs(f"{OUTPUT_PATH}/metadata")

    entries = pd.read_json(f"{DATA_PATH}/all_correct_samples.jsonl", lines=True)

    values = entries.to_dict(orient="records")
    main(values, num_processes=15, threads_per_process=15)
    # process_sample(values[53], threading.Lock())  # For testing purposes
