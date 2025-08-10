from __future__ import annotations

import json
import os
import random
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Process, Queue

import dotenv
import pandas as pd
from google.cloud import storage
from openai import OpenAI
from synth.recordings.repair_dir import fix_string
from tqdm import tqdm

dotenv.load_dotenv()

client = OpenAI()
OUTPUT_PATH = "/home/jonathan_inductionlabs_com/induction-labs/repos/synth/src/synth/recordings/synth_captions_post_processed"

_client = storage.Client()

DATA_PATH = "/home/jonathan_inductionlabs_com/induction-labs/repos/synth/src/synth/recordings/all_human_snapshot_0807_actually_fixed"
entries = pd.read_json(f"{DATA_PATH}/samples.jsonl", lines=True)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    os.makedirs(f"{OUTPUT_PATH}/metadata")


all_samples = pd.read_json(
    "/home/jonathan_inductionlabs_com/induction-labs/repos/synth/src/synth/recordings/all_trajectories/all_samples.jsonl",
    lines=True,
)
all_good_samples = all_samples[all_samples["reward"] == 1.0]


def random_cot_examples(n: int = 5):
    results = []
    for sample in all_good_samples["attempt_id"].sample(n):
        cot = json.load(
            open(
                f"/home/jonathan_inductionlabs_com/induction-labs/repos/synth/src/synth/recordings/all_trajectories/metadata/{sample}.json"
            )
        )
        res = (
            random.choice(cot)["text"]
            .split("Action: ")[0]
            .split("Thought: ")[-1]
            .strip()
        )
        results.append(res)

    return results


print("loading random COT examples")
random_cots = random_cot_examples(1000)
print("loaded random COT examples")

INSTRUCTION = """You rewrite a user's computer use steps into a concise, high-level thought for why they took an action. You will be given a history of all steps taken, the current step to write a thought for, and the instruction the user was given to perform on the computer.
Do not output hidden or lengthy reasoning; keep it short and externally checkable. It should read like a natural thought a human would have.

Your goal is to:

* Log new, relevant information the moment it's discovered, especially if it's useful for future steps. Feel free to use the future steps to inform your current thinking, but obviously the thinking shouldn't reference anything related to the future that is not certain or that can be reasonably inferred from the current step.
* Adjust or correct earlier thinking when needed, for instance using the word "Actually" if you realized that you took an unnecessary action or redundant step.
* Only include planning if it hasn't been stated before or is especially important at this step.
* Avoid repeating facts or plans already stated in earlier steps.
* Be concise but rich in unique details - pack in all fresh, relevant context from the current frame and history.
* Always end with a single clear sentence describing the action to take now.

Action format:
```
click(start_box='<|box_start|>(x1,y1)<|box_end|>')               # Left click
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')         # Double left click
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')        # Right click
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')  # Drag
hotkey(key='ctrl c')                                             # 1-3 lowercase keys, space-separated
type(content='xxx')                                              # Use \\' , \\" , \\n escapes; \\n to submit
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down|up|right|left')           # Scroll
wait()                                                           # Wait 5s & screenshot
finished()                                                       # Complete task
call_user()                                                      # Ask user for help
```

# User instruction
{USER_INSTRUCTION}

# User (each step)

History of all steps taken (text-only digest):
{HISTORY}

**Current step is step {STEP_NO}**:

* Rough Thinking for current step: {ROUGH_TEXT}
* Action for current step: {ACTION}

Attached is also a screenshot of step {STEP_NO} for context.

Write one freeform paragraph for step {STEP_NO} following the rules above. Be direct, skip filler, keep the narrative moving forward. Correct mistakes and acknowledge pivots if detected in the actions. End with a description of the action to take now.

# Examples of output style (not the content):

{EXAMPLES}
"""


def format_actions_and_thinking(actions, good_thinking, thinking):
    assert len(good_thinking) + len(thinking) == len(actions), (
        "Mismatch in lengths of good_actions, actions, and thinking"
    )

    formatted_actions_and_thinking = ""
    for i, (action, thinking) in enumerate(
        zip(actions, good_thinking + thinking, strict=False)
    ):
        is_good_thinking = i < len(good_thinking)
        thinking_tag = "Thinking:" if is_good_thinking else "Rough Thinking:"
        formatted_actions_and_thinking += (
            f"**Step {i + 1}**\n{thinking_tag} {thinking}\nAction: {action}\n\n"
        )

    return formatted_actions_and_thinking.strip()


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

    good_thoughts = []
    for i in range(len(entry["actions"])):
        history = format_actions_and_thinking(
            entry["actions"], good_thoughts, entry["thinking"][i:]
        )
        image = image_and_action_pairs[i]["image"]
        instruction = INSTRUCTION.format(
            HISTORY=history,
            USER_INSTRUCTION=entry["instruction"],
            STEP_NO=i + 1,
            ROUGH_TEXT=entry["thinking"][i],
            ACTION=entry["actions"][i],
            EXAMPLES="\n".join(random.sample(random_cots, 5)),
        )

        response = client.responses.create(
            model="o3",
            reasoning={"effort": "high"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image}",
                        },
                    ],
                }  # pyright: ignore[reportArgumentType]
            ],
        )

        print(response.output_text)
        good_thoughts.append(response.output_text)

    final_block = []
    for i, (img, thinking, action) in enumerate(
        zip(
            [a["image"] for a in image_and_action_pairs],
            good_thoughts,
            entry["actions"],
            strict=False,
        )
    ):
        thinking = fix_string(thinking)
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
    values = entries.to_dict(orient="records")
    main(values, num_processes=12, threads_per_process=12)
    # process_sample(values[53], threading.Lock())  # For testing purposes
