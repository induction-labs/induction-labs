from __future__ import annotations

import json
import time

from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key as KeyboardKey
from pynput.mouse import Button as MouseButtonEnum
from pynput.mouse import Controller as MouseController
from synapse.actions.models import Action, KeyButton, MouseButton, MouseMove, Scroll


def load_actions(path):
    """Read actions.log and parse each line into an Action, preserving order."""
    actions = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            # Pydantic model parsing
            action = Action.model_validate(data)
            actions.append(action)
    return actions


def get_mouse_button(button_name: str):
    """Map string to pynput.mouse.Button."""
    try:
        return getattr(MouseButtonEnum, button_name)
    except AttributeError as e:
        raise ValueError(f"Unknown mouse button: {button_name}") from e


def get_keyboard_key(key_str: str):
    """Map string to pynput.keyboard.Key or leave as character."""
    # Try special Key enum first
    try:
        return getattr(KeyboardKey, key_str)
    except AttributeError:
        # fallback to literal character
        return key_str


def replay(actions):
    mouse = MouseController()
    keyboard = KeyboardController()

    if not actions:
        print("No actions to replay.")
        return

    # Sort by timestamp just in case
    actions.sort(key=lambda a: a.timestamp)
    start_ts = actions[0].timestamp
    prev_ts = start_ts

    for act in actions:
        # wait for the same interval as recorded
        delay_sec = act.timestamp - prev_ts
        if delay_sec > 0:
            time.sleep(delay_sec)
        prev_ts = act.timestamp

        a = act.action
        # Dispatch based on action type
        if isinstance(a, MouseMove):
            # absolute positioning
            mouse.position = (a.x/1.25, a.y/1.25)

        elif isinstance(a, MouseButton):
            btn = get_mouse_button(a.button)
            if a.is_down:
                mouse.press(btn)
            else:
                mouse.release(btn)

        elif isinstance(a, Scroll):
            mouse.scroll(a.delta_x, a.delta_y)

        elif isinstance(a, KeyButton):
            key = get_keyboard_key(a.key)
            if a.is_down:
                keyboard.press(key)
            else:
                keyboard.release(key)

        else:
            # Unknown action type
            print(f"Skipping unsupported action: {a}")


log_path = r"C:\Users\jonat\Documents\coding\induction-labs\repos\actioncollector\action_capture_jonathanwindows_2025-06-28_103905_JU1TV_action_capture_000000.jsonl"
actions = load_actions(log_path)
print(f"Loaded {len(actions)} actions — starting replay in 2 seconds...")
time.sleep(2)
replay(actions)
print("Replay finished.")
