import json
import time
import sys
from actioncollector.models import Action, MouseMove, MouseButton, Scroll, KeyButton
from pynput.mouse import Controller as MouseController, Button as MouseButtonEnum
from pynput.keyboard import Controller as KeyboardController, Key as KeyboardKey

def load_actions(path):
    """Read actions.log and parse each line into an Action, preserving order."""
    actions = []
    with open(path, "r") as f:
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
    except AttributeError:
        raise ValueError(f"Unknown mouse button: {button_name}")

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
        delay_ms = act.timestamp - prev_ts
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        prev_ts = act.timestamp

        a = act.action
        # Dispatch based on action type
        if isinstance(a, MouseMove):
            # absolute positioning
            mouse.position = (a.x, a.y)

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

log_path = "tmp/action_capture/jonathan/2025-06-11_104921_XH06E/action_capture_000000.jsonl"
actions = load_actions(log_path)
print(f"Loaded {len(actions)} actions — starting replay in 2 seconds...")
time.sleep(2)
replay(actions)
print("Replay finished.")
