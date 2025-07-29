from __future__ import annotations

import math

SHIFT_MAP = {
    # Numbers row
    "1": "!",
    "2": "@",
    "3": "#",
    "4": "$",
    "5": "%",
    "6": "^",
    "7": "&",
    "8": "*",
    "9": "(",
    "0": ")",
    # Other symbols
    "`": "~",
    "-": "_",
    "=": "+",
    "[": "{",
    "]": "}",
    "\\": "|",
    ";": ":",
    "'": '"',
    ",": "<",
    ".": ">",
    "/": "?",
    # Letters
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "e": "E",
    "f": "F",
    "g": "G",
    "h": "H",
    "i": "I",
    "j": "J",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "o": "O",
    "p": "P",
    "q": "Q",
    "r": "R",
    "s": "S",
    "t": "T",
    "u": "U",
    "v": "V",
    "w": "W",
    "x": "X",
    "y": "Y",
    "z": "Z",
}


def normalize_key_to_physical(key):
    """Convert any key to its physical key identifier (unshifted form)"""
    # Create reverse lookup for shifted chars -> unshifted
    reverse_map = {v.lower(): k for k, v in SHIFT_MAP.items()}
    key_lower = key.lower()
    return reverse_map.get(key_lower, key_lower)


def parse_actions(raw_actions):
    # Sort by timestamp to ensure chronological order
    actions = sorted(raw_actions, key=lambda x: x["timestamp"])
    for action in actions:
        print(action)
    parsed_actions = []
    keys_pressed_with_shift = {}  # maps physical key to shift state when pressed

    # State tracking
    mouse_down_pos = None
    mouse_down_time = None

    last_click_time = None
    last_click_pos = None
    mouse_activity_since_typing = False

    modifier_keys = {"ctrl": False, "alt": False, "shift": False}
    typing_buffer = []
    typing_start_time = None
    modifier_start_time = None
    last_key_time = None

    # Scroll grouping
    last_scroll_time = None
    last_scroll_pos = None

    def add_parsed_action(action_str, timestamp):
        parsed_actions.append({"action": action_str, "timestamp": timestamp})

    i = 0
    while i < len(actions):
        action = actions[i]["action"]
        timestamp = actions[i]["timestamp"]

        # Handle mouse button events
        if action["action"] == "mouse_button":
            # Mouse activity interrupts typing session
            mouse_activity_since_typing = True
            if typing_buffer:
                content = "".join(typing_buffer)
                add_parsed_action(f"type(content='{content}')", typing_start_time)
                typing_buffer = []
                typing_start_time = None
                last_key_time = None

            if action["button"] == "left":
                if action["is_down"]:
                    # Left mouse down - record position for potential drag
                    mouse_down_pos = (action["x"], action["y"])
                    mouse_down_time = timestamp
                else:
                    # Left mouse up - determine if click or drag
                    if mouse_down_pos:
                        up_pos = (action["x"], action["y"])
                        distance = math.sqrt(
                            (up_pos[0] - mouse_down_pos[0]) ** 2
                            + (up_pos[1] - mouse_down_pos[1]) ** 2
                        )

                        if distance >= 5:  # Minimum distance threshold for drag
                            add_parsed_action(
                                f"drag(start_point='<point>{mouse_down_pos[0]} {mouse_down_pos[1]}</point>', end_point='<point>{up_pos[0]} {up_pos[1]}</point>')",
                                mouse_down_time,
                            )
                        else:
                            # Check for double click (within 500ms and ~same position)
                            if (
                                last_click_time
                                and last_click_pos
                                and timestamp - last_click_time <= 0.5
                                and math.sqrt(
                                    (up_pos[0] - last_click_pos[0]) ** 2
                                    + (up_pos[1] - last_click_pos[1]) ** 2
                                )
                                <= 5
                            ):
                                # Replace last click with double click
                                if parsed_actions and parsed_actions[-1][
                                    "action"
                                ].startswith("click("):
                                    parsed_actions[-1]["action"] = (
                                        f"left_double(point='<point>{up_pos[0]} {up_pos[1]}</point>')"
                                    )
                                else:
                                    add_parsed_action(
                                        f"left_double(point='<point>{up_pos[0]} {up_pos[1]}</point>')",
                                        mouse_down_time,
                                    )
                            else:
                                # Single click
                                add_parsed_action(
                                    f"click(point='<point>{up_pos[0]} {up_pos[1]}</point>')",
                                    mouse_down_time,
                                )

                            last_click_time = timestamp
                            last_click_pos = up_pos

                        mouse_down_pos = None

            elif action["button"] == "right" and action["is_down"]:
                # Right click (only care about mouse up)
                add_parsed_action(
                    f"right_single(point='<point>{action['x']} {action['y']}</point>')",
                    timestamp,
                )

        # Handle keyboard events
        elif action["action"] == "key_button":
            key = action["key"].lower()
            if action["is_down"] and typing_start_time is None:
                typing_start_time = timestamp

            if key in modifier_keys:
                # Track modifier key state
                modifier_keys[key] = action["is_down"]
                if modifier_start_time is None and action["is_down"]:
                    modifier_start_time = timestamp
            else:
                if action["is_down"]:
                    # Check for hotkeys (but exclude shift-only combinations)
                    non_shift_modifiers = [
                        k for k, v in modifier_keys.items() if v and k != "shift"
                    ]
                    if non_shift_modifiers:
                        # It's a hotkey - flush any typing buffer first
                        if typing_buffer:
                            content = "".join(typing_buffer)
                            add_parsed_action(
                                f"type(content='{content}')", typing_start_time
                            )
                            typing_buffer = []
                            typing_start_time = None

                        hotkey_combo = " ".join([*non_shift_modifiers, key])
                        add_parsed_action(
                            f"hotkey(key='{hotkey_combo}')", modifier_start_time
                        )
                        modifier_start_time = None
                        mouse_activity_since_typing = False
                    else:
                        # Regular key pressed - record shift state
                        physical_key = normalize_key_to_physical(action["key"])
                        keys_pressed_with_shift[physical_key] = modifier_keys.get(
                            "shift", False
                        )

                    # Flush typing buffer if interrupted or too much time passed
                    if typing_buffer and (
                        last_key_time is None
                        or timestamp - last_key_time > 2.0
                        or mouse_activity_since_typing
                    ):
                        content = "".join(typing_buffer)
                        add_parsed_action(
                            f"type(content='{content}')", typing_start_time
                        )
                        typing_buffer = []
                        typing_start_time = timestamp
                else:
                    # Key released - determine actual character and add to typing
                    physical_key = normalize_key_to_physical(action["key"])

                    if physical_key in keys_pressed_with_shift:
                        shift_was_held = keys_pressed_with_shift.pop(physical_key)

                        # Determine final character
                        original_key = action["key"]
                        if shift_was_held:
                            final_char = SHIFT_MAP.get(physical_key, original_key)
                        else:
                            final_char = physical_key

                        # Handle special keys
                        if final_char.lower() == "backspace":
                            if (
                                typing_buffer
                                and last_key_time
                                and timestamp - last_key_time <= 2.0
                                and not mouse_activity_since_typing
                            ):
                                typing_buffer.pop()
                            else:
                                # Standalone backspace
                                if typing_buffer:
                                    content = "".join(typing_buffer)
                                    add_parsed_action(
                                        f"type(content='{content}')", typing_start_time
                                    )
                                    typing_buffer = []
                                    typing_start_time = None
                                add_parsed_action(
                                    "type(content='<Backspace>')", timestamp
                                )
                        elif final_char.lower() == "space":
                            typing_buffer.append(" ")
                            mouse_activity_since_typing = False
                        elif final_char.lower() == "tab":
                            typing_buffer.append("\\t")
                            mouse_activity_since_typing = False
                        elif final_char.lower() == "enter":
                            typing_buffer.append("\\n")
                            mouse_activity_since_typing = False
                        elif len(final_char) == 1:  # Single character
                            typing_buffer.append(final_char)
                            mouse_activity_since_typing = False
                        else:
                            # Other special keys - flush typing buffer
                            if typing_buffer:
                                content = "".join(typing_buffer)
                                add_parsed_action(
                                    f"type(content='{content}')", typing_start_time
                                )
                                typing_buffer = []
                                typing_start_time = None
                            mouse_activity_since_typing = False

                        last_key_time = timestamp

        # Handle scroll events
        elif action["action"] == "scroll":
            scroll_pos = (action["x"], action["y"])

            # Check if this scroll is close to the previous one (within 1 second and 50 pixels)
            if (
                last_scroll_time
                and last_scroll_pos
                and timestamp - last_scroll_time <= 1.0
                and math.sqrt(
                    (scroll_pos[0] - last_scroll_pos[0]) ** 2
                    + (scroll_pos[1] - last_scroll_pos[1]) ** 2
                )
                <= 50
            ):
                # Skip this scroll as it's too close to the previous one
                pass
            else:
                # Add new scroll action
                if action["delta_y"] != 0:
                    direction = "down" if action["delta_y"] < 0 else "up"
                else:
                    direction = "left" if action["delta_x"] < 0 else "right"

                add_parsed_action(
                    f"scroll(point='<point>{action['x']} {action['y']}</point>', direction='{direction}')",
                    timestamp,
                )
                last_scroll_time = timestamp
                last_scroll_pos = scroll_pos

        i += 1

    # Flush any remaining typing buffer
    if typing_buffer:
        content = "".join(typing_buffer)
        add_parsed_action(f"type(content='{content}')", typing_start_time)

    return parsed_actions


# Example usage:
# parsed = parse_actions(your_keylog_data)
# for action in parsed:
#     print(action)
