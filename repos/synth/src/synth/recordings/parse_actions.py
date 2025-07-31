from __future__ import annotations

import math

from synth.recordings.action_models import (
    Action,
    ClickAction,
    DragAction,
    HotkeyAction,
    LeftDoubleAction,
    Point,
    RightSingleAction,
    ScrollAction,
    TypeAction,
)

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


def parse_actions(raw_actions) -> list[Action]:
    # Sort by timestamp to ensure chronological order
    actions = sorted(raw_actions, key=lambda x: x["timestamp"])

    parsed_actions = []
    keys_currently_pressed = {}  # tracks which keys are currently pressed

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

    enter_pressed_before_shift_up = False

    def add_parsed_action(action_instance, timestamp, end_timestamp):
        """Add a parsed action with the given timestamp"""
        parsed_actions.append(
            Action(
                action=action_instance, timestamp=timestamp, end_timestamp=end_timestamp
            )
        )

    def key_timestamp():
        # either return the time of shift (if set) or typing start time
        return typing_start_time

    last_key_time_cosmetic = None

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
                type_action = TypeAction(content=content)
                add_parsed_action(type_action, key_timestamp(), last_key_time_cosmetic)
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
                            drag_action = DragAction(
                                start_point=Point(
                                    x=mouse_down_pos[0], y=mouse_down_pos[1]
                                ),
                                end_point=Point(x=up_pos[0], y=up_pos[1]),
                            )
                            add_parsed_action(drag_action, mouse_down_time, timestamp)
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
                                if parsed_actions and isinstance(
                                    parsed_actions[-1].action, ClickAction
                                ):
                                    # Replace the last action with a double click
                                    double_click_action = LeftDoubleAction(
                                        point=Point(x=up_pos[0], y=up_pos[1])
                                    )
                                    parsed_actions[-1] = Action(
                                        action=double_click_action,
                                        timestamp=parsed_actions[-1].timestamp,
                                        end_timestamp=timestamp,
                                    )
                                else:
                                    double_click_action = LeftDoubleAction(
                                        point=Point(x=up_pos[0], y=up_pos[1])
                                    )
                                    add_parsed_action(
                                        double_click_action, mouse_down_time, timestamp
                                    )
                            else:
                                # Single click
                                click_action = ClickAction(
                                    point=Point(x=up_pos[0], y=up_pos[1])
                                )
                                add_parsed_action(
                                    click_action, mouse_down_time, timestamp
                                )

                            last_click_time = timestamp
                            last_click_pos = up_pos

                        mouse_down_pos = None

            elif action["button"] == "right" and action["is_down"]:
                # Right click (only care about mouse up)
                right_click_action = RightSingleAction(
                    point=Point(x=action["x"], y=action["y"])
                )
                add_parsed_action(right_click_action, timestamp, timestamp)

        # Handle keyboard events
        elif action["action"] == "key_button":
            key = action["key"].lower()
            if key == "shift_r":
                key = "shift"

            if key in modifier_keys:
                # Track modifier key state
                modifier_keys[key] = action["is_down"]
                if modifier_start_time is None and action["is_down"]:
                    modifier_start_time = timestamp

                if (
                    key == "shift"
                    and not action["is_down"]
                    and enter_pressed_before_shift_up
                ):
                    typing_buffer.append("</shift>")
                    enter_pressed_before_shift_up = False

                if key == "shift" and action["is_down"] and typing_start_time is None:
                    # Record shift press time
                    typing_start_time = timestamp

                if key == "shift" and not action["is_down"]:
                    last_key_time_cosmetic = timestamp
            else:
                if action["is_down"]:
                    # Set typing start time if needed
                    if typing_start_time is None:
                        typing_start_time = timestamp

                    # Check for hotkeys (but exclude shift-only combinations)
                    non_shift_modifiers = [
                        k for k, v in modifier_keys.items() if v and k != "shift"
                    ]
                    if non_shift_modifiers:
                        # It's a hotkey - flush any typing buffer first
                        if typing_buffer:
                            content = "".join(typing_buffer)
                            type_action = TypeAction(content=content)
                            add_parsed_action(
                                type_action, key_timestamp(), last_key_time_cosmetic
                            )
                            typing_buffer = []
                            typing_start_time = None

                        all_modifiers = [k for k, v in modifier_keys.items() if v]
                        hotkey_combo = [*all_modifiers, key]
                        hotkey_action = HotkeyAction(key=hotkey_combo)
                        lookahead_modifier_keys_future = []
                        lookahead_i = i
                        while (
                            lookahead_i + 1 < len(actions)
                            and actions[lookahead_i + 1]["action"]["action"]
                            == "key_button"
                            and not actions[lookahead_i + 1]["action"]["is_down"]
                        ):
                            lookahead_i += 1
                            next_key = actions[lookahead_i]["action"]["key"].lower()
                            if modifier_keys.get(next_key):
                                lookahead_modifier_keys_future.append(
                                    actions[lookahead_i]["timestamp"]
                                )
                            else:
                                if lookahead_i != i + 1:
                                    break
                        add_parsed_action(
                            hotkey_action,
                            modifier_start_time,
                            max(lookahead_modifier_keys_future),
                        )
                        modifier_start_time = None
                        mouse_activity_since_typing = False
                        typing_start_time = None
                    else:
                        # Regular key pressed - process immediately on DOWN event
                        # Flush typing buffer if interrupted or too much time passed
                        if typing_buffer and (
                            last_key_time is None
                            or timestamp - last_key_time > 2.0
                            or mouse_activity_since_typing
                        ):
                            content = "".join(typing_buffer)
                            type_action = TypeAction(content=content)
                            ts = key_timestamp()
                            add_parsed_action(type_action, ts, last_key_time_cosmetic)
                            typing_buffer = []
                            typing_start_time = timestamp

                        # Determine character based on CURRENT shift state
                        physical_key = normalize_key_to_physical(action["key"])
                        shift_is_held = modifier_keys.get("shift", False)

                        # Track this key as currently pressed for cleanup
                        keys_currently_pressed[physical_key] = True

                        # Determine final character
                        original_key = action["key"]
                        if shift_is_held:
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
                                last_char = typing_buffer.pop()
                                if last_char == "</shift>":
                                    shift = last_char
                                    poped_other_shift = False
                                    if typing_buffer:
                                        typing_buffer.pop()
                                        if typing_buffer:
                                            keep_next = typing_buffer[-1]
                                            if keep_next == "<shift>":
                                                typing_buffer.pop()
                                                poped_other_shift = True

                                    if not poped_other_shift:
                                        typing_buffer.append(shift)
                            else:
                                # Standalone backspace
                                if typing_buffer:
                                    content = "".join(typing_buffer)
                                    type_action = TypeAction(content=content)
                                    add_parsed_action(
                                        type_action,
                                        key_timestamp(),
                                        last_key_time_cosmetic,
                                    )
                                    typing_buffer = []
                                    typing_start_time = None

                                backspace_action = TypeAction(content="<Backspace>")
                                add_parsed_action(
                                    backspace_action, timestamp, timestamp
                                )
                        elif final_char.lower() == "space":
                            typing_buffer.append(" ")
                            mouse_activity_since_typing = False
                        elif final_char.lower() == "tab":
                            typing_buffer.append("\\t")
                            mouse_activity_since_typing = False
                        elif final_char.lower() == "enter":
                            if not enter_pressed_before_shift_up and modifier_keys.get(
                                "shift", False
                            ):
                                typing_buffer.append("<shift>")
                                enter_pressed_before_shift_up = True

                            typing_buffer.append("\\n")
                            mouse_activity_since_typing = False
                        elif len(final_char) == 1:  # Single character
                            typing_buffer.append(final_char)
                            mouse_activity_since_typing = False
                        else:
                            # Other special keys - flush typing buffer
                            if typing_buffer:
                                content = "".join(typing_buffer)
                                type_action = TypeAction(content=content)
                                add_parsed_action(
                                    type_action, key_timestamp(), last_key_time_cosmetic
                                )
                                typing_buffer = []
                                typing_start_time = None
                            mouse_activity_since_typing = False

                        last_key_time = timestamp
                else:
                    # Key released - just cleanup
                    physical_key = normalize_key_to_physical(action["key"])
                    # Remove from tracking dict if it exists
                    keys_currently_pressed.pop(physical_key, None)
                    last_key_time_cosmetic = timestamp

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

                scroll_action = ScrollAction(
                    point=Point(x=action["x"], y=action["y"]), direction=direction
                )
                add_parsed_action(
                    scroll_action,
                    last_scroll_time if last_scroll_time else timestamp,
                    timestamp,
                )
                last_scroll_time = timestamp
                last_scroll_pos = scroll_pos

        i += 1

    # Flush any remaining typing buffer
    if typing_buffer:
        content = "".join(typing_buffer)
        type_action = TypeAction(content=content)
        add_parsed_action(type_action, key_timestamp(), last_key_time_cosmetic)

    return parsed_actions
