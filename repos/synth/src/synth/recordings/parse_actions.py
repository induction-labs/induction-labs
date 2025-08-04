from __future__ import annotations

import math
from enum import Enum
from typing import cast

from pydantic import BaseModel
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


class SpecialKeys(str, Enum):
    CAPSLOCK = "caps_lock"
    NUMLOCK = "num_lock"
    SCROLLLOCK = "scroll_lock"


# State for Capslock, Numlock, Scrolllock
class LockState(BaseModel):
    state: int = 0  # 4-state: 0=off up, 1=on down, 2=off down, 3=on up

    def press(self, is_down: bool):
        # Check that if we are up, we must be going down, and vice versa
        assert self.state % 2 == (0 if is_down else 1), "Invalid lock state transition"
        return LockState(state=(self.state + 1) % 4)

    def is_active(self) -> bool:
        return self.state in (1, 2)


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

CAPSLOCKS_KEYS = set("abcdefghijklmnopqrstuvwxyz")
NUMLOCK_KEYS = set("0123456789")


def normalize_key_to_physical(key):
    """Convert any key to its physical key identifier (unshifted form)"""
    # Create reverse lookup for shifted chars -> unshifted
    reverse_map = {v.lower(): k for k, v in SHIFT_MAP.items()}
    key_lower = key.lower()
    return reverse_map.get(key_lower, key_lower)


def parse_scroll(
    events: list[dict],
    /,
    page_delta: int = 5,
    point_threshold: int = 50,
    time_threshold: float = 4.0,
) -> list[Action]:
    """
    Collapse raw wheel events into high-level “page scroll” actions.

    Works for both vertical (dy) and horizontal (dx) motion.
    A page is emitted whenever |ΣΔ| ≥ page_delta, or when the group
    is broken by     • pointer jump > point_threshold
                     • time gap     > time_threshold
                     • change of direction or axis.

    Any residual motion < page_delta is still flushed as one page
    when a group closes or the stream ends.
    """
    out: list[Action] = []

    # running-group state ---------------------------------------------------
    origin: tuple[int, int] | None = None  # pointer at group start
    axis: str | None = None  # "x" (horizontal) or "y" (vertical)
    direction: str | None = None  # "up"|"down"|"left"|"right"
    acc: int = 0  # |ΣΔ| accumulated so far
    start_t: float | None = None  # first timestamp contributing
    last_t: float | None = None  # previous event time

    def flush():
        """Emit one residual page, if any, for the current group."""
        nonlocal acc, start_t, last_t, origin, axis, direction
        if acc and start_t is not None and last_t is not None:
            out.append(
                Action(
                    action=ScrollAction(
                        point=Point(x=origin[0], y=origin[1]), direction=direction
                    ),
                    timestamp=start_t,
                    end_timestamp=last_t,
                )
            )
        # reset group state (origin/axis/direction stay - they are set on next use)
        acc = 0
        start_t = None

    # ----------------------------------------------------------------------
    for ev in events:
        dx = ev["action"]["delta_x"]
        dy = ev["action"]["delta_y"]
        if dx == 0 and dy == 0:  # ignore “no-op” wheel events
            continue

        # Pick the *dominant* axis for this event.  (Best-effort: we assume
        # most mice send either dx OR dy ≠ 0 - if both are non-zero we use
        # whichever has the larger magnitude.)
        if abs(dy) > 0:
            delta = dy
            axis_now = "y"
            dir_now = "up" if delta > 0 else "down"
        else:
            delta = dx
            axis_now = "x"
            dir_now = "right" if delta > 0 else "left"

        x = ev["action"]["x"]
        y = ev["action"]["y"]
        t = ev["timestamp"]

        # initialise group on the first scroll event ever
        if origin is None:
            origin = (x, y)
            axis = axis_now
            direction = dir_now

        # does this event *break* the current group?
        if (
            axis_now != axis
            or dir_now != direction
            or abs(x - origin[0]) > point_threshold
            or abs(y - origin[1]) > point_threshold
            or (last_t is not None and t - last_t > time_threshold)
        ):
            flush()
            origin = (x, y)
            axis = axis_now
            direction = dir_now

        if start_t is None:  # first contribution to (possibly new) page
            start_t = t

        acc += abs(delta)
        last_t = t

        # spit out complete pages immediately
        while acc >= page_delta:
            out.append(
                Action(
                    action=ScrollAction(
                        point=Point(x=origin[0], y=origin[1]), direction=direction
                    ),
                    timestamp=start_t,
                    end_timestamp=t,
                )
            )
            acc -= page_delta
            start_t = t if acc else None  # leftovers start “now”

    flush()  # residue at end of stream
    return out


def convert_key_for_lock_state(
    original_key: str,
    lock_keys: dict[SpecialKeys, LockState],
    modifier_keys: dict[str, bool],
) -> str:
    physical_key = normalize_key_to_physical(original_key)
    assert isinstance(physical_key, str)
    # Capslocks overrides shift. We just return uppercase if capslock is on and the key is a letter
    if lock_keys[SpecialKeys.CAPSLOCK].is_active() and physical_key in CAPSLOCKS_KEYS:
        return SHIFT_MAP[physical_key]

    # TODO: Handle numlock behavior
    shift_held = modifier_keys.get("shift", False)
    if shift_held:
        return SHIFT_MAP.get(physical_key, physical_key)
    return physical_key


def parse_actions(raw_actions: list[dict]) -> list[Action]:
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
    lock_keys = {
        SpecialKeys.CAPSLOCK: LockState(),
        SpecialKeys.NUMLOCK: LockState(),
        SpecialKeys.SCROLLLOCK: LockState(),
    }

    typing_buffer = []
    typing_start_time = None
    modifier_start_time = None
    last_key_time = None

    # Scroll grouping
    # last_scroll_time = None
    # last_scroll_pos = None

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

    scroll_events = []

    for i, action in enumerate(actions):
        action = actions[i]["action"]
        timestamp = actions[i]["timestamp"]

        if (
            action["action"] != "scroll"
            and action["action"] != "mouse_move"
            and scroll_events
        ):
            # flush scroll events if any
            scroll_actions = parse_scroll(scroll_events)
            parsed_actions.extend(scroll_actions)
            scroll_events = []

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
                    modifier_start_time = None

                if key == "shift" and action["is_down"] and typing_start_time is None:
                    # Record shift press time
                    typing_start_time = timestamp

                if key == "shift" and not action["is_down"]:
                    last_key_time_cosmetic = timestamp
                continue
            # Check for lock keys
            if key in lock_keys:
                key = cast(SpecialKeys, key)
                lock_keys[key] = lock_keys[key].press(action["is_down"])

                continue

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
                        and actions[lookahead_i + 1]["action"]["action"] == "key_button"
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
                    original_key = action["key"]
                    assert isinstance(original_key, str)
                    final_char = convert_key_for_lock_state(
                        original_key, lock_keys, modifier_keys
                    )
                    physical_key = normalize_key_to_physical(action["key"])
                    keys_currently_pressed[physical_key] = True

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
                            add_parsed_action(backspace_action, timestamp, timestamp)
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
            scroll_events.append(actions[i])

    # Flush any remaining typing buffer
    if typing_buffer:
        content = "".join(typing_buffer)
        type_action = TypeAction(content=content)
        add_parsed_action(type_action, key_timestamp(), last_key_time_cosmetic)

    if scroll_events:
        # Flush any remaining scroll events
        scroll_actions = parse_scroll(scroll_events)
        parsed_actions.extend(scroll_actions)

    return sorted(parsed_actions, key=lambda x: x.timestamp)
