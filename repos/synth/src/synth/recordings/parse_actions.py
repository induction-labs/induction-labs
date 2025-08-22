from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel
from synapse.utils.logging import logging
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

logger = logging.getLogger(__name__)


class SpecialKeys(str, Enum):
    ENTER = "enter"
    ESCAPE = "esc"
    CAPSLOCK = "caps_lock"
    NUMLOCK = "num_lock"
    SCROLLLOCK = "scroll_lock"
    BACKSPACE = "backspace"
    SPACE = "space"
    TAB = "tab"
    ALT = "alt"
    OPTION = "option"
    DELETE = "delete"
    CMD = "cmd"
    CTRL = "ctrl"
    # Arrows
    SHIFT = "shift"
    L_ARROW = "left"
    R_ARROW = "right"
    U_ARROW = "up"
    D_ARROW = "down"


type LockKeys = Literal[
    SpecialKeys.CAPSLOCK, SpecialKeys.NUMLOCK, SpecialKeys.SCROLLLOCK
]

type ArrowKeys = Literal[
    SpecialKeys.L_ARROW,
    SpecialKeys.R_ARROW,
    SpecialKeys.U_ARROW,
    SpecialKeys.D_ARROW,
]
ArrowKeysSet: set[ArrowKeys] = {
    SpecialKeys.L_ARROW,
    SpecialKeys.R_ARROW,
    SpecialKeys.U_ARROW,
    SpecialKeys.D_ARROW,
}


# State for Capslock, Numlock, Scrolllock
class LockState(BaseModel):
    state: int = 0  # 4-state: 0=off up, 1=on down, 2=off down, 3=on up

    def press(self, is_down: bool):
        # Check that if we are up, we must be going down, and vice versa
        if self.state % 2 != (0 if is_down else 1):
            logger.warning(f"Invalid lock state transition: {self.state} -> {is_down}")
        # assert self.state % 2 == (0 if is_down else 1), "Invalid lock state transition"
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
reverse_map = {v.lower(): k for k, v in SHIFT_MAP.items()}


def normalize_key_to_physical(key):
    """Convert any key to its physical key identifier (unshifted form)"""
    # Create reverse lookup for shifted chars -> unshifted
    key_lower = key.lower()
    return reverse_map.get(key_lower, key_lower)


# TODO: Take screensize as argument
def parse_scroll(
    events: list[dict],
    /,
    page_delta: int = 1024,
    point_threshold: int = 70,
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
    acc_dx: int = 0  # |ΣΔ| accumulated so far
    acc_dy: int = 0  # |ΣΔ| accumulated so far
    start_t: float | None = None  # first timestamp contributing
    last_t: float | None = None  # previous event time
    last_scroll_pos: tuple[int, int] | None = None  # last scroll position

    def flush():
        """Emit one residual page, if any, for the current group."""
        nonlocal \
            acc_dx, \
            acc_dy, \
            start_t, \
            last_t, \
            origin, \
            axis, \
            direction, \
            last_scroll_pos
        if (acc_dx or acc_dy) and start_t is not None and last_t is not None:
            out.append(
                Action(
                    action=ScrollAction(
                        point=Point(x=origin[0], y=origin[1]),
                        direction=direction,
                        displacement=(acc_dx, acc_dy),
                    ),
                    timestamp=start_t,
                    end_timestamp=last_t,
                )
            )
        # reset group state (origin/axis/direction stay - they are set on next use)
        acc_dx = 0
        acc_dy = 0
        last_t = None
        origin = None
        axis = None
        direction = None
        start_t = None
        last_scroll_pos = None

    # ----------------------------------------------------------------------
    for ev in events:
        dx = ev["action"]["delta_x"]
        dy = ev["action"]["delta_y"]
        if dx == 0 and dy == 0:  # ignore “no-op” wheel events
            continue

        # Pick the *dominant* axis for this event.  (Best-effort: we assume
        # most mice send either dx OR dy ≠ 0 - if both are non-zero we use
        # whichever has the larger magnitude.)
        if dy == 0 and dx == 0:
            if origin is not None:
                x = ev["action"]["x"]
                y = ev["action"]["y"]
                t = ev["timestamp"]
            continue
        if abs(dy) > 0:
            axis_now = "y"
            dir_now = "up" if dy > 0 else "down"
        else:
            axis_now = "x"
            dir_now = "right" if dx > 0 else "left"

        x = ev["action"]["x"]
        y = ev["action"]["y"]
        t = ev["timestamp"]

        # initialise group on the first scroll event ever
        if origin is None:
            origin = (x, y)
            axis = axis_now
            direction = dir_now
        if last_scroll_pos is None:
            last_scroll_pos = (x, y)

        # does this event *break* the current group?
        if (
            abs(x - last_scroll_pos[0]) > point_threshold
            or abs(y - last_scroll_pos[1]) > point_threshold
            or (last_t is not None and t - last_t > time_threshold)
        ):
            # print(
            #     "flush1",
            #     ev["timestamp"],
            #     x,
            #     last_scroll_pos[0],
            #     abs(x - last_scroll_pos[0]) > point_threshold,
            #     abs(y - last_scroll_pos[1]) > point_threshold,
            #     last_t is not None and t - last_t > time_threshold,
            # )
            flush()
            origin = (x, y)
            axis = axis_now
            direction = dir_now

        if start_t is None:  # first contribution to (possibly new) page
            start_t = t

        acc_dx += dx
        acc_dy += dy
        last_t = t
        last_scroll_pos = (x, y)

        acc_change = acc_dx if axis_now == "x" else acc_dy

        # spit out complete pages immediately
        if abs(acc_change) >= page_delta:
            # print("flush2", start_t)
            flush()

    flush()  # residue at end of stream
    return out


def convert_key_for_lock_state(
    original_key: str,
    lock_keys: dict[LockKeys, LockState],
    modifier_keys: dict[SpecialKeys, bool],
) -> str:
    physical_key = normalize_key_to_physical(original_key)
    assert isinstance(physical_key, str)
    # Capslocks overrides shift. We just return uppercase if capslock is on and the key is a letter
    if lock_keys[SpecialKeys.CAPSLOCK].is_active() and physical_key in CAPSLOCKS_KEYS:
        return SHIFT_MAP[physical_key]

    # TODO: Handle numlock behavior
    shift_held = modifier_keys.get(SpecialKeys.SHIFT, False)
    if shift_held:
        return SHIFT_MAP.get(physical_key, physical_key)
    return physical_key


WINDOWS_ESCAPE_SEQS = {
    "\x01": "a",
    "\x02": "b",
    "\x03": "c",
    "\x04": "d",
    "\x05": "e",
    "\x06": "f",
    "\x07": "g",
    "\x08": "h",
    "\x09": "i",
    "\x0a": "j",
    "\x0b": "k",
    "\x0c": "l",
    "\x0d": "m",
    "\x0e": "n",
    "\x0f": "o",
    "\x10": "p",
    "\x11": "q",
    "\x12": "r",
    "\x13": "s",
    "\x14": "t",
    "\x15": "u",
    "\x16": "v",
    "\x17": "w",
    "\x18": "x",
    "\x19": "y",
    "\x1a": "z",
    "shift_r": SpecialKeys.SHIFT,
    "shift_l": SpecialKeys.SHIFT,
    "ctrl_l": SpecialKeys.CTRL,
    "ctrl_r": SpecialKeys.CTRL,
    "alt_l": SpecialKeys.ALT,
    "alt_r": SpecialKeys.ALT,
}


def normalize_windows_escape_seqs(key: str) -> str:
    if key in WINDOWS_ESCAPE_SEQS:
        return WINDOWS_ESCAPE_SEQS[key]

    return key


def key_timestamp(typing_buffer: list[tuple[str, float]]) -> float:
    # either return the time of shift (if set) or typing start time
    assert typing_buffer, "Typing buffer should not be empty"
    return typing_buffer[0][1]


def create_type_action(
    typing_buffer: list[tuple[str, float]],
    last_key_time_cosmetic: float | None,
    timestamp: float,
) -> Action:
    """Flush the typing buffer and return a TypeAction"""
    content = "".join(text for text, _ in typing_buffer)
    type_action = TypeAction(content=content)
    action = Action(
        action=type_action,
        timestamp=key_timestamp(typing_buffer),
        end_timestamp=last_key_time_cosmetic or timestamp,
    )
    return action


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

    modifier_keys = {
        SpecialKeys.CTRL: False,
        SpecialKeys.ALT: False,
        SpecialKeys.SHIFT: False,
        SpecialKeys.CMD: False,
    }
    lock_keys: dict[LockKeys, LockState] = {
        SpecialKeys.CAPSLOCK: LockState(),
        SpecialKeys.NUMLOCK: LockState(),
        SpecialKeys.SCROLLLOCK: LockState(),
    }

    typing_buffer: list[tuple[str, float]] = []
    modifier_start_time = None

    # Scroll grouping
    # last_scroll_time = None
    # last_scroll_pos = None

    enter_pressed_before_shift_up = False

    def get_modifiers():
        return {k.value for k, v in modifier_keys.items() if v}

    def add_parsed_action(action_instance, timestamp, end_timestamp):
        """Add a parsed action with the given timestamp"""
        parsed_actions.append(
            Action(
                action=action_instance, timestamp=timestamp, end_timestamp=end_timestamp
            )
        )

    last_key_time_cosmetic = None

    def find_next_key_action(lookahead_i: int, default: float):
        while lookahead_i + 1 < len(actions):
            lookahead_i += 1
            action = actions[lookahead_i]["action"]
            if action["action"] not in {"key_button", "mouse_button"}:
                continue

            key_endtime = actions[lookahead_i]["timestamp"]
            assert isinstance(key_endtime, float), "Expected timestamp to be a float"
            return key_endtime
        return default

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
            if action["is_down"]:
                # Mouse down interrupts typing session
                mouse_activity_since_typing = True
                if typing_buffer:
                    parsed_actions.append(
                        create_type_action(
                            typing_buffer, last_key_time_cosmetic, timestamp
                        )
                    )
                    typing_buffer = []

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
                                modifiers=get_modifiers(),
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
                                        point=Point(x=up_pos[0], y=up_pos[1]),
                                        modifiers=get_modifiers(),
                                    )
                                    parsed_actions[-1] = Action(
                                        action=double_click_action,
                                        timestamp=parsed_actions[-1].timestamp,
                                        end_timestamp=timestamp,
                                    )
                                else:
                                    double_click_action = LeftDoubleAction(
                                        point=Point(x=up_pos[0], y=up_pos[1]),
                                        modifiers=get_modifiers(),
                                    )
                                    add_parsed_action(
                                        double_click_action, mouse_down_time, timestamp
                                    )
                            else:
                                # Single click

                                click_action = ClickAction(
                                    point=Point(x=up_pos[0], y=up_pos[1]),
                                    modifiers=get_modifiers(),
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
                    point=Point(x=action["x"], y=action["y"]),
                    modifiers=get_modifiers(),
                )
                add_parsed_action(right_click_action, timestamp, timestamp)

        # Handle keyboard events
        elif action["action"] == "key_button":
            action["key"] = normalize_windows_escape_seqs(action["key"])
            key = action["key"].lower()

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
                    typing_buffer.append(("</shift>", timestamp))
                    enter_pressed_before_shift_up = False
                    modifier_start_time = None

                if key == "shift" and not action["is_down"]:
                    last_key_time_cosmetic = timestamp
                continue
            # Check for lock keys
            if key in lock_keys:
                print(timestamp)
                lock_keys[key] = lock_keys[key].press(action["is_down"])
                continue

            if action["is_down"]:
                # Set typing start time if needed

                # Check for hotkeys (but exclude shift-only combinations)
                non_shift_modifiers = [
                    k for k, v in modifier_keys.items() if v and k != "shift"
                ]
                if non_shift_modifiers:
                    # It's a hotkey - flush any typing buffer first
                    if typing_buffer:
                        parsed_actions.append(
                            create_type_action(
                                typing_buffer, last_key_time_cosmetic, timestamp
                            )
                        )
                        typing_buffer = []

                    hotkey_action = HotkeyAction(key=key, modifiers=get_modifiers())
                    hotkey_endtime = find_next_key_action(i, timestamp + 0.5)
                    if key == "tab" or key == "`":
                        hotkey_endtime += 0.1

                    add_parsed_action(
                        hotkey_action,
                        timestamp,
                        hotkey_endtime if hotkey_endtime else timestamp,
                    )
                    modifier_start_time = None
                    mouse_activity_since_typing = False
                else:
                    # Regular key pressed - process immediately on DOWN event
                    # Flush typing buffer if interrupted or too much time passed
                    if typing_buffer and (
                        timestamp - typing_buffer[-1][1] > 2.0
                        or mouse_activity_since_typing
                    ):
                        parsed_actions.append(
                            create_type_action(
                                typing_buffer, last_key_time_cosmetic, timestamp
                            )
                        )
                        typing_buffer = []

                    # Determine character based on CURRENT shift state
                    original_key = action["key"]
                    assert isinstance(original_key, str)
                    final_char = convert_key_for_lock_state(
                        original_key, lock_keys, modifier_keys
                    )
                    physical_key = normalize_key_to_physical(action["key"])
                    keys_currently_pressed[physical_key] = True

                    # Handle special keys
                    match final_char.lower():
                        case SpecialKeys.BACKSPACE:
                            if (
                                typing_buffer
                                and timestamp - typing_buffer[-1][1] <= 2.0
                                and not mouse_activity_since_typing
                            ):
                                last_char, last_char_time = typing_buffer.pop()
                                if last_char == "</shift>":
                                    shift = last_char
                                    poped_other_shift = False
                                    if typing_buffer:
                                        _, last_char_time = typing_buffer.pop()
                                        if typing_buffer:
                                            keep_next = typing_buffer[-1][0]
                                            if keep_next == "<shift>":
                                                typing_buffer.pop()
                                                poped_other_shift = True

                                    if not poped_other_shift:
                                        typing_buffer.append((shift, last_char_time))
                            else:
                                # Standalone backspace
                                if typing_buffer:
                                    parsed_actions.append(
                                        create_type_action(
                                            typing_buffer,
                                            last_key_time_cosmetic,
                                            timestamp,
                                        )
                                    )
                                    typing_buffer = []

                                backspace_action = TypeAction(content="<Backspace>")
                                hotkey_endtime = find_next_key_action(
                                    i, timestamp + 0.5
                                )
                                add_parsed_action(
                                    backspace_action, timestamp, hotkey_endtime
                                )
                        case SpecialKeys.ESCAPE:
                            if typing_buffer:
                                parsed_actions.append(
                                    create_type_action(
                                        typing_buffer, last_key_time_cosmetic, timestamp
                                    )
                                )
                                typing_buffer = []
                            escape_action = HotkeyAction(
                                key=final_char.lower(),
                                modifiers=get_modifiers(),
                            )
                            end_timestamp = find_next_key_action(i, timestamp + 0.5)
                            add_parsed_action(escape_action, timestamp, end_timestamp)
                        case SpecialKeys.DELETE:
                            if typing_buffer:
                                parsed_actions.append(
                                    create_type_action(
                                        typing_buffer, last_key_time_cosmetic, timestamp
                                    )
                                )
                                typing_buffer = []
                            delete_action = HotkeyAction(
                                key=final_char.lower(), modifiers=get_modifiers()
                            )
                            hotkey_endtime = find_next_key_action(i, timestamp + 0.5)
                            add_parsed_action(delete_action, timestamp, hotkey_endtime)
                        case SpecialKeys.SPACE:
                            typing_buffer.append((" ", timestamp))
                            mouse_activity_since_typing = False

                        case SpecialKeys.TAB:
                            typing_buffer.append(("\\t", timestamp))
                            mouse_activity_since_typing = False

                        case SpecialKeys.ENTER:
                            if not enter_pressed_before_shift_up and modifier_keys.get(
                                SpecialKeys.SHIFT, False
                            ):
                                typing_buffer.append(("<shift>", timestamp))
                                enter_pressed_before_shift_up = True

                            typing_buffer.append(("\\n", timestamp))
                            mouse_activity_since_typing = False
                        case (
                            SpecialKeys.L_ARROW
                            | SpecialKeys.R_ARROW
                            | SpecialKeys.U_ARROW
                            | SpecialKeys.D_ARROW
                        ):
                            if typing_buffer:
                                parsed_actions.append(
                                    create_type_action(
                                        typing_buffer, last_key_time_cosmetic, timestamp
                                    )
                                )
                                typing_buffer = []

                            arrow_action = HotkeyAction(
                                key=final_char.lower(),
                                modifiers=get_modifiers(),
                            )
                            end_timestamp = find_next_key_action(i, timestamp + 0.5)
                            add_parsed_action(arrow_action, timestamp, end_timestamp)
                        case n if len(n) == 1:  # Single character
                            typing_buffer.append((final_char, timestamp))
                            mouse_activity_since_typing = False
                        case unknown_char:
                            # Other special keys - flush typing buffer
                            print(f"Unhandled key: {unknown_char}")
                            if typing_buffer:
                                parsed_actions.append(
                                    create_type_action(
                                        typing_buffer, last_key_time_cosmetic, timestamp
                                    )
                                )
                                typing_buffer = []
                            mouse_activity_since_typing = False

                last_key_time_cosmetic = None
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
        content = "".join(text for text, _ in typing_buffer)
        type_action = TypeAction(content=content)
        add_parsed_action(
            type_action, key_timestamp(typing_buffer), last_key_time_cosmetic
        )

    if scroll_events:
        # Flush any remaining scroll events
        scroll_actions = parse_scroll(scroll_events)
        parsed_actions.extend(scroll_actions)

    return sorted(parsed_actions, key=lambda x: x.timestamp)
