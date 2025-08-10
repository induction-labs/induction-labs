from __future__ import annotations

import math
import string
from collections import defaultdict
from dataclasses import dataclass

from synapse.actions.keyboard_tokenizer import KeyActionType, Tokenizer
from synapse.actions.models import Action, KeyButton

windows_mapping = {
    "cmd": "win",
}
linux_mapping = {}
macos_mapping = {}

keyboard_mapping = {
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
    "-": "_",
    "=": "+",
    "`": "~",
    "[": "{",
    "]": "}",
    "\\": "|",
    ";": ":",
    "'": '"',
    ",": "<",
    ".": ">",
    "/": "?",
}

for lower in string.ascii_lowercase:
    upper = lower.upper()
    keyboard_mapping[lower] = upper

State = dict[str, bool]

# --- helpers --------------------------------------------------------------

MODIFIERS = {"shift", "ctrl", "control", "alt", "cmd", "win", "meta", "option", "super"}

# unshifted → shifted and the reverse map
SHIFT_MAP = keyboard_mapping
UNSHIFT_MAP = {s: u for u, s in SHIFT_MAP.items()}


def _base_key(k: str) -> str:
    """
    Collapse all glyph variants of the same physical key to a single id.
    e.g. '#', '3'  →  '3'      'A', 'a'  →  'a'
    """
    return UNSHIFT_MAP.get(k, k.lower())


def _label(k: str, shift: bool) -> str:
    """Human label for a key event, taking current shift state into account."""
    if k in MODIFIERS:
        return k  # modifiers are never remapped
    if shift:
        return SHIFT_MAP.get(k, k.upper() if len(k) == 1 else k)
    return k


@dataclass
class DownRec:
    ts: float  # timestamp of the DOWN
    seg: int  # segment index where the DOWN token lives
    tok_idx: int  # position of the DOWN token inside that segment (-1 if truncated)
    label: str  # label used for the token ("A", "#", …)
    candidate: bool  # still eligible to become a CLICK


# --- main ------------------------------------------------------------------


def keys_to_tokens(
    actions: list[Action],
    segments: list[float],
    tokenizer: Tokenizer,
    clock_tick_len: float,
    time_per_segment: float,
    press_threshold: float = 0.1,
    os_map: dict[str, str] = macos_mapping,
) -> list[tuple[list[int], State]]:
    """
    Return a list with one entry per segment:
        ( [token_int , …] , { key_label : True , … } )
    """
    # ---------- pre-flight --------------------------------------------------
    # os_map = {**windows_mapping, **linux_mapping, **macos_mapping}
    max_ticks = int(time_per_segment / clock_tick_len)
    wait_tok = tokenizer.mappings["[wait]"]
    end_tok = tokenizer.mappings["[end]"]

    # sort events just in case
    events = sorted(
        [a for a in actions if getattr(a.action, "action", "") == "key_button"],
        key=lambda a: a.timestamp,
    )

    # ---------- state for the whole walk -----------------------------------
    segments_out: list[tuple[list[int], State]] = []
    pressed_keys: set[str] = set()  # labels currently held
    shift_pressed = False

    pending_downs: dict[str, list[DownRec]] = defaultdict(list)

    # ---------- per-segment scratch ----------------------------------------
    seg_idx = 0
    seg_start = segments[seg_idx]
    seg_end = segments[seg_idx + 1] if seg_idx + 1 < len(segments) else math.inf

    tokens: list[int] = []
    last_tick = -1  # last tick index of *any* token in this segment
    tick_count = 0
    truncated = False

    def _add_token(tok: int) -> int | None:
        nonlocal tick_count, truncated
        if truncated or tick_count >= max_ticks:
            truncated = True
            return None
        tokens.append(tok)
        tick_count += 1
        return len(tokens) - 1  # position of the newly inserted token

    def _flush_segment():
        """Finish current segment and start a new, empty one."""
        nonlocal tokens, last_tick, tick_count, truncated
        # add [end] unless we already ran out of room
        if not truncated:
            _add_token(end_tok)
        # capture snapshot of pressed keys
        segments_out.append((tokens, dict.fromkeys(pressed_keys, True)))
        # reset per-segment scratch
        tokens, last_tick, tick_count, truncated = [], -1, 0, False

    # ---------- walk every event -------------------------------------------
    for evt in events:
        k_raw = os_map.get(evt.action.key, evt.action.key)
        is_down = evt.action.is_down

        while evt.timestamp >= seg_end:
            # move to the correct segment
            _flush_segment()
            seg_idx += 1
            if seg_idx >= len(segments) - 1:
                seg_end = math.inf
            else:
                seg_end = segments[seg_idx + 1]
            seg_start = segments[seg_idx]

        # derive tick index (ceil to future)
        offset = max(0.0, evt.timestamp - seg_start)
        tick = math.ceil(offset / clock_tick_len - 1e-12)

        expected_seg_start = seg_idx * time_per_segment
        if evt.timestamp < expected_seg_start:
            # Use actual segment start for events before expected start
            offset = max(0.0, evt.timestamp - seg_start)
        else:
            # Use expected segment start for events after expected start
            offset = max(0.0, evt.timestamp - expected_seg_start)
        tick = math.ceil(offset / clock_tick_len - 1e-12)

        if tick <= last_tick:
            tick = last_tick + 1
        # insert waits
        while not truncated and last_tick < tick - 1:
            last_tick += 1
            _add_token(wait_tok)
        last_tick = tick if not truncated else last_tick

        # ---- handle DOWN ---------------------------------------------------
        if is_down:
            label = _label(k_raw, shift_pressed)
            base = _base_key(k_raw)

            # previous unmatched downs for this key lose click eligibility
            if pending_downs[base]:
                pending_downs[base][-1].candidate = False

            tok_idx = _add_token(tokenizer.mappings[label][KeyActionType.DOWN])
            pending_downs[base].append(
                DownRec(
                    evt.timestamp,
                    seg_idx,
                    tok_idx,
                    label,
                    candidate=(label not in MODIFIERS),
                )
            )
            pressed_keys.add(label)
            if label == "shift":
                shift_pressed = True
            continue

        # ---- handle UP -----------------------------------------------------
        base = _base_key(k_raw)
        if not pending_downs[base]:
            continue  # stray UP - ignore

        down = pending_downs[base].pop()
        delta = evt.timestamp - down.ts
        can_click = (
            down.candidate
            and delta <= press_threshold
            and down.label not in MODIFIERS
            and len(pending_downs[base]) == 0  # no extra downs still pending
            and not truncated
            and down.tok_idx is not None
        )

        if can_click:
            # rewrite the earlier DOWN into a CLICK, no UP emitted
            target_tokens = tokens if down.seg == seg_idx else segments_out[down.seg][0]
            target_tokens[down.tok_idx] = tokenizer.mappings[down.label][
                KeyActionType.CLICK
            ]
            last_tick -= 1
        else:
            # normal UP token
            label = down.label
            _add_token(tokenizer.mappings[label][KeyActionType.UP])

        pressed_keys.discard(down.label)
        if down.label == "shift":
            shift_pressed = "shift" in pressed_keys

    # --------- final flush & return ----------------------------------------
    _flush_segment()
    return segments_out


def get_first_segment_tokens(
    token_with_state: list[tuple[list[int], State]],
) -> list[int]:
    return token_with_state[0][0]


def debug_actions(actions: list[int], tokenizer: Tokenizer) -> str:
    s = ""
    for action in actions:
        s += tokenizer.debug_reverse_mapping(action)
        s += ";\n"

    return s


def test_simple_key_with_wait():
    tokenizer = Tokenizer.load(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
    )
    actions = [
        Action.from_action_type(KeyButton(key="a", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="b", is_down=True), timestamp=1),
        Action.from_action_type(KeyButton(key="c", is_down=True), timestamp=2),
        Action.from_action_type(KeyButton(key="a", is_down=False), timestamp=3),
        Action.from_action_type(KeyButton(key="b", is_down=False), timestamp=4),
        Action.from_action_type(KeyButton(key="c", is_down=False), timestamp=5),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.5,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    expected = [
        tokenizer.mappings["a"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["[wait]"],  # 0.5
        tokenizer.mappings["b"][KeyActionType.DOWN],  # 1.0
        tokenizer.mappings["[wait]"],  # 1.5
        tokenizer.mappings["c"][KeyActionType.DOWN],  # 2.0
        tokenizer.mappings["[wait]"],  # 2.5
        tokenizer.mappings["a"][KeyActionType.UP],  # 3.0
        tokenizer.mappings["[wait]"],  # 3.5
        tokenizer.mappings["b"][KeyActionType.UP],  # 4.0
        tokenizer.mappings["[wait]"],  # 4.5
        tokenizer.mappings["c"][KeyActionType.UP],  # 5.0
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )

    tokens_no_wait = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=1.0,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    expected = [
        tokenizer.mappings["a"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["b"][KeyActionType.DOWN],  # 1.0
        tokenizer.mappings["c"][KeyActionType.DOWN],  # 2.0
        tokenizer.mappings["a"][KeyActionType.UP],  # 3.0
        tokenizer.mappings["b"][KeyActionType.UP],  # 4.0
        tokenizer.mappings["c"][KeyActionType.UP],  # 5.0
        # NOTE: no [end] since we are at the max # of steps
    ]

    assert tokens_no_wait == expected, (
        f"got:\n{debug_actions(tokens_no_wait, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )

    tokens_wait_round_up = get_first_segment_tokens(
        keys_to_tokens(
            [
                Action.from_action_type(KeyButton(key="a", is_down=True), timestamp=0),
                Action.from_action_type(
                    KeyButton(key="b", is_down=True), timestamp=1.2
                ),
                Action.from_action_type(
                    KeyButton(key="a", is_down=False), timestamp=1.3
                ),
                Action.from_action_type(
                    KeyButton(key="b", is_down=False), timestamp=1.4
                ),
            ],
            timestamps,
            tokenizer,
            clock_tick_len=0.5,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    # NOTE: round up (ceil) the wait time to the next clock tick
    expected = [
        tokenizer.mappings["a"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["[wait]"],  # 0.5
        tokenizer.mappings["[wait]"],  # 1.0
        tokenizer.mappings["b"][
            KeyActionType.DOWN
        ],  # 1.5 note that we are rounding from 1.2 to 1.5
        tokenizer.mappings["a"][KeyActionType.UP],  # 2.0
        tokenizer.mappings["b"][KeyActionType.UP],  # 2.0
        tokenizer.mappings["[end]"],
    ]
    assert tokens_wait_round_up == expected, (
        f"got:\n{debug_actions(tokens_wait_round_up, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )


def test_modifier():
    tokenizer = Tokenizer.load(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
    )
    actions = [
        Action.from_action_type(KeyButton(key="cmd", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=1),
        Action.from_action_type(KeyButton(key="cmd", is_down=False), timestamp=2),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=3),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.5,
            time_per_segment=6,
            press_threshold=10,
        )
    )

    expected = [
        tokenizer.mappings["cmd"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["[wait]"],  # 0.5
        tokenizer.mappings["shift"][KeyActionType.DOWN],  # 1.0
        tokenizer.mappings["[wait]"],  # 1.5
        tokenizer.mappings["cmd"][KeyActionType.UP],  # 2.0
        tokenizer.mappings["[wait]"],  # 2.5
        tokenizer.mappings["shift"][KeyActionType.UP],  # 3.0
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )


def test_modifier_windows():
    tokenizer = Tokenizer.load(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
    )
    actions = [
        Action.from_action_type(KeyButton(key="cmd", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=1),
        Action.from_action_type(KeyButton(key="cmd", is_down=False), timestamp=2),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=3),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.5,
            time_per_segment=6,
            press_threshold=10,
            os_map=windows_mapping,  # use windows mapping
        )
    )

    expected = [
        tokenizer.mappings["win"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["[wait]"],  # 0.5
        tokenizer.mappings["shift"][KeyActionType.DOWN],  # 1.0
        tokenizer.mappings["[wait]"],  # 1.5
        tokenizer.mappings["win"][KeyActionType.UP],  # 2.0
        tokenizer.mappings["[wait]"],  # 2.5
        tokenizer.mappings["shift"][KeyActionType.UP],  # 3.0
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )


def test_simple_with_press_threshold():
    tokenizer = Tokenizer.load(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
    )
    actions = [
        Action.from_action_type(KeyButton(key="a", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="b", is_down=True), timestamp=0.05),
        Action.from_action_type(KeyButton(key="c", is_down=True), timestamp=0.1),
        Action.from_action_type(KeyButton(key="a", is_down=False), timestamp=0.12),
        Action.from_action_type(KeyButton(key="b", is_down=False), timestamp=0.15),
        Action.from_action_type(KeyButton(key="c", is_down=False), timestamp=0.25),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.05,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    expected = [
        tokenizer.mappings["a"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["b"][KeyActionType.CLICK],  # 0.05
        tokenizer.mappings["c"][KeyActionType.DOWN],  # 0.1
        tokenizer.mappings["a"][KeyActionType.UP],  # 0.15
        tokenizer.mappings["[wait]"],  # 0.20
        tokenizer.mappings["c"][KeyActionType.UP],  # 0.25
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )


def test_simple_with_press_threshold_2():
    tokenizer = Tokenizer.load(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
    )
    actions = [
        Action.from_action_type(KeyButton(key="a", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="c", is_down=True), timestamp=0.1),
        Action.from_action_type(KeyButton(key="a", is_down=False), timestamp=0.12),
        Action.from_action_type(KeyButton(key="b", is_down=True), timestamp=0.14),
        Action.from_action_type(KeyButton(key="b", is_down=False), timestamp=0.15),
        Action.from_action_type(KeyButton(key="d", is_down=True), timestamp=0.24),
        Action.from_action_type(KeyButton(key="c", is_down=False), timestamp=0.25),
        Action.from_action_type(KeyButton(key="d", is_down=True), timestamp=0.30),
        Action.from_action_type(KeyButton(key="d", is_down=False), timestamp=0.35),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.05,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    expected = [
        tokenizer.mappings["a"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["[wait]"],  # 0.05
        tokenizer.mappings["c"][KeyActionType.DOWN],  # 0.1
        tokenizer.mappings["a"][KeyActionType.UP],  # 0.15
        tokenizer.mappings["b"][KeyActionType.CLICK],  # 0.20
        tokenizer.mappings["d"][KeyActionType.DOWN],  # 0.25
        tokenizer.mappings["c"][KeyActionType.UP],  # 0.30
        tokenizer.mappings["d"][KeyActionType.DOWN],  # 0.35
        tokenizer.mappings["d"][KeyActionType.UP],  # 0.40
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )


def test_down_with_click_somehow():
    tokenizer = Tokenizer.load(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
    )
    actions = [
        Action.from_action_type(KeyButton(key="a", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="a", is_down=True), timestamp=0.03),
        Action.from_action_type(KeyButton(key="a", is_down=False), timestamp=0.05),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.05,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    # NOTE: click should not apply here; even though there is a is_down and not is_down pair with click_threshold < 0.1,
    # we should consider this a compound key press, not a click since there is a down between.
    expected = [
        tokenizer.mappings["a"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["a"][KeyActionType.DOWN],  # 0.05
        tokenizer.mappings["a"][KeyActionType.UP],  # 0.10
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )


def test_down_with_shift():
    tokenizer = Tokenizer.load(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
    )
    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="a", is_down=True), timestamp=0.03),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.05),
        Action.from_action_type(KeyButton(key="a", is_down=False), timestamp=0.13),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.05,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    # NOTE: modifiers can never be click actions
    # also, notice that shift is pressed so the key should be the corresponding shifted key
    # and the shift key should be down at the end
    expected = [
        tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["A"][KeyActionType.CLICK],  # 0.05
        tokenizer.mappings["shift"][KeyActionType.UP],  # 0.10
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )

    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="a", is_down=True), timestamp=0.03),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.05),
        Action.from_action_type(KeyButton(key="a", is_down=False), timestamp=0.21),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.05,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    expected = [
        tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["A"][KeyActionType.DOWN],  # 0.05
        tokenizer.mappings["shift"][KeyActionType.UP],  # 0.10
        tokenizer.mappings["[wait]"],  # 0.15
        tokenizer.mappings["[wait]"],  # 0.20
        tokenizer.mappings["A"][KeyActionType.UP],  # 0.25
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )

    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="A", is_down=True), timestamp=0.03),
        Action.from_action_type(KeyButton(key="A", is_down=False), timestamp=0.04),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.05),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.05,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    expected = [
        tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["A"][KeyActionType.CLICK],  # 0.05
        tokenizer.mappings["shift"][KeyActionType.UP],  # 0.10
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )

    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="A", is_down=True), timestamp=0.03),
        Action.from_action_type(KeyButton(key="A", is_down=False), timestamp=0.14),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.3),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.05,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    expected = [
        tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["A"][KeyActionType.DOWN],  # 0.05
        tokenizer.mappings["[wait]"],  # 0.10
        tokenizer.mappings["A"][KeyActionType.UP],  # 0.15
        tokenizer.mappings["[wait]"],  # 0.20
        tokenizer.mappings["[wait]"],  # 0.25
        tokenizer.mappings["shift"][KeyActionType.UP],  # 0.30
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )


def test_down_with_shift_special():
    tokenizer = Tokenizer.load(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
    )
    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="#", is_down=True), timestamp=0.03),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.05),
        Action.from_action_type(KeyButton(key="3", is_down=False), timestamp=0.13),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.05,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    # NOTE: notice that # and 3 are the same key, so we make it the "shifted" version of the key
    expected = [
        tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["#"][KeyActionType.CLICK],  # 0.05
        tokenizer.mappings["shift"][KeyActionType.UP],  # 0.10
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )

    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="#", is_down=True), timestamp=0.03),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.05),
        Action.from_action_type(KeyButton(key="3", is_down=False), timestamp=0.21),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.05,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    # NOTE: even for a gap (when the shifted key is not a click), we still rewrite the UP event to
    # be the shifted version of the key
    expected = [
        tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["#"][KeyActionType.DOWN],  # 0.05
        tokenizer.mappings["shift"][KeyActionType.UP],  # 0.10
        tokenizer.mappings["[wait]"],  # 0.15
        tokenizer.mappings["[wait]"],  # 0.20
        tokenizer.mappings["#"][KeyActionType.UP],  # 0.25
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )

    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="#", is_down=True), timestamp=0.03),
        Action.from_action_type(KeyButton(key="#", is_down=False), timestamp=0.04),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.05),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.05,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    expected = [
        tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["#"][KeyActionType.CLICK],  # 0.05
        tokenizer.mappings["shift"][KeyActionType.UP],  # 0.10
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )

    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="#", is_down=True), timestamp=0.03),
        Action.from_action_type(KeyButton(key="#", is_down=False), timestamp=0.14),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.3),
    ]
    timestamps = [0.0, 6.0]
    tokens = get_first_segment_tokens(
        keys_to_tokens(
            actions,
            timestamps,
            tokenizer,
            clock_tick_len=0.05,
            time_per_segment=6,
            press_threshold=0.1,
        )
    )

    expected = [
        tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
        tokenizer.mappings["#"][KeyActionType.DOWN],  # 0.05
        tokenizer.mappings["[wait]"],  # 0.10
        tokenizer.mappings["#"][KeyActionType.UP],  # 0.15
        tokenizer.mappings["[wait]"],  # 0.20
        tokenizer.mappings["[wait]"],  # 0.25
        tokenizer.mappings["shift"][KeyActionType.UP],  # 0.30
        tokenizer.mappings["[end]"],
    ]
    assert tokens == expected, (
        f"got:\n{debug_actions(tokens, tokenizer)}\n\nexpected:\n{debug_actions(expected, tokenizer)}"
    )


def test_multiple_segments_shift():
    tokenizer = Tokenizer.load(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
    )
    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="#", is_down=True), timestamp=0.24),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.45),
        Action.from_action_type(KeyButton(key="3", is_down=False), timestamp=0.46),
        Action.from_action_type(KeyButton(key="a", is_down=True), timestamp=0.7),
        Action.from_action_type(KeyButton(key="a", is_down=False), timestamp=0.75),
    ]
    timestamps = [0.0, 0.45, 1.0]
    tokens = keys_to_tokens(
        actions,
        timestamps,
        tokenizer,
        clock_tick_len=0.05,
        time_per_segment=0.5,
        press_threshold=0.1,
    )

    assert tokens == [
        (
            [
                tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
                tokenizer.mappings["[wait]"],  # 0.05
                tokenizer.mappings["[wait]"],  # 0.10
                tokenizer.mappings["[wait]"],  # 0.15
                tokenizer.mappings["[wait]"],  # 0.20
                tokenizer.mappings["#"][KeyActionType.DOWN],  # 0.25
                tokenizer.mappings["[end]"],
            ],
            {
                # current keys that are pressed
                "shift": True,
                "#": True,
            },
        ),
        (
            [
                # NOTE: even though the shift has a timing of 0.45 which technically is under the first segement according to time_per_segment, the timestamp is 0.45 we split it into the next segment
                # the time_per_segment should only be used in the case where there are too many actions in a segment and we need to truncate (see below test)
                tokenizer.mappings["shift"][
                    KeyActionType.UP
                ],  # "0.50" (actually time 0.46 but we are in the next segment)
                tokenizer.mappings["#"][KeyActionType.UP],  # 0.55
                tokenizer.mappings["[wait]"],  # 0.60
                tokenizer.mappings["[wait]"],  # 0.65
                tokenizer.mappings["a"][KeyActionType.CLICK],  # 0.70
                tokenizer.mappings["[end]"],
            ],
            {
                # no state since shift and # released
            },
        ),
    ]

    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="#", is_down=True), timestamp=0.24),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.45),
        Action.from_action_type(KeyButton(key="3", is_down=False), timestamp=0.46),
        Action.from_action_type(KeyButton(key="a", is_down=True), timestamp=0.7),
        Action.from_action_type(KeyButton(key="a", is_down=False), timestamp=0.75),
    ]
    timestamps = [0.0, 0.45, 1.0]
    tokens = keys_to_tokens(
        actions,
        timestamps,
        tokenizer,
        clock_tick_len=0.05,
        time_per_segment=0.6,
        press_threshold=0.1,
    )

    assert tokens == [
        (
            [
                tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
                tokenizer.mappings["[wait]"],  # 0.05
                tokenizer.mappings["[wait]"],  # 0.10
                tokenizer.mappings["[wait]"],  # 0.15
                tokenizer.mappings["[wait]"],  # 0.20
                tokenizer.mappings["#"][KeyActionType.DOWN],  # 0.25
                tokenizer.mappings["[end]"],
            ],
            {
                # current keys that are pressed
                "shift": True,
                "#": True,
            },
        ),
        (
            [
                tokenizer.mappings["shift"][
                    KeyActionType.UP
                ],  # "0.60" (actually time 0.46 but we are in the next segment)
                tokenizer.mappings["#"][KeyActionType.UP],  # 0.65
                tokenizer.mappings["a"][KeyActionType.CLICK],  # 0.70
                tokenizer.mappings["[end]"],
            ],
            {
                # no state since shift and # released
            },
        ),
    ]


def test_multiple_segments_wait():
    tokenizer = Tokenizer.load(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
    )
    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="#", is_down=True), timestamp=0.24),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.30),
        Action.from_action_type(KeyButton(key="3", is_down=False), timestamp=0.35),
        Action.from_action_type(KeyButton(key="a", is_down=True), timestamp=0.7),
        Action.from_action_type(KeyButton(key="a", is_down=False), timestamp=0.75),
    ]
    timestamps = [0.0, 0.45, 1.0]
    tokens = keys_to_tokens(
        actions,
        timestamps,
        tokenizer,
        clock_tick_len=0.05,
        time_per_segment=0.6,
        press_threshold=0.1,
    )

    assert tokens == [
        (
            [
                tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
                tokenizer.mappings["[wait]"],  # 0.05
                tokenizer.mappings["[wait]"],  # 0.10
                tokenizer.mappings["[wait]"],  # 0.15
                tokenizer.mappings["[wait]"],  # 0.20
                tokenizer.mappings["#"][KeyActionType.DOWN],  # 0.25
                tokenizer.mappings["shift"][KeyActionType.UP],  # 0.30
                tokenizer.mappings["#"][KeyActionType.UP],  # 0.35
                tokenizer.mappings["[end]"],
            ],
            {},
        ),
        (
            [
                tokenizer.mappings["[wait]"],  # 0.60
                tokenizer.mappings["[wait]"],  # 0.65
                tokenizer.mappings["a"][KeyActionType.CLICK],  # 0.70
                tokenizer.mappings["[end]"],
            ],
            {},
        ),
    ]


def test_multiple_segments_shift_truncation():
    tokenizer = Tokenizer.load(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"
    )
    actions = [
        Action.from_action_type(KeyButton(key="shift", is_down=True), timestamp=0),
        Action.from_action_type(KeyButton(key="A", is_down=True), timestamp=0.10),
        Action.from_action_type(KeyButton(key="A", is_down=False), timestamp=0.11),
        Action.from_action_type(KeyButton(key="B", is_down=True), timestamp=0.12),
        Action.from_action_type(KeyButton(key="B", is_down=False), timestamp=0.13),
        Action.from_action_type(KeyButton(key="B", is_down=True), timestamp=0.14),
        Action.from_action_type(KeyButton(key="B", is_down=False), timestamp=0.15),
        Action.from_action_type(KeyButton(key="B", is_down=True), timestamp=0.16),
        Action.from_action_type(KeyButton(key="A", is_down=True), timestamp=0.17),
        Action.from_action_type(KeyButton(key="shift", is_down=False), timestamp=0.45),
        Action.from_action_type(KeyButton(key="B", is_down=False), timestamp=0.46),
        Action.from_action_type(KeyButton(key="A", is_down=False), timestamp=0.47),
    ]
    timestamps = [0.0, 0.45, 1.0]
    tokens = keys_to_tokens(
        actions,
        timestamps,
        tokenizer,
        clock_tick_len=0.1,
        time_per_segment=0.5,
        press_threshold=0.001,
    )

    assert tokens == [
        (
            [
                tokenizer.mappings["shift"][KeyActionType.DOWN],  # 0.0
                tokenizer.mappings["A"][KeyActionType.DOWN],  # 0.10
                tokenizer.mappings["A"][KeyActionType.UP],  # 0.20
                tokenizer.mappings["B"][KeyActionType.DOWN],  # 0.30
                tokenizer.mappings["B"][KeyActionType.UP],  # 0.40
            ],
            {
                # NOTE: Truncate keys over the time_per_segment, but leave even the unpressed keys in the state
                # B and A (at t=0.16, and t=0.17 respectively) are not pressed (they are truncated), but we still keep them in the state
                # also, note that overall the timestamps + the time per segment is not inclusive at the end (e.g. time=0.45 means it
                # is at the start of the first segment not the end of the first)
                # we also have no [end] since the sequence was truncated
                "B": True,
                "A": True,
                "shift": True,
            },
        ),
        (
            [
                tokenizer.mappings["shift"][KeyActionType.UP],  # 0.50
                tokenizer.mappings["B"][KeyActionType.UP],  # 0.60
                tokenizer.mappings["A"][KeyActionType.UP],  # 0.70
                tokenizer.mappings["[end]"],
            ],
            {
                # no state
            },
        ),
    ]
