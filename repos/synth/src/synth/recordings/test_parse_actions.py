# test_actions.py
from __future__ import annotations

import pytest
from synth.recordings.action_models import (
    Action,
    ClickAction,
    DragAction,
    HotkeyAction,
    LeftDoubleAction,
    Point,
    ScrollAction,
    TypeAction,
)
from synth.recordings.parse_actions import parse_actions


# ---------- tiny helpers to build events -----------------------------------
def mb(x, y, down, t):  # mouse button
    return {
        "action": {
            "action": "mouse_button",
            "button": "left",
            "x": x,
            "y": y,
            "is_down": down,
        },
        "timestamp": t,
    }


def mv(x, y, t):  # bare mouse-move
    return {"action": {"action": "mouse_move", "x": x, "y": y}, "timestamp": t}


def key(k, down, t):  # keyboard button
    return {
        "action": {"action": "key_button", "key": k, "is_down": down},
        "timestamp": t,
    }


def scroll(dx, dy, x, y, t):
    return {
        "action": {"action": "scroll", "delta_x": dx, "delta_y": dy, "x": x, "y": y},
        "timestamp": t,
    }


def action_model(action_instance, timestamp):
    """Helper to create Action model instances"""
    return Action(action=action_instance, timestamp=timestamp)


# ---------------------------------------------------------------------------


# 1 ───── Click (<5 px) vs Drag (≥5 px) ──────────────────────────────────────
def test_click_vs_drag():
    ev_click = [mb(10, 10, True, 0.0), mb(13, 11, False, 0.1)]  # dist ≈3.2
    ev_drag = [mb(20, 20, True, 1.0), mb(25, 20, False, 1.1)]  # dist = 5

    out = parse_actions(ev_click + ev_drag)
    expected = [
        action_model(ClickAction(point=Point(x=13, y=11)), 0.0),
        action_model(
            DragAction(start_point=Point(x=20, y=20), end_point=Point(x=25, y=20)), 1.0
        ),
    ]
    assert out == expected


# 2 ───── Double-click within 500 ms, same point ─────────────────────────────
def test_double_click_window():
    ev = [
        mb(50, 50, True, 0.0),
        mb(50, 50, False, 0.05),
        mb(50, 50, True, 0.3),
        mb(50, 50, False, 0.35),
    ]
    expected = [action_model(LeftDoubleAction(point=Point(x=50, y=50)), 0.0)]
    assert parse_actions(ev) == expected


# 3 ───── Hotkey eats key-ups, no stray type ─────────────────────────────────
def test_hotkey_ctrl_s():
    ev = [
        key("ctrl", True, 0.0),
        key("s", True, 0.1),
        key("s", False, 0.2),
        key("ctrl", False, 0.3),
    ]
    expected = [action_model(HotkeyAction(key=["ctrl", "s"]), 0.0)]
    assert parse_actions(ev) == expected


def test_hotkey_ctrl_a():
    ev = [
        key("b", True, 0.0),
        key("b", False, 0.05),
        key("ctrl", True, 0.06),
        key("a", True, 0.1),
        key("a", False, 0.2),
        key("ctrl", False, 0.3),
        key("s", True, 0.4),
        key("s", False, 0.5),
    ]
    expected = [
        action_model(TypeAction(content="b"), 0.0),
        action_model(HotkeyAction(key=["ctrl", "a"]), 0.06),
        action_model(TypeAction(content="s"), 0.4),
    ]
    assert parse_actions(ev) == expected


def test_weird_shift_behaviour():
    ev = [
        key("shift", True, 0.0),
        key("q", True, 0.1),
        key("q", False, 0.2),
        key("W", True, 0.3),
        key("shift", False, 0.4),
        key("w", False, 0.5),
        key("a", True, 0.6),
        key("a", False, 0.7),
    ]
    expected = [action_model(TypeAction(content="QWa"), 0.0)]
    assert parse_actions(ev) == expected


def test_weird_shift_behaviour2():
    ev = [
        key("shift", True, 0.0),
        key("q", True, 0.1),
        key("shift", False, 0.2),
        key("q", False, 0.3),
        key("a", True, 0.4),
        key("a", False, 0.5),
    ]
    expected = [action_model(TypeAction(content="Qa"), 0.0)]
    assert parse_actions(ev) == expected


def test_weird_shift_behaviour3():
    ev = [
        key("shift", True, 0.0),
        key("Q", True, 0.1),
        key("shift", False, 0.2),
        key("Q", False, 0.3),
        key("a", True, 0.4),
        key("a", False, 0.5),
    ]
    expected = [action_model(TypeAction(content="Qa"), 0.0)]
    assert parse_actions(ev) == expected


def test_weird_shift_behaviour4():
    ev = [
        key("shift", True, 0.0),
        key("Q", True, 0.1),
        key("shift", False, 0.2),
        key("q", False, 0.3),
        key("a", True, 0.4),
        key("a", False, 0.5),
    ]
    expected = [action_model(TypeAction(content="Qa"), 0.0)]
    assert parse_actions(ev) == expected


def test_weird_shift_behaviour5():
    ev = [
        key("shift", True, 0.0),
        key("$", True, 0.1),
        key("shift", False, 0.2),
        key("4", False, 0.3),
        key("a", True, 0.4),
        key("a", False, 0.5),
    ]
    expected = [action_model(TypeAction(content="$a"), 0.0)]
    assert parse_actions(ev) == expected


def test_weird_shift_behaviour6():
    ev = [
        key("shift", True, 0.0),
        key("<", True, 0.1),
        key("shift", False, 0.2),
        key(",", False, 0.3),
        key("a", True, 0.4),
        key("a", False, 0.5),
    ]
    expected = [action_model(TypeAction(content="<a"), 0.0)]
    assert parse_actions(ev) == expected


def test_weird_shift_behaviour7():
    ev = [
        key("shift", True, 0.0),
        key("<", True, 0.1),
        key("shift", False, 0.2),
        key(",", False, 0.3),
        key("a", True, 0.4),
        key("a", False, 0.5),
    ]
    expected = [action_model(TypeAction(content="<a"), 0.0)]
    assert parse_actions(ev) == expected


# 4 ───── Typing groups (<2 s gap) and splits (>2 s) ────────────────────────
def test_typing_groups_and_gap():
    ev = (
        [key(c, True, i * 0.2) for i, c in enumerate("hello")]
        + [key(c, False, i * 0.2 + 0.1) for i, c in enumerate("hello")]
        + [key("h", True, 3.0), key("h", False, 3.1)]
    )
    out = parse_actions(ev)
    expected = [
        action_model(TypeAction(content="hello"), 0.0),
        action_model(TypeAction(content="h"), 3.0),
    ]
    assert out == expected


# 5 ───── Backspace inside a group removes char; standalone is emitted ──────
def test_backspace_behaviour():
    ev1 = [
        key("a", True, 0.0),
        key("a", False, 0.05),
        key("b", True, 0.1),
        key("b", False, 0.15),
        key("backspace", True, 0.2),
        key("backspace", False, 0.25),
    ]
    ev2 = [key("backspace", True, 2.3), key("backspace", False, 2.35)]

    out = parse_actions(ev1 + ev2)
    expected = [
        action_model(TypeAction(content="a"), 0.0),  # 'b' removed
        action_model(TypeAction(content="<Backspace>"), 2.35),  # standalone
    ]
    assert out == expected


def test_type_over_time():
    ev = [
        key("a", True, 0.0),
        key("a", False, 0.10),
        key("b", True, 0.95),
        key("b", False, 1.05),
        key("c", True, 1.8),
        key("c", False, 1.85),
        key("d", True, 2.3),
        key("d", False, 2.34),
    ]
    out = parse_actions(ev)
    expected = [action_model(TypeAction(content="abcd"), 0.0)]
    assert out == expected


# 6 ───── Enter encoded as newline in content string ────────────────────────
def test_enter_newline():
    ev = [
        key("x", True, 0.0),
        key("x", False, 0.05),
        key("enter", True, 0.1),
        key("enter", False, 0.15),
    ]
    expected = [action_model(TypeAction(content="x\\n"), 0.0)]
    assert parse_actions(ev) == expected


def test_tab_enter():
    ev = [
        key("p", True, 0.0),
        key("p", False, 0.05),
        key("tab", True, 0.0),
        key("tab", False, 0.05),
        key("enter", True, 0.1),
        key("enter", False, 0.15),
    ]
    expected = [action_model(TypeAction(content="p\\t\\n"), 0.0)]
    assert parse_actions(ev) == expected


# 7 ───── Scroll direction mapping ──────────────────────────────────────────
@pytest.mark.parametrize(
    "dx,dy,expect",
    [
        (0, 3, "up"),
        (0, -2, "down"),
        (4, 0, "right"),
        (-5, 0, "left"),
        (6, 2, "up"),
        (2, 6, "up"),
    ],
)
def test_scroll_directions(dx, dy, expect):
    ev = [scroll(dx, dy, 100, 100, 0.0)]
    out = parse_actions(ev)
    expected = [
        action_model(ScrollAction(point=Point(x=100, y=100), direction=expect), 0.0)
    ]
    assert out == expected


# 8 ───── Lone mouse-moves are ignored ───────────────────────────────────────
def test_ignore_mouse_moves():
    ev = [mv(10, 10, 0.0), mv(20, 20, 0.1)]
    assert parse_actions(ev) == []
