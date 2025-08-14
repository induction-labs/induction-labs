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
from synth.recordings.parse_actions import SpecialKeys, parse_actions


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


def action_model(action_instance, timestamp, end_t=None):
    """Helper to create Action model instances"""
    end_t = end_t if end_t is not None else timestamp
    return Action(action=action_instance, timestamp=timestamp, end_timestamp=end_t)


# ---------------------------------------------------------------------------


# 1 ───── Click (<5 px) vs Drag (≥5 px) ──────────────────────────────────────
def test_click_vs_drag():
    ev_click = [mb(10, 10, True, 0.0), mb(13, 11, False, 0.1)]  # dist ≈3.2
    ev_drag = [mb(20, 20, True, 1.0), mb(25, 20, False, 1.1)]  # dist = 5

    out = parse_actions(ev_click + ev_drag)
    expected = [
        action_model(ClickAction(point=Point(x=13, y=11)), 0.0, end_t=0.1),
        action_model(
            DragAction(start_point=Point(x=20, y=20), end_point=Point(x=25, y=20)),
            1.0,
            end_t=1.1,
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
    expected = [
        action_model(LeftDoubleAction(point=Point(x=50, y=50)), 0.0, end_t=0.35)
    ]
    assert parse_actions(ev) == expected


# 3 ───── Hotkey eats key-ups, no stray type ─────────────────────────────────
def test_hotkey_ctrl_s():
    ev = [
        key("ctrl", True, 0.0),
        key("shift", True, 0.1),
        key("s", True, 0.1),
        key("s", False, 0.2),
        key("shift", False, 0.25),
        key("ctrl", False, 0.3),
        key("b", True, 0.4),
        key("b", False, 0.45),
    ]
    expected = [
        Action(
            action=HotkeyAction(
                action_type="hotkey", modifiers={"shift", "ctrl"}, key="s"
            ),
            timestamp=0.1,
            end_timestamp=0.2,
        ),
        action_model(TypeAction(content="b"), 0.4, end_t=0.45),
    ]
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
        action_model(TypeAction(content="b"), 0.0, end_t=0.05),
        Action(
            action=HotkeyAction(action_type="hotkey", modifiers={"ctrl"}, key="a"),
            timestamp=0.1,
            end_timestamp=0.2,
        ),
        action_model(TypeAction(content="s"), 0.4, end_t=0.5),
    ]
    assert parse_actions(ev) == expected


def test_hotkey_then_type():
    ev = [
        key("b", True, 0.0),
        key("b", False, 0.05),
        key("ctrl", True, 0.06),
        key("a", True, 0.1),
        key("a", False, 0.2),
        key("ctrl", False, 0.3),
        key("shift_r", True, 0.4),
        key("enter", True, 0.5),
        key("enter", False, 0.6),
        key("shift_r", False, 0.7),
    ]
    expected = [
        action_model(TypeAction(content="b"), 0.0, end_t=0.05),
        Action(
            action=HotkeyAction(action_type="hotkey", modifiers={"ctrl"}, key="a"),
            timestamp=0.1,
            end_timestamp=0.2,
        ),
        action_model(TypeAction(content="<shift>\\n</shift>"), 0.5, end_t=0.7),
    ]
    assert parse_actions(ev) == expected


def test_hotkey_then_type_enter_testing():
    ev = [
        key("a", True, 0.0),
        key("shift", True, 0.4),
        key("enter", True, 0.5),
        key("enter", False, 0.6),
        key("enter", True, 0.65),
        key("enter", False, 0.7),
        key("shift", False, 0.75),
        key("a", False, 0.8),
    ]
    expected = [
        action_model(TypeAction(content="a<shift>\\n\\n</shift>"), 0.0, end_t=0.8),
    ]
    assert parse_actions(ev) == expected


def test_hotkey_then_type_enter_testing_backspace():
    ev = [
        key("a", True, 0.0),
        key("shift", True, 0.4),
        key("enter", True, 0.5),
        key("enter", False, 0.6),
        key("enter", True, 0.65),
        key("enter", False, 0.7),
        key("shift", False, 0.75),
        key("a", False, 0.8),
        key("backspace", True, 0.9),
        key("backspace", False, 1.0),
    ]
    expected = [
        action_model(TypeAction(content="a<shift>\\n</shift>"), 0.0, end_t=1.0),
    ]
    assert parse_actions(ev) == expected


def test_hotkey_then_type_enter_backspace_immediate():
    ev = [
        key("a", True, 0.0),
        key("shift", True, 0.4),
        key("enter", True, 0.5),
        key("enter", False, 0.6),
        key("shift", False, 0.75),
        key("a", False, 0.8),
        key("backspace", True, 0.9),
        key("backspace", False, 1.0),
        key("shift", True, 1.2),
        key("enter", True, 1.3),
        key("enter", False, 1.35),
        key("backspace", True, 1.36),
        key("backspace", False, 1.37),
        key("enter", True, 1.4),
        key("enter", False, 1.55),
        key("shift", False, 1.6),
    ]
    expected = [
        action_model(TypeAction(content="a<shift>\\n</shift>"), 0.0, end_t=1.6),
    ]
    assert parse_actions(ev) == expected


def test_shifting():
    ev = [
        key("shift", True, 0.4),
        key("A", True, 0.5),
        key("A", False, 0.6),
        key("A", True, 0.65),
        key("shift", False, 0.75),
        key("a", False, 0.8),
    ]
    expected = [
        action_model(TypeAction(content="AA"), 0.5, end_t=0.8),
    ]
    assert parse_actions(ev) == expected


def test_weird_shift_behaviour():
    ev = [
        key("shift", True, 0.0),
        key("q", True, 0.1),
        key("q", False, 0.2),
        key("W", True, 0.3),
        key("P", True, 0.35),
        key("shift", False, 0.4),
        key("P", False, 0.45),
        key("w", False, 0.5),
        key("a", True, 0.6),
        key("a", False, 0.7),
    ]
    expected = [action_model(TypeAction(content="QWPa"), 0.1, end_t=0.7)]
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
    expected = [action_model(TypeAction(content="Qa"), 0.1, end_t=0.5)]
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
    expected = [action_model(TypeAction(content="Qa"), 0.1, end_t=0.5)]
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
    expected = [action_model(TypeAction(content="Qa"), 0.1, end_t=0.5)]
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
    expected = [action_model(TypeAction(content="$a"), 0.1, end_t=0.5)]
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
    expected = [action_model(TypeAction(content="<a"), 0.1, end_t=0.5)]
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
    expected = [action_model(TypeAction(content="<a"), 0.1, end_t=0.5)]
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
        action_model(TypeAction(content="hello"), 0.0, end_t=0.9),
        action_model(TypeAction(content="h"), 3.0, end_t=3.1),
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
        action_model(TypeAction(content="a"), 0.0, end_t=0.25),  # 'b' removed
        # XXX: backspace timeings are slightly off right now
        action_model(TypeAction(content="<Backspace>"), 2.3, end_t=2.35),  # standalone
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
    expected = [action_model(TypeAction(content="abcd"), 0.0, end_t=2.34)]
    assert out == expected


# 6 ───── Enter encoded as newline in content string ────────────────────────
def test_enter_newline():
    ev = [
        key("x", True, 0.0),
        key("x", False, 0.05),
        key("enter", True, 0.1),
        key("enter", False, 0.15),
    ]
    expected = [action_model(TypeAction(content="x\\n"), 0.0, end_t=0.15)]
    assert parse_actions(ev) == expected


def test_enter_newline_shortcut():
    ev = [
        key("x", True, 0.0),
        key("x", False, 0.05),
        key("enter", True, 0.1),
        key("enter", False, 0.15),
        key("ctrl", True, 0.2),
        key("c", True, 0.25),
        key("c", False, 0.3),
        key("ctrl", False, 0.35),
    ]
    expected = [
        action_model(TypeAction(content="x\\n"), 0.0, end_t=0.15),
        Action(
            action=HotkeyAction(action_type="hotkey", modifiers={"ctrl"}, key="c"),
            timestamp=0.25,
            end_timestamp=0.3,
        ),
    ]
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
    expected = [action_model(TypeAction(content="p\\t\\n"), 0.0, end_t=0.15)]
    assert parse_actions(ev) == expected


def test_click_typing():
    ev = [
        {
            "action": {
                "action": "mouse_button",
                "button": "left",
                "x": 498,
                "y": 635,
                "is_down": True,
            },
            "timestamp": 0,
        },
        {
            "action": {"action": "key_button", "key": "a", "is_down": True},
            "timestamp": 0.05,
        },
        {
            "action": {
                "action": "mouse_button",
                "button": "left",
                "x": 498,
                "y": 635,
                "is_down": False,
            },
            "timestamp": 0.1,
        },
        {
            "action": {"action": "key_button", "key": "a", "is_down": False},
            "timestamp": 0.15,
        },
    ]
    expected = [
        Action(
            action=ClickAction(
                action_type="click", modifiers=set(), point=Point(x=498, y=635)
            ),
            timestamp=0,
            end_timestamp=0.1,
        ),
        Action(
            action=TypeAction(action_type="type", content="a"),
            timestamp=0.05,
            end_timestamp=0.15,
        ),
    ]

    assert parse_actions(ev) == expected


def test_composite_timings():
    ev = [
        scroll(0, 3, 100, 100, 0.0),
        key("p", True, 0.02),
        key("p", False, 0.05),
        mb(100, 100, True, 0.07),
        mb(100, 100, False, 0.08),
        key("enter", True, 0.1),
        key("enter", False, 0.15),
    ]
    expected = [
        action_model(
            ScrollAction(
                point=Point(x=100, y=100), direction="up", displacement=(0, 3)
            ),
            0.0,
        ),
        action_model(TypeAction(content="p"), 0.02, end_t=0.05),
        action_model(ClickAction(point=Point(x=100, y=100)), 0.07, end_t=0.08),
        action_model(TypeAction(content="\\n"), 0.1, end_t=0.15),
    ]
    assert parse_actions(ev) == expected


# 7 ───── Scroll direction mapping ──────────────────────────────────────────
@pytest.mark.parametrize(
    "dx,dy,expect",
    [
        (0, 3, "up"),
        (0, -2, "down"),
        (4, 0, "right"),
        (-4, 0, "left"),
        (4, 2, "up"),
        (2, 4, "up"),
    ],
)
def test_scroll_directions(dx, dy, expect):
    ev = [scroll(dx, dy, 100, 100, 0.0)]
    out = parse_actions(ev)
    expected = [
        action_model(
            ScrollAction(
                point=Point(x=100, y=100), direction=expect, displacement=(dx, (dy))
            ),
            0.0,
        )
    ]
    assert out == expected


def test_multiple_scrolls():
    ev = [
        scroll(0, 3, 100, 100, 0.0),
        scroll(0, 3, 100, 100, 0.1),
        scroll(0, 3, 100, 100, 0.2),
    ]
    out = parse_actions(ev)
    expected = [
        action_model(
            ScrollAction(
                point=Point(x=100, y=100), direction="up", displacement=(0, 9)
            ),
            0.0,
            end_t=0.2,
        ),
    ]
    assert out == expected


def test_multiple_scrolls_move_a_bit():
    ev = [
        scroll(0, 3, 100, 100, 0.0),
        scroll(0, 3, 100, 102, 0.1),
        scroll(0, 3, 100, 100, 0.2),
    ]
    out = parse_actions(ev)
    expected = [
        action_model(
            ScrollAction(
                point=Point(x=100, y=100), direction="up", displacement=(0, 9)
            ),
            0.0,
            end_t=0.2,
        ),
    ]
    assert out == expected


def test_multiple_scrolls_move_a_lot():
    ev = [
        scroll(0, 3, 100, 100, 0.0),
        scroll(0, 3, 150, 900, 0.1),
        scroll(0, 3, 100, 100, 0.2),
    ]
    out = parse_actions(ev)
    expected = [
        action_model(
            ScrollAction(
                point=Point(x=100, y=100), direction="up", displacement=(0, 3)
            ),
            0.0,
            end_t=0.0,
        ),
        action_model(
            ScrollAction(
                point=Point(x=150, y=900), direction="up", displacement=(0, 3)
            ),
            0.1,
            end_t=0.1,
        ),
        action_model(
            ScrollAction(
                point=Point(x=100, y=100), direction="up", displacement=(0, 3)
            ),
            0.2,
            end_t=0.2,
        ),
    ]
    assert out == expected


def test_multiple_scrolls_with_stuff_between():
    ev = [
        scroll(0, 1, 100, 100, 0.0),
        scroll(0, 1, 100, 100, 0.1),
        scroll(0, 1, 100, 100, 0.2),
        key("a", True, 0.3),
        key("a", False, 0.4),
        scroll(0, 1, 100, 100, 0.5),
        scroll(0, 1, 100, 100, 0.6),
        scroll(0, 1, 100, 100, 0.7),
    ]
    out = parse_actions(ev)
    expected = [
        action_model(
            ScrollAction(
                point=Point(x=100, y=100), direction="up", displacement=(0, 3)
            ),
            0.0,
            end_t=0.2,
        ),
        action_model(TypeAction(content="a"), 0.3, end_t=0.4),
        action_model(
            ScrollAction(
                point=Point(x=100, y=100), direction="up", displacement=(0, 3)
            ),
            0.5,
            end_t=0.7,
        ),
    ]
    assert out == expected


def test_interleaved_backspace():
    ev = [
        key("a", True, 0.0),
        key("a", False, 0.05),
        key("b", True, 0.1),
        key("b", False, 0.15),
        key("backspace", True, 0.2),
        key("backspace", False, 0.25),
        key("backspace", True, 0.3),
        key("backspace", False, 0.35),
        key("backspace", True, 0.4),
        key("backspace", False, 0.45),
        key("c", True, 0.5),
        key("c", False, 0.55),
    ]
    out = parse_actions(ev)
    expected = [
        action_model(
            TypeAction(content="<Backspace>"), 0.4, end_t=0.45
        ),  # nothing left
        action_model(TypeAction(content="c"), 0.5, end_t=0.55),
    ]
    assert out == expected


def test_windows_special_keys():
    ev = [
        {
            "action": {"action": "key_button", "key": "ctrl_l", "is_down": True},
            "timestamp": 0,
        },
        {
            "action": {"action": "key_button", "key": "\x03", "is_down": True},
            "timestamp": 0.1,
        },
        {
            "action": {"action": "key_button", "key": "\x03", "is_down": False},
            "timestamp": 0.2,
        },
        {
            "action": {"action": "key_button", "key": "ctrl_l", "is_down": False},
            "timestamp": 0.3,
        },
    ]
    out = parse_actions(ev)
    expected = [
        Action(
            action=HotkeyAction(action_type="hotkey", modifiers={"ctrl"}, key="c"),
            timestamp=0.1,
            end_timestamp=0.2,
        )
    ]
    assert out == expected


def test_arrow_keys_with_space():
    ev = [
        ({"action": "key_button", "key": "left", "is_down": True}, 1754667368.6583781),
        ({"action": "key_button", "key": "left", "is_down": False}, 1754667368.7578633),
        ({"action": "key_button", "key": "left", "is_down": True}, 1754667368.8691633),
        ({"action": "key_button", "key": "left", "is_down": False}, 1754667368.8965094),
        ({"action": "key_button", "key": "space", "is_down": True}, 1754667369.3502343),
        (
            {"action": "key_button", "key": "space", "is_down": False},
            1754667369.4981089,
        ),
        ({"action": "key_button", "key": "right", "is_down": True}, 1754667369.8629737),
        (
            {"action": "key_button", "key": "right", "is_down": False},
            1754667369.9980738,
        ),
        ({"action": "key_button", "key": "right", "is_down": True}, 1754667370.0774493),
    ]
    ev = [{"action": a[0], "timestamp": a[1]} for a in ev]
    out = parse_actions(ev)
    expected = [
        Action(
            action=HotkeyAction(action_type="hotkey", modifiers=set(), key="left"),
            timestamp=1754667368.6583781,
            end_timestamp=1754667368.6583781,
        ),
        Action(
            action=HotkeyAction(action_type="hotkey", modifiers=set(), key="left"),
            timestamp=1754667368.8691633,
            end_timestamp=1754667368.8691633,
        ),
        Action(
            action=TypeAction(action_type="type", content=" "),
            timestamp=1754667369.3502343,
            end_timestamp=1754667369.4981089,
        ),
        Action(
            action=HotkeyAction(action_type="hotkey", modifiers=set(), key="right"),
            timestamp=1754667369.8629737,
            end_timestamp=1754667369.8629737,
        ),
        Action(
            action=HotkeyAction(action_type="hotkey", modifiers=set(), key="right"),
            timestamp=1754667370.0774493,
            end_timestamp=1754667370.0774493,
        ),
    ]
    assert out == expected


# def test_multiple_scrolls_more_than_5():
#     ev = [
#         scroll(0, 1, 100, 100, 0.0),
#         scroll(0, 1, 100, 100, 0.1),
#         scroll(0, 1, 100, 100, 0.2),
#         scroll(0, 1, 100, 100, 0.3),
#         scroll(0, 1, 100, 100, 0.4),
#         # break here and add a new scroll
#         scroll(0, 1, 100, 100, 0.5),
#         scroll(0, 1, 100, 100, 0.6),
#         scroll(0, 3, 100, 100, 0.7),
#         # break into two
#         scroll(0, 10, 100, 100, 0.8),
#         # boundary condition
#         scroll(0, 2, 100, 100, 0.9),
#         scroll(0, 4, 100, 100, 1.0),
#         scroll(0, 2, 100, 100, 1.1),
#         scroll(0, 1, 200, 200, 1.2),
#         scroll(0, 1, 200, 201, 1.3),
#         scroll(0, 1, 200, 200, 1.4),
#         scroll(0, 1, 203, 204, 1.5),
#         scroll(0, 1, 200, 200, 1.6),
#         scroll(0, 4, 200, 200, 1.6),
#         scroll(0, 1, 200, 200, 5.7),
#     ]
#     out = parse_actions(ev)
#     expected = [
#         action_model(
#             ScrollAction(
#                 point=Point(x=100, y=100), direction="up", displacement=(0, 3)
#             ),
#             0.0,
#             end_t=0.4,
#         ),
#         action_model(
#             ScrollAction(
#                 point=Point(x=100, y=100), direction="up", displacement=(0, 3)
#             ),
#             0.5,
#             end_t=0.7,
#         ),
#         action_model(
#             ScrollAction(
#                 point=Point(x=100, y=100), direction="up", displacement=(0, 3)
#             ),
#             0.8,
#             end_t=0.8,
#         ),
#         action_model(
#             ScrollAction(
#                 point=Point(x=100, y=100), direction="up", displacement=(0, 3)
#             ),
#             0.8,
#             end_t=0.8,
#         ),
#         action_model(
#             ScrollAction(
#                 point=Point(x=100, y=100), direction="up", displacement=(0, 3)
#             ),
#             0.9,
#             end_t=1.0,
#         ),
#         # even though there aren't 5 remaining we'll scroll anyways
#         action_model(
#             ScrollAction(
#                 point=Point(x=100, y=100), direction="up", displacement=(0, 3)
#             ),
#             1.0,
#             end_t=1.1,
#         ),
#         # break since the point is different by > 50px
#         action_model(
#             ScrollAction(
#                 point=Point(x=200, y=200), direction="up", displacement=(0, 3)
#             ),
#             1.2,
#             end_t=1.6,
#         ),
#         action_model(
#             ScrollAction(
#                 point=Point(x=200, y=200), direction="up", displacement=(0, 3)
#             ),
#             1.6,
#             end_t=1.6,
#         ),
#         # break here since the time diff >5
#         action_model(
#             ScrollAction(
#                 point=Point(x=200, y=200), direction="up", displacement=(0, 3)
#             ),
#             5.7,
#             end_t=5.7,
#         ),
#     ]
#     assert out == expected


# 8 ───── Lone mouse-moves are ignored ───────────────────────────────────────
def test_ignore_mouse_moves():
    ev = [mv(10, 10, 0.0), mv(20, 20, 0.1)]
    assert parse_actions(ev) == []


# 9 ───── Toggle keys (capslock, numlock) handling ──────────────────────────
def test_capslock_handling():
    """Test that capslock toggles case for subsequent letters"""
    ev = [
        key(SpecialKeys.CAPSLOCK, True, 0.0),
        key(SpecialKeys.CAPSLOCK, False, 0.05),
        key("a", True, 0.1),
        key("a", False, 0.15),
        key("b", True, 0.2),
        key("b", False, 0.25),
    ]
    expected = [
        action_model(TypeAction(content="AB"), 0.1, end_t=0.25),
    ]
    assert parse_actions(ev) == expected


def test_numlock_handling():
    """Test that numlock key presses don't generate typing content but don't break typing flow"""
    ev = [
        key("x", True, 0.0),
        key("x", False, 0.05),
        key(SpecialKeys.NUMLOCK, True, 0.1),
        key(SpecialKeys.NUMLOCK, False, 0.15),
        key("y", True, 0.2),
        key("y", False, 0.25),
    ]
    expected = [
        action_model(TypeAction(content="xy"), 0.0, end_t=0.25),
    ]
    assert parse_actions(ev) == expected


def test_capslock_double_toggle():
    """Test capslock toggled twice returns to original state"""
    ev = [
        key("a", True, 0.0),
        key("a", False, 0.05),
        key(SpecialKeys.CAPSLOCK, True, 0.1),
        key("b", True, 0.15),
        key("b", False, 0.2),
        key(SpecialKeys.CAPSLOCK, False, 0.25),
        key(SpecialKeys.CAPSLOCK, True, 0.3),
        key(SpecialKeys.CAPSLOCK, False, 0.35),
        key("c", True, 0.4),
        key("c", False, 0.45),
    ]
    expected = [
        action_model(TypeAction(content="aBc"), 0.0, end_t=0.45),
    ]
    assert parse_actions(ev) == expected
