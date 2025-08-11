from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Union

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: int
    y: int


class ActionBase(BaseModel, ABC):
    """Abstract base class for all actions with text dump capability"""

    @abstractmethod
    def dump_to_text(self) -> str:
        """Convert the action back to its original text format"""


def get_modifiers_str(modifiers: set[str]) -> str:
    """Convert a set of modifier keys to a space-separated string"""
    modifier_str = " ".join(modifiers)
    return f", modifiers='{modifier_str}'" if modifiers else ""


class ClickAction(ActionBase):
    action_type: Literal["click"] = "click"
    modifiers: set[str] = Field(
        default_factory=set,
        description="Modifier keys in lowercase, e.g., ['ctrl', 'shift']",
    )
    point: Point

    def dump_to_text(self) -> str:
        modifier_part = get_modifiers_str(self.modifiers)
        return f"click(start_box='({self.point.x},{self.point.y})'{modifier_part})"


class LeftDoubleAction(ActionBase):
    action_type: Literal["left_double"] = "left_double"
    modifiers: set[str] = Field(
        default_factory=set,
        description="Modifier keys in lowercase, e.g., ['ctrl', 'shift']",
    )
    point: Point

    def dump_to_text(self) -> str:
        modifier_part = get_modifiers_str(self.modifiers)
        return (
            f"left_double(start_box='({self.point.x},{self.point.y})'{modifier_part})"
        )


class RightSingleAction(ActionBase):
    action_type: Literal["right_single"] = "right_single"
    point: Point
    modifiers: set[str] = Field(
        default_factory=set,
        description="Modifier keys in lowercase, e.g., ['ctrl', 'shift']",
    )

    def dump_to_text(self) -> str:
        modifier_part = get_modifiers_str(self.modifiers)
        return (
            f"right_single(start_box='({self.point.x},{self.point.y})'{modifier_part})"
        )


class DragAction(ActionBase):
    action_type: Literal["drag"] = "drag"
    start_point: Point
    end_point: Point
    modifiers: set[str] = Field(
        default_factory=set,
        description="Modifier keys in lowercase, e.g., ['ctrl', 'shift']",
    )

    def dump_to_text(self) -> str:
        modifier_part = get_modifiers_str(self.modifiers)
        return (
            f"drag(start_box='({self.start_point.x},{self.start_point.y})', "
            f"end_box='({self.end_point.x},{self.end_point.y})'{modifier_part})"
        )


class HotkeyAction(ActionBase):
    action_type: Literal["hotkey"] = "hotkey"
    modifiers: set[str] = Field(
        ..., description="Modifier keys in lowercase, e.g., ['ctrl', 'shift']"
    )
    key: str = Field(..., description="Keys in lowercase, max 3 keys")

    def dump_to_text(self) -> str:
        modifier_str = get_modifiers_str(self.modifiers)
        return f"hotkey(key='{self.key}'{modifier_str})"


class TypeAction(ActionBase):
    action_type: Literal["type"] = "type"
    content: str = Field(
        ..., description="Content with escape characters \\', \\\", \\n"
    )

    def dump_to_text(self) -> str:
        return f"type(content='{self.content}')"


class ScrollAction(ActionBase):
    action_type: Literal["scroll"] = "scroll"
    point: Point
    direction: Literal["down", "up", "right", "left"]
    displacement: tuple[int, int] = Field(
        ...,
        description="Displacement in pixels (dx, dy) for the scroll action",
    )

    def dump_to_text(self) -> str:
        return f"scroll(direction='{self.direction}', start_box='({self.point.x},{self.point.y})')"


class WaitAction(ActionBase):
    action_type: Literal["wait"] = "wait"

    def dump_to_text(self) -> str:
        return "wait()"


class FinishedAction(ActionBase):
    action_type: Literal["finished"] = "finished"

    def dump_to_text(self) -> str:
        return "finished()"


actionType = Union[  # noqa: UP007
    ClickAction,
    LeftDoubleAction,
    RightSingleAction,
    DragAction,
    HotkeyAction,
    TypeAction,
    ScrollAction,
    WaitAction,
    FinishedAction,
]


class Action(BaseModel):
    action: actionType = Field(discriminator="action_type")
    timestamp: float
    end_timestamp: float

    def dump_to_text(self) -> str:
        """Dump the action to its original text format"""
        return self.action.dump_to_text()


# Example usage:
if __name__ == "__main__":
    import time

    # Create some example actions
    click_action = Action(
        action=ClickAction(point=Point(x=100, y=200)), timestamp=time.time()
    )

    drag_action = Action(
        action=DragAction(start_point=Point(x=50, y=50), end_point=Point(x=150, y=150)),
        timestamp=time.time(),
    )

    hotkey_action = Action(
        action=HotkeyAction(key=["ctrl", "c"]), timestamp=time.time()
    )

    type_action = Action(
        action=TypeAction(content="Hello World\\n"), timestamp=time.time()
    )

    scroll_action = Action(
        action=ScrollAction(point=Point(x=300, y=400), direction="down"),
        timestamp=time.time(),
    )

    # Test the dump_to_text methods
    print("Text format dumps:")
    print(click_action.dump_to_text())
    print(drag_action.dump_to_text())
    print(hotkey_action.dump_to_text())
    print(type_action.dump_to_text())
    print(scroll_action.dump_to_text())
