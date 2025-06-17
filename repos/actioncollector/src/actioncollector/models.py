from typing import Literal, Union, Optional, Self
import time
from pydantic import BaseModel, Field


# Mouse action models
class MouseMove(BaseModel):
    action: Literal["mouse_move"] = "mouse_move"
    x: int
    y: int


class MouseButton(BaseModel):
    action: Literal["mouse_button"] = "mouse_button"
    button: Literal["left", "right", "middle"]
    x: int
    y: int
    is_down: bool


class Scroll(BaseModel):
    action: Literal["scroll"] = "scroll"
    delta_x: int
    delta_y: int

    x: int
    y: int


# Keyboard action models
class KeyButton(BaseModel):
    action: Literal["key_button"] = "key_button"
    key: str
    is_down: bool


actionType = Union[MouseMove, MouseButton, Scroll, KeyButton]

class Action(BaseModel):
    action: actionType = Field(discriminator="action")
    timestamp: float

    @staticmethod
    def from_action_type(action: actionType, timestamp: float=None) -> Self:
        return Action(action=action, timestamp=time.time_ns()*1e-9 if timestamp is None else timestamp)
