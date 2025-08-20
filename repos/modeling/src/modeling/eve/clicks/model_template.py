import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum

from pydantic import BaseModel

from modeling.eve.os_world.agents.uitars15 import (
    COMPUTER_USE_15_ONLY_CLICKS,
    LANG_EN,
    THOUGHT_LONG,
)


class BaseClickModelTemplate(BaseModel, ABC):
    @abstractmethod
    def instruction_text(self, instruction: str) -> str:
        pass

    @abstractmethod
    def extract_coordinates(
        self, response: str, image_dimensions: tuple[float, float]
    ) -> tuple[float, float] | None:
        pass


def convert_point_to_coordinates(text, is_answer=False):
    # Match the two numbers inside <point> tags
    pattern = r"<point>(\d+)\s+(\d+)</point>"

    def replace_match(match):
        x1, y1 = map(int, match.groups())
        x = (x1 + x1) // 2  # Truncated integer division
        y = (y1 + y1) // 2  # Truncated integer division
        # if is_answer:
        return f"<|box_start|>({x}, {y})<|box_end|>"  # Return only in (x, y) format
        # return f"({x},{y})"  # Return in labeled format

    # Remove [EOS] and replace <point> coordinates
    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()


class UITarsModelTemplate(BaseClickModelTemplate):
    def instruction_text(self, instruction: str) -> str:
        instruction = instruction.capitalize()
        if not instruction.endswith("."):
            instruction += "."

        return COMPUTER_USE_15_ONLY_CLICKS.format(
            thought_mode=THOUGHT_LONG,
            language=LANG_EN,
            instruction=instruction,
        )

    def extract_coordinates(
        self, response: str, image_dimensions: tuple[float, float]
    ) -> tuple[float, float] | None:
        content = response.strip()
        if "<point>" in content:
            content = convert_point_to_coordinates(content)
        if "start_point=" in content:
            content = content.replace("start_point=", "start_box=")
        if "end_point=" in content:
            content = content.replace("end_point=", "end_box=")
        if "point=" in content:
            content = content.replace("point=", "start_box=")

        action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", content, re.DOTALL)
        action_text = action_match.group(1).strip() if action_match else content

        print(f"{action_text=}")
        click_match = re.search(
            r"click\(start_box='<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>'\)",
            action_text,
        )
        if click_match:
            rel_x = float(click_match.group(1))
            rel_y = float(click_match.group(2))
            pred_x = round(rel_x)
            pred_y = round(rel_y)
            return pred_x, pred_y

        # Process double click action
        double_click_match = re.search(
            r"left_double\(start_box='<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>'\)",
            action_text,
        )
        if double_click_match:
            rel_x = float(double_click_match.group(1))
            rel_y = float(double_click_match.group(2))
            pred_x = round(rel_x)
            pred_y = round(rel_y)
            return pred_x, pred_y

        # Process generic coordinate pattern
        coord_match = re.search(r"\((\d+),\s*(\d+)\)", content)
        if coord_match:
            rel_x = int(coord_match.group(1))
            rel_y = int(coord_match.group(2))
            pred_x = round(rel_x)
            pred_y = round(rel_y)
            return pred_x, pred_y

        # Process x=X, y=Y format
        x_match = re.search(r"x\s*=\s*(\d+)", content, re.IGNORECASE)
        y_match = re.search(r"y\s*=\s*(\d+)", content, re.IGNORECASE)
        if x_match and y_match:
            rel_x = int(x_match.group(1))
            rel_y = int(y_match.group(1))
            pred_x = round(rel_x)
            pred_y = round(rel_y)
            return pred_x, pred_y

        # Process box format
        box_match = re.search(
            r"<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>", content
        )
        if box_match:
            rel_x = int(box_match.group(1))
            rel_y = int(box_match.group(2))
            pred_x = round(rel_x)
            pred_y = round(rel_y)
            return pred_x, pred_y

        return None


class ModelTemplateChoice(str, Enum):
    uitars = "uitars"


MODEL_TEMPLATES: Mapping[ModelTemplateChoice, BaseClickModelTemplate] = {
    ModelTemplateChoice.uitars: UITarsModelTemplate(),
}
