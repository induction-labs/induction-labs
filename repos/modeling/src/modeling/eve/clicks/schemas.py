from __future__ import annotations

from pydantic import BaseModel, computed_field

from modeling.eve.clicks.api_client import ClickModelClientResponse


class ClickInput(BaseModel):
    id: str
    image_url: str
    instruction: str
    id: str
    width: int
    height: int
    x1: float
    y1: float
    x2: float
    y2: float


class AugmentedEvaluationResult(BaseModel):
    input: ClickInput
    response: ClickModelClientResponse
    prompt_text: str
    prediction_point: tuple[float, float] | None = None

    @computed_field
    @property
    def center_coords(self) -> tuple[float, float] | None:
        center_x = (self.input.x1 + self.input.x2) / 2.0
        center_y = (self.input.y1 + self.input.y2) / 2.0
        return (center_x, center_y)

    @computed_field
    @property
    def x_error(self) -> float | None:
        if self.center_coords and self.prediction_point is not None:
            return self.prediction_point[0] - self.center_coords[0]
        return None

    @computed_field
    @property
    def y_error(self) -> float | None:
        if self.center_coords and self.prediction_point is not None:
            return self.prediction_point[1] - self.center_coords[1]
        return None

    @computed_field
    @property
    def pixel_distance(self) -> float | None:
        if self.x_error is not None and self.y_error is not None:
            return ((self.x_error) ** 2 + (self.y_error) ** 2) ** 0.5
        return None

    @computed_field
    @property
    def is_in_bbox(self) -> bool:
        if self.prediction_point is not None:
            return (
                self.input.x1 <= self.prediction_point[0] <= self.input.x2
                and self.input.y1 <= self.prediction_point[1] <= self.input.y2
            )
        return False
