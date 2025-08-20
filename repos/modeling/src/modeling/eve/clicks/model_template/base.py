from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseClickModelTemplate(BaseModel, ABC):
    @abstractmethod
    def instruction_text(self, instruction: str) -> str:
        pass

    @abstractmethod
    def extract_coordinates(
        self, response: str, image_dimensions: tuple[float, float]
    ) -> tuple[float, float] | None:
        pass
