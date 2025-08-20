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

    def format_messages(self, base64_image: str, prompt_text: str) -> list[dict]:
        assert base64_image.startswith("data:image/png;base64,"), (
            f"Invalid image data URI {base64_image[:30]}..."
        )
        multimodal_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text,
                }
            ],
        }
        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                }
            ],
        }
        return [multimodal_message, image_message]
