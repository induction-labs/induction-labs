from .base import BaseClickModelTemplate

prompt_origin = "Outline the position corresponding to the instruction: {instruction}. The output should be only [x1,y1,x2,y2]."


class VenusGroundModelTemplate(BaseClickModelTemplate):
    def instruction_text(self, instruction: str) -> str:
        return prompt_origin.format(
            instruction=instruction,
        )

    def extract_coordinates(
        self, response: str, image_dimensions: tuple[float, float]
    ) -> tuple[float, float] | None:
        box = eval(response)
        if not isinstance(box, list) or len(box) != 4:
            print(f"Invalid box format: {box}")
            return None
        # input_width, input_height = image_dimensions
        input_width, input_height = 1, 1
        abs_x1 = float(box[0]) / input_width
        abs_y1 = float(box[1]) / input_height
        abs_x2 = float(box[2]) / input_width
        abs_y2 = float(box[3]) / input_height
        bbox = [abs_x1, abs_y1, abs_x2, abs_y2]
        point = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        return point

    def format_messages(self, base64_image: str, prompt_text: str) -> list[dict]:
        assert base64_image.startswith("data:image/png;base64,"), (
            f"Invalid image data URI {base64_image[:30]}..."
        )
        multimodal_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
        return [multimodal_message]
