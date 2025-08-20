from __future__ import annotations

import base64
import io
import json
import time

import requests
from PIL import Image
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential


class ClickModelClientResponse(BaseModel):
    raw_response: str
    content: str
    latency_seconds: float


class ClickModelClient:
    def __init__(
        self,
        api_url: str,
        api_key: str = "super-secret-key",
        max_tokens: int = 128,
        temperature: float = 0.0,
        frequency_penalty: float = 0.0,
        model_name: str = "",
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.model_name = model_name

    def _encode_image(self, image_data_uri: str) -> str:
        if "," in image_data_uri:
            base64_data = image_data_uri.split(",")[1]
        else:
            base64_data = image_data_uri

        return base64_data

    def _extract_image_dimensions(self, base64_data: str) -> tuple[int, int]:
        try:
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            return width, height
        except Exception as e:
            print(f"Error extracting image dimensions: {e}")
            return 1024, 768

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=15)
    )
    def call_model(
        self,
        messages: list[dict],
    ) -> ClickModelClientResponse:
        # base64_image = self._encode_image(image_data_uri)
        # width, height = self._extract_image_dimensions(base64_image)

        # Note: UI-TARS is not a generalist VLM: prompting it with plain English will cause the model to severely collapse.
        # Hence, it is unclear how to change the given computer use prompt, so we just use the default one provided in the UI-TARS repo.
        # drag, right_single, hotkey, type, scroll, wait, finished, call_user are here but we won't use them and will treat it as a failure.
        # assert "{instruction}" in self.prompt_template, (
        #     "Prompt template must contain {instruction} placeholder"
        # )

        # text = self.prompt_template.format(
        #     instruction=prompt,
        # )
        # Prepare the multimodal message

        request_data = {
            "messages": messages,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "frequency_penalty": self.frequency_penalty,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        start_time = time.time()

        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            json=request_data,
            headers=headers,
            timeout=7200,
        )

        end_time = time.time()
        latency = end_time - start_time
        assert response.status_code == 200, f"API call failed: {response.text}"

        result = response.json()

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        return ClickModelClientResponse(
            raw_response=json.dumps(result),
            content=content,
            latency_seconds=latency,
        )
