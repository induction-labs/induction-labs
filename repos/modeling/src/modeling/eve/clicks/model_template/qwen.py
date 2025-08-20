import json
import re

from .base import BaseClickModelTemplate

# Qwen action space and prompt from QwenVLClient
QWEN_ACTION_SPACE = {
    "type": "function",
    "function": {
        "name_for_human": "computer_use",
        "name": "computer_use",
        "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\\n* The screen's resolution is {RES_WIDTH}x{RES_HEIGHT}.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        "parameters": {
            "properties": {
                "action": {
                    "description": "The action to perform. The available actions are:\\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n* `type`: Type a string of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button.\\n* `left_click_drag`: Click and drag the cursor from a start coordinate to an end coordinate on the screen.\\n* `right_click`: Click the right mouse button.\\n* `double_click`: Double-click the left mouse button.\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `wait`: Wait for the change to happen.\\n* `terminate`: Terminate the current task when it is completed.",
                    "enum": [
                        "key",
                        "type",
                        "mouse_move",
                        "left_click",
                        "left_click_drag",
                        "right_click",
                        "double_click",
                        "scroll",
                        "wait",
                        "terminate",
                    ],
                    "type": "string",
                },
                "keys": {
                    "description": "Required only by `action=key`.",
                    "type": "array",
                },
                "text": {
                    "description": "Required only by `action=type`.",
                    "type": "string",
                },
                "start_coordinate": {
                    "description": "(x, y): The starting x (pixels from the left edge) and y (pixels from the top edge) coordinates. Required only by `action=left_click_drag`.",
                    "type": "array",
                },
                "end_coordinate": {
                    "description": "(x, y): The ending x (pixels from the left edge) and y (pixels from the top edge) coordinates. Required only by `action=left_click_drag`.",
                    "type": "array",
                },
                "coordinate": {
                    "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required by `action=mouse_move, action=left_click, action=right_click, action=double_click`.",
                    "type": "array",
                },
                "pixels": {
                    "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.",
                    "type": "number",
                },
            },
            "required": ["action"],
            "type": "object",
        },
        "args_format": "Format the arguments as a JSON object.",
    },
}

BASE_PROMPT_TEMPLATE = (
    """# Tools

You MUST call a single function to assist with the user query. Do not call multiple functions, and do not answer the user's query without calling a function.

You are provided with function signatures within <tools></tools> XML tags:
<tools>"""
    + json.dumps(QWEN_ACTION_SPACE)
    + """</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
)


class QwenModelTemplate(BaseClickModelTemplate):
    def instruction_text(self, instruction: str) -> str:
        # Use the exact same system prompt as QwenVLClient
        width = 1024  # Default resolution, could be parameterized
        height = 768

        system_prompt = BASE_PROMPT_TEMPLATE.replace("{RES_WIDTH}", str(width)).replace(
            "{RES_HEIGHT}", str(height)
        )

        return system_prompt + f"\n\n{instruction}"

    def extract_coordinates(
        self, response: str, image_dimensions: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Extract coordinates using the exact same logic as QwenVLClient.parse_prediction()"""

        # Use the exact same parsing logic as QwenVLClient
        content = response

        print(f"Content: {content}")

        tool_call_match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, flags=re.DOTALL
        )
        if not tool_call_match:
            return None

        try:
            json_text = tool_call_match.group(1)
            data = json.loads(json_text)

            if "arguments" not in data:
                return None

            args = data["arguments"]
            action_str = args.get("action")

            if not action_str:
                return None

            pred_x = None
            pred_y = None
            if (
                "coordinate" in args
                and isinstance(args["coordinate"], list)
                and len(args["coordinate"]) == 2
            ):
                pred_x = int(args["coordinate"][0])
                pred_y = int(args["coordinate"][1])

            if pred_x is not None and pred_y is not None:
                return float(pred_x), float(pred_y)

        except Exception as e:
            print(f"Error parsing prediction: {e}")

        return None

    def format_messages(self, base64_image: str, prompt_text: str) -> list[dict]:
        """Format messages for Qwen's expected format."""
        assert base64_image.startswith("data:image/png;base64,"), (
            f"Invalid image data URI {base64_image[:30]}..."
        )

        # Qwen expects a single message with both text and image
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_image}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]


# eve clicks run --model-template qwen --output gs://induction-labs/evals/clicks/Qwen/Qwen2.5-VL-7B-Instruct/test_clicks  --sample-size 10

# eve clicks run --num-workers 32 --model-template qwen  --print-cmd | mdl k8s eve Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-3B-Instruct
