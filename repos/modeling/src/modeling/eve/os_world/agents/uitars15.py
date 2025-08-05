import ast
import base64
import logging
import math
import re
from typing import Literal

import aiohttp

logger = logging.getLogger("desktopenv.agent")

FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def convert_point_to_coordinates(text, is_answer=False):
    # Match the two numbers inside <point> tags
    pattern = r"<point>(\d+)\s+(\d+)</point>"

    def replace_match(match):
        x1, y1 = map(int, match.groups())
        x = (x1 + x1) // 2  # Truncated integer division
        y = (y1 + y1) // 2  # Truncated integer division
        if is_answer:
            return f"({x},{y})"  # Return only in (x, y) format
        return f"({x},{y})"  # Return in labeled format

    # Remove [EOS] and replace <point> coordinates
    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()


# 定义一个函数来解析每个 action
def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode="eval")

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值, 这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {"function": func_name, "args": kwargs}

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None


def escape_single_quotes(text):
    # 匹配未转义的单引号 (不匹配 \\')
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: float, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def linear_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    if width * height > max_pixels:
        """
        If the image exceeds or falls below the pixel limits, 
        calculate a scaling factor (resize_factor) so that the total number of pixels 
        is reduced to be equal to or less than max_pixels. 
        This scaling factor is calculated using the square root to ensure the aspect ratio remains unchanged, 
        allowing the original relative coordinates to be reused directly without conversion.
        """
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = (
            math.ceil(width * resize_factor),
            math.ceil(height * resize_factor),
        )

    return height, width


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def parse_action_to_structure_output(
    text,
    factor,
    origin_resized_height,
    origin_resized_width,
    model_type="qwen25vl",
    max_pixels=16384 * 28 * 28,
    min_pixels=100 * 28 * 28,
):
    text = text.strip()

    if "<point>" in text:
        text = convert_point_to_coordinates(text)
    if "start_point=" in text:
        text = text.replace("start_point=", "start_box=")
    if "end_point=" in text:
        text = text.replace("end_point=", "end_box=")
    if "point=" in text:
        text = text.replace("point=", "start_box=")

    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height,
            origin_resized_width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        # thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action: |$)"
        # thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action: |$)"
        # thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        # thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action:" in text
    action_str = text.split("Action: ")[-1]

    tmp_all_action = action_str.split("')\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        all_action.append(action_str)

    parsed_actions = [
        parse_action(action.replace("\n", "\\n").lstrip()) for action in all_action
    ]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action, strict=False):
        if action_instance is None:
            print(f"Action can't parse: {raw_str}")
            raise ValueError(f"Action can't parse: {raw_str}")
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "":
                continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param

            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                # Qwen2.5vl output absolute coordinates, qwen2vl output relative coordinates
                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(float(num / smart_resize_height))
                        else:
                            float_numbers.append(float(num / smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [
                        float_numbers[0],
                        float_numbers[1],
                        float_numbers[0],
                        float_numbers[1],
                    ]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append(
            {
                "reflection": reflection,
                "thought": thought,
                "action_type": action_type,
                "action_inputs": action_inputs,
                "text": text,
            }
        )
    return actions


def parsing_response_to_pyautogui_code(
    responses,
    image_height: int,
    image_width: int,
    input_swap: bool = True,
    platform: str = "Ubuntu",
) -> str:
    """
    Parse the output of model M into an action in OSWorld, generating a pyautogui code string.

    Parameters:
        response: A dictionary containing the model output, structured as:
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }

    Returns:
        A generated pyautogui code string.
    """

    pyautogui_code = "import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        observation = response.get("observation", "")
        thought = response.get("thought", "")

        if response_id == 0:
            pyautogui_code += (
                f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
            )
        else:
            pyautogui_code += "\ntime.sleep(1)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})

        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"

            elif hotkey == "arrowup":
                hotkey = "up"

            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = " "
                    convert_keys.append(key)
                pyautogui_code += (
                    f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"
                )

        elif action_type in ["press", "keydown"]:
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if key_to_press == "arrowleft":
                key_to_press = "left"

            elif key_to_press == "arrowright":
                key_to_press = "right"

            elif key_to_press == "arrowup":
                key_to_press = "up"

            elif key_to_press == "arrowdown":
                key_to_press = "down"

            elif key_to_press == "space":
                key_to_press = " "

            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.keyDown({key_to_press!r})"

        elif action_type in ["release", "keyup"]:
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if key_to_press == "arrowleft":
                key_to_press = "left"

            elif key_to_press == "arrowright":
                key_to_press = "right"

            elif key_to_press == "arrowup":
                key_to_press = "up"

            elif key_to_press == "arrowdown":
                key_to_press = "down"

            elif key_to_press == "space":
                key_to_press = " "

            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.keyUp({key_to_press!r})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith(("\n", "\\n")):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += "\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += "\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += "\ntime.sleep(0.5)\n"
                    if content.endswith(("\n", "\\n")):
                        pyautogui_code += "\npyautogui.press('enter')"

                elif "<Backspace>" in stripped_content:
                    assert stripped_content == "<Backspace>", (
                        f"Only support <Backspace> action, {stripped_content=}"
                    )
                    pyautogui_code += "\npyautogui.press('backspace')\n"

                else:
                    pyautogui_code += (
                        f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    )
                    pyautogui_code += "\ntime.sleep(0.5)\n"
                    if content.endswith(("\n", "\\n")):
                        pyautogui_code += "\npyautogui.press('enter')"

        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                # sx = round(float((x1 + x2) / 2) * image_width, 3)
                # sy = round(float((y1 + y2) / 2) * image_height, 3)
                sx = int(round(x1 * 1000, 3))
                sy = int(round(y1 * 1000, 3))

                x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                # ex = round(float((x1 + x2) / 2) * image_width, 3)
                # ey = round(float((y1 + y2) / 2) * image_height, 3)
                ex = int(round(x1 * 1000, 3))
                ey = int(round(y1 * 1000, 3))
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, _x2, _y2 = eval(
                    start_box
                )  # Assuming box is in [x1, y1, x2, y2]
                # x = round(float((x1 + x2) / 2) * image_width, 3)
                # y = round(float((y1 + y2) / 2) * image_height, 3)

                x = int(round(x1 * 1000, 3))
                y = int(round(y1 * 1000, 3))

                # # 先点对应区域, 再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")

            if x is None:
                if "up" in direction.lower():
                    if platform.lower() == "ubuntu":
                        pyautogui_code += "\npyautogui.scroll(-5)"
                    elif platform.lower() == "windows":
                        pyautogui_code += "\npyautogui.scroll(-50)"
                elif "down" in direction.lower():
                    if platform.lower() == "ubuntu":
                        pyautogui_code += "\npyautogui.scroll(5)"
                    elif platform.lower() == "windows":
                        pyautogui_code += "\npyautogui.scroll(50)"
            else:
                if "up" in direction.lower():
                    if platform.lower() == "ubuntu":
                        pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                    elif platform.lower() == "windows":
                        pyautogui_code += f"\npyautogui.scroll(50, x={x}, y={y})"
                elif "down" in direction.lower():
                    if platform.lower() == "ubuntu":
                        pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"
                    elif platform.lower() == "windows":
                        pyautogui_code += f"\npyautogui.scroll(-50, x={x}, y={y})"

        elif action_type in [
            "click",
            "left_single",
            "left_double",
            "right_single",
            "hover",
        ]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, _x2, _y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box

                x = int(round(x1 * 1000, 3))
                y = int(round(y1 * 1000, 3))
                # if x1 > 1 or y1 > 1:
                #     x = round(x1 * 1000, 3)
                #     y = round(y1 * 1000, 3)
                # else:
                #     x = round(float((x1 + x2) / 2) * image_width, 3)
                #     y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += (
                        f"\npyautogui.doubleClick({x}, {y}, button='left')"
                    )
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

        elif action_type in ["finished"]:
            pyautogui_code = "DONE"

        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code


def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(
                r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action
            )

            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(
                    f"{coord_type}='({x},{y})'",
                    f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'",
                )
            processed_actions.append(updated_action)

        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


COMPUTER_USE_15 = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() # Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


## Note
- Use {language} in `Thought` part.
- {thought_mode}

## User Instruction
{instruction}
"""

THOUGHT_BRIEF = "Generate a well-defined and practical strategy in the `Thought` section, summarizing your next move and its objective."
THOUGHT_LONG = "Compose a step-by-step approach in the `Thought` part, specifying your next action and its focus."
LANG_EN = "English"
LANG_ZH = "Chinese"

COMPUTER_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags
After the <think> tags, you should place final answer, which concludes your summarized thought and your action.

For example,
```
<think>detailed reasoning content here</think>
Thought: a small plan and finally summarize your next action (with its target element) in one sentence
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Output Example
<think>Now that...</think>
Thought: Let's click ...
Action: click(point='<point>100 200</point>')

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- If you have executed several same actions (like repeatedly clicking the same point) but the screen keeps no change, please try to execute a modified action when necessary.

## User Instruction
{instruction}
"""

MOBILE_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""

COMPUTER_USE_NO_THINKING = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use Chinese in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""


class UITarsAgent:
    """
    UI-TARS Agent based on Seed1.5-VL model implementation.
    Integrates the GUI folder UI-TARS-1.5 implementation with the mm_agents architecture.
    """

    def __init__(
        self,
        # Generation settings
        model_endpoint: str,
        use_vllm: bool = False,
        # Prompt settings
        screenshot_pyautogui_prompt: str = "uitars_v1",
        # Parse settings
        which_parsed_actions: str = "all",
        # Outside infos
        max_steps: int = 100,
        temperature: float = 1,
        # UI-TARS specific settings
        use_thinking: bool = True,
        language: Literal["zh", "en"] = "zh",
    ):
        """
        Initialize UI-TARS Agent.

        Args:
            api_key: API key for the model service
            base_url: Base URL for the API service
            screenshot_pyautogui_prompt: Prompt version
            which_parsed_actions: Which actions to parse
            max_steps: Maximum steps for the agent
            use_thinking: Whether to use thinking mode
            language: Language for responses
            openai_client: OpenAI client instance
        """

        self.logger = logger
        self.language = language
        self.use_vllm = use_vllm
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []

        self.system_prompt = COMPUTER_USE_15

        self.action_parse_res_factor = 1000
        self.model_type = "doubao"
        self.history_n = 5
        self.platform = "ubuntu"

        self.use_thinking = use_thinking

        self.model_endpoint = model_endpoint
        self.temperature = temperature

        self.aiohttp_client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300, connect=5)
        )

    def reset(self, _logger=None):
        global logger
        logger = (
            _logger if _logger is not None else logging.getLogger("desktopenv.agent")
        )

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []

    def pretty_print_messages(self, messages):
        """Pretty print messages while hiding base64 encoded images."""

        def format_message(msg):
            if not isinstance(msg, dict):
                return str(msg)

            formatted = {}
            for key, value in msg.items():
                if key == "content":
                    if isinstance(value, list):
                        formatted_content = []
                        for item in value:
                            if isinstance(item, dict) and "type" in item:
                                if item["type"] == "image_url" and "image_url" in item:
                                    # Replace base64 image with placeholder
                                    formatted_content.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": "[BASE64_IMAGE_DATA]"},
                                        }
                                    )
                                else:
                                    formatted_content.append(item)
                            else:
                                formatted_content.append(item)
                        formatted[key] = formatted_content
                    else:
                        formatted[key] = value
                else:
                    formatted[key] = value
            return formatted

        if isinstance(messages, list):
            return [format_message(msg) for msg in messages]
        return format_message(messages)

    # def inference_without_thinking(self, messages):
    #     print(messages)
    #     return None

    async def inference(self, messages):
        async with self.aiohttp_client.post(
            self.model_endpoint,
            json={"messages": messages, "temperature": self.temperature},
        ) as response:
            if response.status != 200:
                raise Exception(f"Error in inference: {response.status}")

            data = await response.json()
            if self.use_vllm:
                return data["choices"][0]["message"]["content"]
            else:
                return data["response"]

    async def predict(
        self, task_instruction: str, obs: dict
    ) -> tuple[str | dict | None, list]:
        """Predict the next action based on the current observation."""

        self.task_instruction = task_instruction

        assert len(self.observations) == len(self.actions) and len(self.actions) == len(
            self.thoughts
        ), (
            f"The number of observations and actions should be the same. Got {len(self.observations)=} {len(self.actions)=} {len(self.thoughts)=}"
        )

        # Convert binary screenshot to base64 if needed
        screenshot = obs["screenshot"]
        if isinstance(screenshot, bytes):
            screenshot = base64.b64encode(screenshot).decode("utf-8")

        self.history_images.append(screenshot)

        self.observations.append({"screenshot": screenshot, "accessibility_tree": None})

        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n :]

        images = self.history_images

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt.format(
                            instruction=task_instruction,
                            language=LANG_ZH if self.language == "zh" else LANG_EN,
                            thought_mode=THOUGHT_LONG
                            if self.use_thinking
                            else THOUGHT_BRIEF,
                        ),
                    }
                ],
            }
        ]

        def create_image_message(image_data):
            if self.use_vllm:
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                }
            else:
                return {"type": "image", "image": f"data:image/png;base64,{image_data}"}

        image_num = 0
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                # send at most history_n images to the model
                if history_idx + self.history_n > len(self.history_responses):
                    messages.append(
                        {
                            "role": "user",
                            "content": [create_image_message(images[image_num])],
                        }
                    )
                    image_num += 1

                messages.append({"role": "assistant", "content": history_response})
            messages.append(
                {"role": "user", "content": [create_image_message(images[image_num])]}
            )
            image_num += 1
        else:
            messages.append(
                {"role": "user", "content": [create_image_message(images[image_num])]}
            )
            image_num += 1

        try_times = 3
        origin_resized_height = 1080
        origin_resized_width = 1920
        prediction = None
        while True:
            if try_times <= 0:
                self.logger.error(
                    "Reach max retry times to fetch response from client, as error flag."
                )
                return prediction, ["INTERNAL_FAIL"]

            try:
                prediction = await self.inference(messages)
            except Exception as e:
                self.logger.error(
                    f"Error when fetching response from client, with error:\n{e}"
                )
                prediction = None
                try_times -= 1
                continue

            try:
                parsed_dict = parse_action_to_structure_output(
                    prediction,
                    self.action_parse_res_factor,
                    origin_resized_height,
                    origin_resized_width,
                    self.model_type,
                )
                parsed_pyautogui_code = parsing_response_to_pyautogui_code(
                    parsed_dict,
                    origin_resized_height,
                    origin_resized_width,
                    platform=self.platform,
                    input_swap=False,
                )
                break
            except Exception as e:
                self.logger.error(
                    f"Error when parsing response from client, with error:\n{e}"
                )
                prediction = None
                try_times -= 1

        self.history_responses.append(prediction)

        try:
            parsed_dict = parse_action_to_structure_output(
                prediction,
                self.action_parse_res_factor,
                origin_resized_height,
                origin_resized_width,
                self.model_type,
            )
            parsed_pyautogui_code = parsing_response_to_pyautogui_code(
                parsed_dict,
                origin_resized_height,
                origin_resized_width,
                platform=self.platform,
                input_swap=False,
            )

        except Exception as e:
            self.logger.error(f"Parsing action error: {prediction}, with error:\n{e}")
            return prediction, ["PARSE_FAIL"]

        thoughts = ""
        for parsed_response in parsed_dict:
            if parsed_response.get("thought"):
                thoughts += parsed_response["thought"]
        self.thoughts.append(thoughts)
        for parsed_response in parsed_dict:
            if "action_type" in parsed_response:
                if parsed_response["action_type"] == FINISH_WORD:
                    self.actions.append(["DONE"])

                    return prediction, ["DONE"]

                elif parsed_response["action_type"] == WAIT_WORD:
                    self.actions.append(["WAIT"])

                    return prediction, ["WAIT"]

                elif parsed_response["action_type"] == ENV_FAIL_WORD:
                    self.actions.append(["FAIL"])
                    return prediction, ["FAIL"]

        self.actions.append([parsed_pyautogui_code])

        return prediction, [parsed_pyautogui_code]
