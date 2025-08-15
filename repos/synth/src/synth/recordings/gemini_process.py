from __future__ import annotations

import base64
import datetime
import json
import multiprocessing
import secrets
import string
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from enum import Enum
from functools import wraps
from multiprocessing import Manager, Process
from multiprocessing.managers import ListProxy
from queue import Empty, Queue
from typing import Literal

import av
import dotenv
import gcsfs
import litellm
import pandas as pd
import PIL
import PIL.Image
import tqdm
from PIL import ImageDraw
from pydantic import BaseModel, computed_field
from smart_open import open as smart_open
from synapse.utils.logging import configure_logging, logging
from synapse.video_loader.typess import VideoResolution, resolution_1080p
from synapse.video_loader.video import (
    VideoMetadataFetcher,
    download_video_with_ffmpeg_copy,
    get_resize_filter,
    smart_resize,
)
from synth.recordings.action_models import FinishedAction
from synth.recordings.image_utils import pil_to_bytes
from synth.recordings.parse_actions import (
    Action,
    HotkeyAction,
    Point,
    ScrollAction,
    TypeAction,
    parse_actions,
)
from synth.recordings.synth_captions_generated_samples import (
    # COMMON_INSTRUCTION,
    extract_frames_by_pts_from_container,
    get_actions,
)
from synth.recordings.transform_coordinates import (
    Platform,
    RecordingMetadata,
    transform_action_coords_list,
)

dotenv.load_dotenv()

logger = configure_logging(__file__, logging.INFO)

litellm.drop_params = True
# DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"
DEFAULT_MODEL = "o3"


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry on

    Returns:
        Decorated function that retries on failure
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        # Final attempt failed, re-raise the exception
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise e

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


# In addition, take note of whether the screen is ready or is in a transition state. Sometimes, the screenshot is taken before the page has finished loading or the application has finished rendering. In these rare cases, add `is_loading: true`. Default to `is_loading: false` unless there's clear evidence it's still loading.

COMMON_INSTRUCTION = """
# Instruction
Simulate the user's internal monologue as they perform a computer task. You are given the action the user just took in the current screenshot. In a first-person, present-tense inner voice, explain why this specific action is the right move to advance or complete the task. Be concrete and insightful—reference on-screen cues, trade-offs, and how this step helps solve the problem. Focus strictly on the action just taken; you may briefly allude to prior or next steps, but do not directly reference them.

## Style
- Write in the future tense (i.e., "I'll do X") and in a simple, concise tone. It should only be a few sentences long, no Markdown. 
- Avoid fancy words and physical-world metaphors—stick to what's on screen. 
- Include exact text typed and any keyboard shortcuts used.
- When referencing elements on the screen, include their position in the screenshot, e.g., "the button in the top right corner" or "the text field in the middle of the screen".
- Click positions have been highlighted with a red circle in the screenshot they are taken from. Do not reference the red circle in the screenshot, it is only there to help you see where the click was made.
- Start with a description of what is on the screen or what has changed since the last action. Then write the reasoning to decide on what action to perform next.

## Example
Here's an example of the style I'm looking for:

Chrome is open on the screen, but I don't see a favorites folder. To add it, I first need to open the settings menu in the browser. I noticed there's a three-dot icon in the upper right corner of the browser window, which is the entry point for the settings menu. I'll click on it to reveal more options.
""".strip()

PROMPT_WITHOUT_NEXT = (
    COMMON_INSTRUCTION
    + "\n"
    + """## Context
Here's the task the user has been given:
{instruction}

Here's the reasoning the user has done so far:
{old_turns}

**Here's the new action the user took that you should write the monolouge for**:
{new_action}

Now, write the user's internal monologue about what is on the screen, how they decided to take this action, and what they plan to do next. The monologue should be focused on the action just taken.
"""
)


class ModelProvider(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"


@retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    exceptions=(Exception,),
)
def call_model(
    model_name: str,
    messages: list[dict],
    max_tokens: int = 4096,
    reasoning_effort: Literal["low", "medium", "high"] = "high",
    response_format: dict | None = None,
) -> tuple[str, dict]:
    """
    Unified interface for calling different AI models using LiteLLM.

    Messages should contain properly formatted content with interleaved text and images.
    For multi-modal content, use the format:
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Your text here"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        }
    ]

    Args:
        model_name: Name of the model to use (e.g., "gemini-2.0-flash", "gpt-4o", "claude-3-5-sonnet-20241022")
        messages: List of message dictionaries with 'role' and properly formatted 'content'
        max_tokens: Maximum number of tokens to generate

    Returns:
        Tuple of (generated text response, cost info dict)
    """
    try:
        response = litellm.completion(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
        )

        # Extract cost information from the response
        cost_info = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0)
            if response.usage
            else 0,
            "completion_tokens": getattr(response.usage, "completion_tokens", 0)
            if response.usage
            else 0,
            "total_tokens": getattr(response.usage, "total_tokens", 0)
            if response.usage
            else 0,
            "model": model_name,
        }

        # Add cache hit information if available
        if response.usage:
            # OpenAI/Anthropic style cache hits
            cost_info["prompt_tokens_cached"] = (
                response.usage.prompt_tokens_details.cached_tokens
                if hasattr(response.usage, "prompt_tokens_details")
                and response.usage.prompt_tokens_details
                else 0
            )

            # Calculate cache hit rate
            cached_tokens = cost_info["prompt_tokens_cached"]
            total_prompt_tokens = cost_info["prompt_tokens"]
            if total_prompt_tokens > 0:
                cost_info["prompt_cache_hit_rate"] = cached_tokens / total_prompt_tokens
            else:
                cost_info["prompt_cache_hit_rate"] = 0.0
        else:
            cost_info["prompt_tokens_cached"] = 0
            cost_info["prompt_cache_hit_rate"] = 0.0

        # Add cost in USD if available
        if (
            hasattr(response, "_hidden_params")
            and "response_cost" in response._hidden_params
        ):
            cost_info["cost_usd"] = response._hidden_params["response_cost"]
        elif hasattr(response, "response_cost"):
            cost_info["cost_usd"] = response.response_cost
        else:
            cost_info["cost_usd"] = 0.0
        content = response.choices[0].message.content.strip()

        return content, cost_info

    except Exception as e:
        logger.error(f"Error calling model {model_name}: {e}")
        raise


def is_close(a: Point, b: Point, tol: float = 10) -> bool:
    return abs(a.x - b.x) <= tol and abs(a.y - b.y) <= tol


def is_good_action_sequence(actions: list[Action], min_typing_percent=0.2) -> bool:
    # Check that there is at least one typing action
    typing_actions = sum(isinstance(action.action, TypeAction) for action in actions)
    if typing_actions / len(actions) < min_typing_percent:
        return False
    hotkey_actions = sum(isinstance(action.action, HotkeyAction) for action in actions)
    if hotkey_actions >= 5:
        return False

    ""
    return True


def reverse_chunk_actions_to_length(
    actions: list[Action],
    min_length: int = 5,
    max_length: int = 10,
) -> list[list[Action]]:
    """
    Create action blocks using a sliding window approach from the end.

    Starts with the last block (including FinishedAction), then slides the window
    backwards. If a block is good, it's added and the window jumps back by the full
    length. If not good, the window slides back by just 1 position.

    Args:
        actions: List of actions to chunk
        length: Length of each chunk (default: 5, but last chunk gets +1 for FinishedAction)

    Returns:
        List of action blocks, each being a list of actions
    """
    actions = actions.copy()

    actions.append(
        Action(
            action=FinishedAction(),
            timestamp=actions[-1].end_timestamp + SS_DELAY + BEFORE_ACTION_BUFFER,
            end_timestamp=actions[-1].end_timestamp + SS_DELAY + BEFORE_ACTION_BUFFER,
        )
    )
    if len(actions) < min_length:
        return []

    action_blocks = []

    # Start with the last block: take the last (length-1) actions + add FinishedAction
    window_end = len(actions)

    # Continue sliding window backwards
    while True:
        # Extract current window
        assert window_end <= len(actions)
        window_start = max(0, window_end - max_length)
        if window_end - window_start < min_length:
            break
        block = actions[window_start:window_end]
        # Skip if block is too short
        assert len(block) == window_end - window_start, (
            f"{len(block)=}, {max_length=} {window_end=}, {window_start=}"
        )

        # Check if this block is good
        if is_good_action_sequence(block):
            action_blocks.append(block)
            # Jump back by full length
            window_end = window_start
        else:
            # Slide back by 1
            window_end -= 1

    return action_blocks


def segment_actions_by_time_gaps(
    actions: list[Action],
    gap_threshold: float = 5.0,
) -> list[list[Action]]:
    """
    Segment actions into chunks based on time gaps between consecutive actions.

    Args:
        actions: List of actions to segment
        gap_threshold: Time threshold in seconds for creating new segment (default: 5.0)
        actions_per_segment: Maximum number of actions per segment (default: 5)

    Returns:
        List of action chunks, where each chunk is a list of consecutive actions
        without large time gaps
    """
    if not actions:
        return []

    # Sort actions by timestamp to ensure proper ordering
    sorted_actions = sorted(actions, key=lambda a: a.timestamp)

    segments = []
    current_segment = [sorted_actions[0]]

    for i in range(1, len(sorted_actions)):
        prev_action = sorted_actions[i - 1]
        current_action = sorted_actions[i]

        # Calculate gap between end of previous action and start of current action
        time_gap = current_action.timestamp - prev_action.end_timestamp

        if time_gap > gap_threshold:
            # Gap is too large, start a new segment
            segments.append(current_segment)
            current_segment = [current_action]
        else:
            # Continue the current segment
            current_segment.append(current_action)

    # Don't forget the last segment
    if current_segment:
        segments.append(current_segment)

    return segments


def is_arrow_action(action: Action) -> bool:
    # Return true if arrow action without modifiers
    return (
        isinstance(action.action, HotkeyAction)
        and (action.action.key in ["up", "down", "left", "right"])
        and not action.action.modifiers
    )


def combine_arrow_actions(actions: list[Action], time_threshold=1.0) -> list[Action]:
    new_actions: list[Action] = []
    i = 0
    while i < len(actions):
        action = actions[i]
        i += 1
        if not is_arrow_action(action):
            new_actions.append(action)
            continue
        combined_arrow_action = action
        while (
            i < len(actions)
            and is_arrow_action(actions[i])
            and actions[i].end_timestamp - combined_arrow_action.timestamp
            < time_threshold
        ):
            next_action = actions[i]
            combined_arrow_action.action.key += " " + next_action.action.key
            combined_arrow_action.end_timestamp = next_action.end_timestamp
            i += 1

        new_actions.append(combined_arrow_action)
    return new_actions


def combine_scroll_actions(actions: list[Action]) -> list[Action]:
    new_actions: list[Action] = []
    i = 0
    while i < len(actions):
        action = actions[i]
        i += 1
        if not isinstance(action.action, ScrollAction):
            new_actions.append(action)
            continue
        combined_scroll_action = action
        while (
            i < len(actions)
            and isinstance(actions[i].action, ScrollAction)
            and is_close(action.action.point, actions[i].action.point, tol=10)
        ):
            next_action = actions[i]
            combined_scroll_action.action.displacement = (
                combined_scroll_action.action.displacement[0]
                + next_action.action.displacement[0],
                combined_scroll_action.action.displacement[1]
                + next_action.action.displacement[1],
            )
            combined_scroll_action.end_timestamp = next_action.end_timestamp
            i += 1
        combined_scroll_action.action.direction = (
            "up" if combined_scroll_action.action.displacement[1] > 0 else "down"
        )
        new_actions.append(combined_scroll_action)
    return new_actions


def combine_actions(actions: list[Action]) -> list[Action]:
    actions = combine_arrow_actions(actions)
    actions = combine_scroll_actions(actions)
    return actions


SS_DELAY = 0.20
BEFORE_ACTION_BUFFER = 0.0


def mean(a: float, b: float, weight=0.5) -> float:
    return a * (1 - weight) + b * weight


def get_timesteps_range(actions: list[Action]) -> list[float]:
    timestamps: list[float] = []
    timestamps.append(actions[0].timestamp - SS_DELAY)
    for i, prev_action in enumerate(actions[:-1]):
        next_action = actions[i + 1]
        end_timestamp = prev_action.end_timestamp - BEFORE_ACTION_BUFFER
        end_timestamp = max(
            end_timestamp,
            mean(
                prev_action.end_timestamp,
                next_action.timestamp - BEFORE_ACTION_BUFFER,
                0.4,
            ),
        )
        end_timestamp = min(
            end_timestamp,
            next_action.timestamp - BEFORE_ACTION_BUFFER,
        )
        timestamps.append(end_timestamp)

    timestamps.append(actions[-1].end_timestamp + SS_DELAY)

    assert len(timestamps) >= 1, "No timestamps generated from actions"
    return timestamps


def text_content(text: str):
    return {
        "type": "text",
        "text": text,
    }


class ImageDetail(str, Enum):
    HIGH = "high"
    LOW = "low"


def image_content(image: bytes, detail=ImageDetail.HIGH):
    image_b64 = base64.b64encode(image).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_b64}",
            "detail": detail.value,
        },
    }


def convert_image_detail(content: dict, detail: ImageDetail) -> dict:
    if content["type"] != "image_url":
        return content
    content = deepcopy(content)
    content["image_url"]["detail"] = detail.value
    return content


def json_schema(schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": "Response", "schema": schema},
        "strict": True,  # ask the provider to strictly enforce the schema
    }


def pyd_model_to_response_format(
    model: type[BaseModel], *, name: str | None = None, strict: bool = True
):
    schema = model.model_json_schema()
    # (Optional belt & suspenders) ensure providers see no extras allowed
    schema.setdefault("additionalProperties", False)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or model.__name__,
            "schema": schema,
        },
    }


class FilterResponseSchema(BaseModel):
    analysis_summary: str
    remove: bool


def get_should_filter(
    base_prompt: list[dict],
    model_name: str = DEFAULT_MODEL,
) -> tuple[FilterResponseSchema, dict] | None:
    content = base_prompt.copy()
    content.append(
        {
            "type": "text",
            "text": """
You are evaluating whether a user recording shows productive, goal-oriented work that would be valuable for training an AI assistant. 

**REMOVE (answer "yes") recordings that contain:**

1. **Passive consumption activities:**
   - Watching videos, movies, or entertainment content
   - Reading articles, news, or social media feeds
   - Scrolling through content without clear purpose
   - Browsing websites without taking meaningful actions

2. **Aimless or idle behavior:**
   - Random clicking or navigation without purpose
   - Repeatedly switching between tabs/applications without progress
   - Scrolling back and forth in the same location
   - Opening and closing applications without using them
   - Extended periods of inactivity or hesitation

3. **Technical issues or mismatched data:**
   - Actions that don't correspond to what's shown in the screenshots
   - Evidence of multi-monitor setups where actions occur off-screen
   - Clear recording software bugs or glitches
   - Screenshots that don't match the described actions

**KEEP (answer "no") recordings that show:**
- Clear task completion or progress toward goals
- Creating, editing, or managing content/documents
- Problem-solving activities
- Software configuration or meaningful interactions
- Purposeful navigation with clear intent

Look for patterns of intentional action sequences that demonstrate the user working toward completing specific tasks.
""".strip(),
        }
    )

    response_format = pyd_model_to_response_format(FilterResponseSchema)
    try:
        model_response_text, cost_info = call_model(
            model_name=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            response_format=response_format,
        )

    except Exception as e:
        logger.error(f"Error determining if content should be filtered: {e}")
        return None

    try:
        response = FilterResponseSchema.model_validate_json(model_response_text)

    except Exception as e:
        logger.error(f"{model_response_text=}")
        logger.error(f"Error validating filter response: {e}")
        return None

    # return False, model_response_text, cost_info
    return response, cost_info


class VideoInstructionResponse(BaseModel):
    is_productive: bool
    instruction: str


def get_video_instruction(
    base_prompt: list[dict],
    model_name: str = DEFAULT_MODEL,
) -> tuple[VideoInstructionResponse, dict] | None:
    """
    Generate an instruction based on the first frame and the actions.

    Returns:
        Tuple of (response_text, instruction, cost_info) or None if failed
    """

    # Build interleaved content with text and images
    content = base_prompt.copy()

    # Add final instruction
    content.append(
        {
            "type": "text",
            "text": """
# Instruction
Analyze what action the user took in the screenshots.
You have two tasks:
1. Decide whether the user is performing a productive task that would be valuable for training an AI assistant.
2. Write a plausible instruction that would result in the behavior shown in the screenshots.

## Task 1: Decide whether to keep or filter the recording
Mark `is_productive: false` for recordings that show:
**Passive consumption activities:** such as watching videos, reading articles, or aimless browsing without clear purpose.
**Aimless or idle behavior:** such as random clicking, switching between tabs without progress, or extended inactivity.

Mark `is_productive: true` for recordings that show:
- Clear task completion or progress toward goals
- Problem-solving activities
- Purposeful navigation with clear intent


## Task 2: Write a plausible instruction that would result in the behaviour shown in the screenshots.
If you marked the recording as productive, write a clear, actionable instruction that an AI assistant could follow to replicate the observed behavior. The instruction should:
Be specific and actionable: Include the exact steps, applications, or websites involved.
Capture the user's intent: Explain what the user is trying to accomplish, not just the mechanical actions.
Use natural language: Write as if you're giving instructions to a helpful assistant.
Include relevant context: Mention any specific data, files, or information being worked with.

If the recording is not productive, simply write `instruction: "This recording is not productive."` followed by a short explanation of why it is not productive.
""".strip(),
        }
    )
    response_format = pyd_model_to_response_format(VideoInstructionResponse)
    try:
        model_response_text, cost_info = call_model(
            model_name=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            response_format=response_format,
        )

    except Exception as e:
        logger.error(f"Error generating video instruction: {e}")
        return None
    response = VideoInstructionResponse.model_validate_json(model_response_text)

    return response, cost_info


class SaveAction(BaseModel):
    step: int
    image: str
    action: str
    thinking: str
    reward: float
    frame_metadata: FrameMetadata

    @computed_field
    @property
    def text(self) -> str:
        """
        Generate a text representation of the action.
        """
        return f"Thought: {self.thinking}\nAction: {self.action}"


class TrainSample(BaseModel):
    actions: list[SaveAction]
    instruction: str
    metadata: dict | None = None


class ThinkingTextResponse(BaseModel):
    user_monologue: str
    # is_loading: bool = False


def get_thinking_texts(
    base_prompt: list[dict],
    instruction: str,
    actions: list[Action],
    model_name: str = DEFAULT_MODEL,
) -> tuple[list[ThinkingTextResponse], list[dict]]:
    """
    Generate thinking texts for each action.

    Returns:
        Tuple of (thinking_texts, cost_infos) where cost_infos is a list of cost dicts for each call
    """
    thinking_texts: list[ThinkingTextResponse] = []
    cost_infos = []
    old_turns: list[str] = []
    response_format = pyd_model_to_response_format(ThinkingTextResponse)
    for i, action in enumerate(actions):
        text_prompt = PROMPT_WITHOUT_NEXT.format(
            instruction=instruction,
            old_turns=old_turns,
            new_action=f"Action #{i + 1}: {action.dump_to_text()}",
        )

        # Build interleaved content with text and the two consecutive frames
        content = base_prompt.copy()
        content.append({"type": "text", "text": text_prompt})

        # Add the frame before the action

        try:
            model_response_text, cost_info = call_model(
                model_name=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                response_format=response_format,
            )
            model_response = ThinkingTextResponse.model_validate_json(
                model_response_text
            )

            thinking_texts.append(model_response)
            cost_infos.append(cost_info)
            logger.debug(f"Thinking text for action {i}: {model_response}")
            old_turns.append(
                f"Action #{i + 1}: {model_response.user_monologue}\n{action.dump_to_text()}"
            )

        except Exception as e:
            logger.error(f"Error generating thinking text for action {i}: {e}")
            # Use a fallback thinking text
            fallback_text = (
                f"Next, I need to perform this action: {action.dump_to_text()}"
            )
            thinking_texts.append(ThinkingTextResponse(user_monologue=fallback_text))
            # Add empty cost info for fallback
            cost_infos.append(
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "model": model_name,
                    "cost_usd": 0.0,
                    "prompt_tokens_cached": 0,
                    "prompt_cache_hit_rate": 0.0,
                }
            )
            old_turns.append(f"{fallback_text}\n{action.dump_to_text()}")

    return thinking_texts, cost_infos


def draw_red_circle(
    image: PIL.Image.Image, point: Point, radius: int = 40, width: int = 10
) -> PIL.Image.Image:
    """
    Draw a red circle on the image at the specified point.
    """
    image = image.copy()
    image = image.copy()
    image = image.convert("L").convert(
        "RGB"
    )  # L = grayscale, then back to RGB for color drawing
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [point.x - radius, point.y - radius, point.x + radius, point.y + radius],
        outline="red",
        width=width,
    )
    return image


def build_base_prompt(
    last_frame: PIL.Image.Image,
    rest_frames: list[PIL.Image.Image],
    actions: list[Action],
) -> list[dict]:
    prompt: list[dict] = [
        text_content(
            "The following are a series of screenshots and the actions a user took from a video recording of performing a task. "
            "The `i`th screenshot is taken right before the user performs the `i`th action, resulting in the `i+1`th screenshot."
        ),
    ]
    # image_content(pil_to_bytes(first_frame, format="PNG")),
    for i, (action, frame) in enumerate(
        zip(actions, rest_frames, strict=True),
    ):
        prompt.append(text_content(f"\n\nScreenshot #{i + 1}:"))
        # text = f"Screenshot {i}:\n"
        text = f"Action #{i + 1}: {action.dump_to_text()}."
        if hasattr(action.action, "point") and action.action.action_type != "scroll":
            point = action.action.point
            text += f" (The point that was clicked ({point.x}, {point.y}) is highlighted in red.)"
            frame = draw_red_circle(frame, point)
        prompt.append(image_content(pil_to_bytes(frame, format="PNG")))
        prompt.append(text_content(text))
    prompt.append(
        text_content(
            "\n\nThe last screenshot is the final state of the screen after all actions have been performed:",
        )
    )
    prompt.append(
        {
            **image_content(pil_to_bytes(last_frame, format="PNG")),
            "cache_control": {"type": "ephemeral"},
        },
    )
    return prompt


def process_actions_range(
    source_dir: str,
    recording_metadata: RecordingMetadata,
    target_resolution: VideoResolution,
    actions: list[Action],
) -> TrainSample | None:
    """
    Process a range of actions to create multiple TrainSamples.

    Args:
        source_dir: Source directory path (e.g., gs://bucket/folder)
        recording_metadata: RecordingMetadata object
        actions: List of actions to process

    Returns:
        List of TrainSamples created from the actions
    """
    first_t, rest_timestamps = get_timesteps_range(actions)
    timestamps = [first_t, *rest_timestamps]
    logger.debug(f"{timestamps=}")
    frames_with_metadata = get_frames_at_timestamps(
        source_dir,
        recording_metadata,
        timestamps,
        target_resolution=target_resolution,
    )
    assert len(frames_with_metadata) == len(timestamps), (
        f"Expected {len(timestamps) + 1} frames, got {len(frames_with_metadata)}"
    )
    logger.debug("Got frames")
    frames = [frame[0] for frame in frames_with_metadata]
    frame_metadatas = [frame[1] for frame in frames_with_metadata]

    base_prompt = build_base_prompt(
        last_frame=frames[-1],
        rest_frames=frames[:-1],
        actions=actions,
    )
    frame_buffers = [pil_to_bytes(frame, format="PNG") for frame in frames]

    try:
        video_instruction = get_video_instruction(
            base_prompt,
        )
        logger.debug("Got video instruction")
        if video_instruction is None:
            logger.warning("Failed to generate video instruction ")
            return None
        if not video_instruction[0].is_productive:
            logger.info(
                f"Recording marked as not productive {video_instruction[0].instruction=}, {source_dir=} skipping."
            )
            return None
        instruction_response, instruction_cost = video_instruction
    except Exception as e:
        logger.error(f"Error generating video instruction {e}", exc_info=True)
        return None
    try:
        thinking_texts, thinking_costs = get_thinking_texts(
            base_prompt,
            instruction_response.instruction,
            actions,
        )
        logger.debug("Got thinking texts")
    except Exception as e:
        logger.error(f"Error generating thinking texts for  {e}", exc_info=True)
        return None
    b64_images = [base64.b64encode(bytes).decode("utf-8") for bytes in frame_buffers]
    save_actions = [
        SaveAction(
            step=i,
            image=b64_images[i],
            action=action.dump_to_text(),
            text=f"Thought: {thinking_texts[i].user_monologue}\nAction: {action.dump_to_text()}",
            thinking=thinking_texts[i].user_monologue,
            frame_metadata=frame_metadatas[i],
            reward=1.0,
            # reward=0.0 if thinking_texts[i].is_loading else 1.0,
        )
        for i, action in enumerate(actions)
    ]
    save_actions.append(
        SaveAction(
            step=len(actions),
            image=b64_images[-1],
            action="finished()",
            text="Action: finished()",
            thinking="",
            frame_metadata=frame_metadatas[-1],  # Use the last frame metadata
            reward=0.0,
        )
    )

    return TrainSample(
        actions=save_actions,
        instruction=instruction_response.instruction,
        metadata={
            "source_dir": source_dir,
            "start_time": float(frame_metadatas[0].timestamp),
            "end_time": float(frame_metadatas[-1].timestamp),
            "instruction_response": instruction_response.model_dump(),
            # "filter_response": filter_decision.model_dump(),
            "thinking_responses": [t.model_dump() for t in thinking_texts],
            "original_resolution": {
                "width": recording_metadata.screen_info.video_width,
                "height": recording_metadata.screen_info.video_height,
            },
            "output_resolution": {
                "width": recording_metadata.screen_info.logical_pixel_width,
                "height": recording_metadata.screen_info.logical_pixel_height,
            },
            "logical_pixel_ratio": recording_metadata.screen_info.logical_pixel_ratio,
            "costs": {
                "instruction_generation": instruction_cost,
                "thinking_texts_generation": thinking_costs,
                # "filter_cost": filter_cost,
            },
        },
    )


def get_actions_df(source_dir: str) -> pd.DataFrame:
    raw_actions = get_actions(source_dir)
    parsed_actions = parse_actions(raw_actions)
    combined_actions = combine_scroll_actions(parsed_actions)
    return pd.DataFrame(
        [
            {
                "action": a,
                "start_timestamp": a.timestamp,
                "end_timestamp": a.end_timestamp,
                "action_type": a.action.action_type,
            }
            for a in combined_actions
        ]
    )


def gen_id(length: int = 8) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def get_video_index(recording_metadata: RecordingMetadata, timestamp: float) -> int:
    """
    Get the video index from the recording metadata.

    Args:
        recording_metadata: RecordingMetadata object

    Returns:
        Video index as a string
    """
    rel_timestamp = timestamp - recording_metadata.timestamp

    if rel_timestamp < 0:
        logger.warning(
            f"Timestamp {timestamp} is before recording start time {recording_metadata.timestamp}. Returning index 0."
        )
        return 0
    video_segment_length = recording_metadata.video_segment_buffer_length
    video_index = int(rel_timestamp // video_segment_length)
    return video_index


class FrameMetadata(BaseModel):
    timestamp: float
    original_timestamp: float
    video_path: str


def get_target_resolution(
    recording_metadata: RecordingMetadata,
    max_pixels: int = resolution_1080p.pixels,
) -> VideoResolution:
    """
    Get the target resolution based on the recording metadata and optional target resolution.
    """
    resize_h, resize_w = smart_resize(
        recording_metadata.screen_info.video_height,
        recording_metadata.screen_info.video_width,
        min_pixels=0,
        max_pixels=max_pixels,
    )
    output_resolution = VideoResolution(width=resize_w, height=resize_h)
    return output_resolution


def get_frames_at_timestamps_from_video(
    video_path: str,
    video_timestamps: list[float],
    recording_metadata: RecordingMetadata,
    target_resolution: VideoResolution | None = None,
) -> list[tuple[PIL.Image.Image, FrameMetadata]]:
    """
    Extract frames at specific timestamps from a video file.
    """
    input_resolution = VideoResolution(
        width=recording_metadata.screen_info.video_width,
        height=recording_metadata.screen_info.video_height,
    )
    buffer_src, buffer_sink, filter_graph = None, None, None
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        download_video_with_ffmpeg_copy(
            source_path=video_path,
            dest_path=f.name,
            ffmpeg_path="/nix/store/q7j5awbg80d38p9my5b5zgn0xadgvbmb-ffmpeg-7.1.1-bin/bin/ffmpeg",
        )

        with av.open(f.name) as video_container:
            # Use consistent filter based on RecordingMetadata
            container_metadata = VideoMetadataFetcher.get_video_metadata(
                video_container.streams.video[0]
            )

            assert container_metadata.time_base == recording_metadata.time_base, (
                f"Container time base {container_metadata.time_base} does not match recording metadata time base {recording_metadata.time_base}"
            )
            assert container_metadata.resolution == input_resolution, (
                f"Container resolution {container_metadata.resolution} {video_path=} does not match input resolution {input_resolution}"
            )

            for timestamp in video_timestamps:
                assert (
                    container_start := float(
                        container_metadata.start_pts * container_metadata.time_base
                    )
                ) <= timestamp, (
                    f"Container start_pts {container_start} {video_path=} does not match recording metadata timestamp {timestamp}"
                )
                assert (
                    container_end := float(
                        container_metadata.end_pts * container_metadata.time_base
                    )
                ) >= timestamp, (
                    f"Container end_pts {container_end} does not match recording metadata timestamp {timestamp}"
                )
            if target_resolution is not None:
                buffer_src, buffer_sink, filter_graph = get_resize_filter(
                    video_container.streams.video[0].format,
                    input_resolution,
                    target_resolution,
                    recording_metadata.time_base,
                )

            # Extract frames at the relative timestamps
            frame_tuples = extract_frames_by_pts_from_container(
                video_container,
                video_timestamps,
                buffer_src,
                buffer_sink,
            )

            # Store frames in the correct positions in results
            frames_with_metadata = [
                (
                    frame,
                    FrameMetadata(
                        timestamp=frame_timestamp,
                        video_path=video_path,
                        original_timestamp=original_timestamp,
                    ),
                )
                for (frame_timestamp, frame), original_timestamp in zip(
                    frame_tuples, video_timestamps, strict=True
                )
            ]
            return frames_with_metadata


def get_frames_at_timestamps(
    source_dir: str,
    recording_metadata: RecordingMetadata,
    timestamps: list[float],
    target_resolution: VideoResolution,
) -> list[tuple[PIL.Image.Image, FrameMetadata]]:
    """
    Extract frames at specific timestamps from videos in a source directory.

    Args:
        source_dir: Source directory path (e.g., gs://bucket/folder)
        recording_metadata: RecordingMetadata object containing video segment info
        timestamps: List of sorted timestamps to extract frames from

    Returns:
        List of tuples (image, metadata) corresponding to the timestamps
    """
    if not timestamps:
        return []

    # Group timestamps by video index
    video_groups: dict[int, list[tuple[int, float]]] = {}
    for i, timestamp in enumerate(timestamps):
        video_index = get_video_index(recording_metadata, timestamp)
        if video_index not in video_groups:
            video_groups[video_index] = []
        video_groups[video_index].append((i, timestamp))

    # Initialize results list with correct size
    results: list[tuple[PIL.Image.Image, FrameMetadata] | None] = [None] * len(
        timestamps
    )

    # Process each video index
    for video_index, timestamp_pairs in video_groups.items():
        video_path = f"{source_dir}/screen_capture_{video_index:06d}.mp4"

        # Extract just the timestamps for this video
        video_timestamps = [ts for _, ts in timestamp_pairs]
        # assert video_timestamps are sorted
        assert video_timestamps == sorted(video_timestamps), (
            f"Timestamps for video {video_index} are not sorted: {video_timestamps} {video_path=}"
        )
        video_metadata = get_video_metadata(video_path)

        container_start = float(video_metadata.start_pts * video_metadata.time_base)
        container_end = float(video_metadata.end_pts * video_metadata.time_base)
        before_video_start_timestamps = [
            ts for ts in video_timestamps if ts < container_start
        ]
        after_video_end_timestamps = [
            ts for ts in video_timestamps if ts > container_end
        ]
        in_video_timestamps = [
            ts for ts in video_timestamps if container_start <= ts <= container_end
        ]
        before_video_frames = []
        after_video_frames = []

        if before_video_start_timestamps:
            logger.warning(
                f"Some timestamps {before_video_start_timestamps} are before the start of the video {video_path}. "
                f"They will be fetched from video {video_index - 1}"
            )
            before_video_path = f"{source_dir}/screen_capture_{video_index - 1:06d}.mp4"
            before_video_frames = get_frames_at_timestamps_from_video(
                before_video_path,
                before_video_start_timestamps,
                recording_metadata,
                target_resolution=target_resolution,
            )

        if after_video_end_timestamps:
            logger.warning(
                f"Some timestamps {after_video_end_timestamps} are after the end of the video {video_path}. "
                f"They will be fetched from video {video_index + 1}"
            )
            after_video_path = f"{source_dir}/screen_capture_{video_index + 1:06d}.mp4"
            after_video_frames = get_frames_at_timestamps_from_video(
                after_video_path,
                after_video_end_timestamps,
                recording_metadata,
                target_resolution=target_resolution,
            )

        # Get frames for the in-video timestamps
        in_video_frames = get_frames_at_timestamps_from_video(
            video_path,
            in_video_timestamps,
            recording_metadata,
            target_resolution=target_resolution,
        )
        video_frames = before_video_frames + in_video_frames + after_video_frames
        # Store frames in the correct positions in results
        for (frame, metadata), (
            original_index,
            _,
        ) in zip(video_frames, timestamp_pairs, strict=True):
            results[original_index] = (frame, metadata)
    results = [result for result in results if result is not None]
    assert len(results) == len(timestamps), "Results length mismatch with timestamps"
    # Filter out None values and return only successful frames
    return results  # type: ignore  # noqa: PGH003


fs = gcsfs.GCSFileSystem()


def get_video_metadata(
    video_path: str,
):
    with fs.open(video_path, "rb") as f:
        video_container = av.open(f)
        video_stream = video_container.streams.video[0]
        return VideoMetadataFetcher.get_video_metadata(video_stream)


def load_metadata(source_dir: str) -> RecordingMetadata:
    """
    Load and validate metadata.json from a source directory.

    Args:
        source_dir: Source directory path (e.g., gs://bucket/folder)

    Returns:
        RecordingMetadata object with validated screen_info

    Raises:
        Exception: If metadata.json cannot be loaded or is invalid
    """
    source_dir = source_dir.rstrip("/")

    try:
        # Remove gs:// prefix and construct metadata path
        gs_path = source_dir.replace("gs://", "")
        metadata_path = f"{gs_path}/metadata.json"

        with fs.open(metadata_path, "r") as f:
            metadata_dict = json.load(f)

        # Convert timestamp to

        metadata_dict["timestamp"] = 0
        # Validate using Pydantic

    except Exception as e:
        raise Exception(
            f"Failed to load or validate metadata from {source_dir}: {e}"
        ) from e
    first_video_path = f"{source_dir}/screen_capture_000000.mp4"
    try:
        # Validate first video
        # We need to do this because action recorder is not saving video timestamp in utc prior to 0.1.2 mac build
        video_metadata = get_video_metadata(first_video_path)
        first_video_start_time = float(
            video_metadata.start_pts * video_metadata.time_base
        )
        metadata_dict["timestamp"] = first_video_start_time
        metadata_dict["time_base"] = video_metadata.time_base
        if metadata_dict.get("platform", None) is None:
            logger.warning(
                f"Platform not specified in metadata {metadata_path=}. Setting to unknown."
            )
            metadata_dict["platform"] = Platform.unknown.value

        metadata = RecordingMetadata(**metadata_dict)

        if metadata.platform.is_windows:
            logger.warning(
                "Windows platform detected. Please ensure the metadata is correct and matches the video resolution."
            )
            metadata.screen_info.video_width = video_metadata.resolution.width
            metadata.screen_info.video_height = video_metadata.resolution.height
            metadata.screen_info.logical_pixel_width = video_metadata.resolution.width
            metadata.screen_info.logical_pixel_height = video_metadata.resolution.height
            metadata.screen_info.logical_pixel_ratio = 1.0

        # Check if video resolution matches screen info
        assert video_metadata.resolution.width == metadata.screen_info.video_width, (
            f"Video width {video_metadata.resolution.width} does not match screen info in metadata {metadata.screen_info.video_width}"
        )

        assert video_metadata.resolution.height == metadata.screen_info.video_height, (
            f"Video height {video_metadata.resolution.height} does not match screen info in metadata {metadata.screen_info.video_height}"
        )

    except Exception as e:
        raise Exception(
            f"Failed to validate first video {first_video_path} in {source_dir}: {e}"
        ) from e
    return metadata


def chunkify(lst, n):
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def save_and_get_metadata(
    source_dir: str,
    output_dir: str,
    train_sample: TrainSample,
) -> dict:
    """
    Process a single video for multiprocessing.

    Args:
        args: Tuple containing (video_path, source_dir, actions_df, select_policy, output_dir, metadata)

    Returns:
        List of dictionaries with train sample metadata
    """

    train_sample_id = (
        train_sample.metadata.get("train_sample_id", gen_id(12))
        if train_sample.metadata
        else gen_id(12)
    )

    # Save the train sample to file
    with smart_open(
        f"{output_dir}/metadata/{train_sample_id}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(train_sample.model_dump()["actions"], f, ensure_ascii=False, indent=0)
    with smart_open(
        f"{output_dir}/train_samples/{train_sample_id}.metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            train_sample.model_dump(exclude={"actions"}),
            f,
            ensure_ascii=False,
            indent=0,
        )

    # Add metadata for the main process to collect
    return {
        "attempt_id": train_sample_id,
        "eval_task_id": train_sample_id,
        "actions": [s.action for s in train_sample.actions],
        "thinking": [s.thinking for s in train_sample.actions],
        "instruction": train_sample.instruction,
        "trajectory_length": len(train_sample.actions),
        "source_dir": source_dir,
        "image_turns_start": 0,
        "image_turns_end": len(train_sample.actions) - 1,
        "text_turns_start": 0,
        "text_turns_end": len(train_sample.actions) - 1,
        "unmask_last_only": False,
    }


def thread_worker(
    source_dir: str,
    output_dir: str,
    recording_metadata: RecordingMetadata,
    target_resolution: VideoResolution,
    actions: list[Action],
):
    """Thread worker function that processes a single video."""
    try:
        result = process_actions_range(
            source_dir, recording_metadata, target_resolution, actions
        )
        if result is None:
            return (None, None)
        metadata = save_and_get_metadata(source_dir, output_dir, result)
        logger.debug(f"Processed {len(actions)} actions from {source_dir}")
        return (metadata, None)  # video_path, results, error
    except Exception:
        return (None, traceback.format_exc())


def process_worker[*T](
    thread_worker: Callable[[*T], tuple[dict | None, str | None]],
    threads_per_proc: int,
    update_queue: Queue[tuple[dict | None, str | None]],
    video_args_chunk: list[tuple[*T]],
):
    """
    Runs in each separate process. Spins up a thread pool to process its chunk of videos.
    Reports completion (with error or not) back via update_queue.
    """
    with ThreadPoolExecutor(max_workers=threads_per_proc) as executor:
        futures = {
            executor.submit(thread_worker, *video_args): video_args
            for video_args in video_args_chunk
        }
        for fut in as_completed(futures):
            logging.debug(f"Future completed: {fut}")
            save_action, err = fut.result()
            # send results so main can aggregate them
            update_queue.put((save_action, err))


def progress_listener[T](
    total: int,
    update_queue: Queue[tuple[T | None, str | None]],
    results_collector: ListProxy[T],
    output_dir: str,
    save_frequency: int = 100,
):
    """
    Runs in a thread in main process to consume updates and show progress bar.
    Also collects results from all processes and saves them incrementally.

    Args:
        total: Total number of items to process
        update_queue: Queue for receiving updates from worker processes
        results_collector: Shared list for collecting results
        output_dir: Directory to save results to
        save_frequency: How often to save results (every N successful samples)
    """
    pbar = tqdm.tqdm(total=total, desc="Processing videos", unit="video")
    completed = 0

    # Track counts for different outcomes
    success_count = 0
    none_count = 0
    error_count = 0

    # Track incremental saving
    local_results = []
    file_counter = 0
    total_saved = 0

    while completed < total:
        try:
            save_action, err = update_queue.get(timeout=1)
        except Empty:
            if completed >= total:
                break
            continue
        except Exception:
            import traceback

            tqdm.tqdm.write(
                f"[!] Error getting update from queue: {traceback.format_exc()}"
            )
            continue

        # Categorize the result and update counts
        if err:
            error_count += 1
            tqdm.tqdm.write(f"[!] Error processing action range:\n{err}")
        elif save_action is not None:
            success_count += 1
            results_collector.append(save_action)
            local_results.append(save_action)

            # Save every save_frequency successful samples
            if len(local_results) >= save_frequency:
                _save_results_batch(local_results, output_dir, file_counter)
                total_saved += len(local_results)
                tqdm.tqdm.write(
                    f"Saved batch {file_counter} with {len(local_results)} samples (total saved: {total_saved})"
                )
                local_results = []
                file_counter += 1
        else:
            none_count += 1

        completed += 1
        pbar.update(1)

        # Update progress bar description with counts
        pbar.set_description(
            f"Processing videos (✓{success_count} ∅{none_count} ✗{error_count})"
        )

        # Periodically write detailed counts
        if completed % 10 == 0 or completed == total:
            tqdm.tqdm.write(
                f"Progress: {completed}/{total} - "
                f"Successes: {success_count}, None returns: {none_count}, Errors: {error_count}"
            )

    # Save any remaining results
    if local_results:
        _save_results_batch(local_results, output_dir, file_counter)
        total_saved += len(local_results)
        tqdm.tqdm.write(
            f"Saved final batch {file_counter} with {len(local_results)} samples"
        )

    # Final summary
    tqdm.tqdm.write(
        f"Final results: {success_count} successes, {none_count} None returns, "
        f"{error_count} errors out of {total} total. Total saved: {total_saved} samples in {file_counter} files."
    )
    pbar.close()


def _save_results_batch(results: list, output_dir: str, file_counter: int) -> None:
    """
    Save a batch of results to a JSONL file.

    Args:
        results: List of results to save
        output_dir: Output directory
        file_counter: Counter for the filename
    """
    try:
        results_df = pd.DataFrame(results)
        output_file = f"{output_dir}/samples_{file_counter}.jsonl"
        results_df.to_json(
            output_file,
            orient="records",
            lines=True,
        )
    except Exception as e:
        tqdm.tqdm.write(f"[!] Error saving batch {file_counter}: {e}")


def filter_actions_time_bounds(
    actions: list[dict],
    time_bounds: tuple[float | None, float | None],
) -> list[dict]:
    """
    Filter actions based on time bounds.

    Args:
        actions: List of Action objects to filter
        time_bounds: Tuple of (start_time, end_time) to filter actions

    Returns:
        List of Action objects that fall within the specified time bounds
    """
    start_time, end_time = time_bounds
    if start_time is None and end_time is None:
        return actions
    filtered_actions = []
    for action in actions:
        if (start_time is None or action["timestamp"] >= start_time) and (
            end_time is None or action["timestamp"] <= end_time
        ):
            filtered_actions.append(action)
    if start_time:
        assert filtered_actions[0]["timestamp"] >= start_time, (
            f"First action timestamp {filtered_actions[0]['timestamp']} is before start time {start_time}"
        )
    if end_time:
        assert filtered_actions[-1]["timestamp"] <= end_time, (
            f"Last action end timestamp {filtered_actions[-1]['timestamp']} is after end time {end_time}"
        )
    logger.info(
        f"Filtered actions from {len(actions)} to {len(filtered_actions)} based on time bounds {time_bounds}"
    )
    return filtered_actions


def process_videos(
    source_folders: list[tuple[str, tuple[float | None, float | None]]],
    output_dir: str,
    num_processes: int | None = None,
    threads_per_process: int = 12,
    max_video_files: int | None = None,
):
    """
    Process videos in parallel using multiprocessing + ThreadPoolExecutor from multiple source folders.

    Args:
        source_folders: List of directories containing source videos and actions
        output_dir: Directory to save processed results
        num_processes: Number of processes to use. If None, uses CPU count
        threads_per_process: Number of threads per process (default: 12)
        max_video_files: Maximum number of video files to process
    """
    # Discover all video files in the source folders

    source_folders = [
        (source.rstrip("/"), time_bounds)
        for source, time_bounds in source_folders
        if source
    ]

    raw_action_sets = {
        k: filter_actions_time_bounds(get_actions(k[0]), k[1]) for k in source_folders
    }

    video_metadatas = {k: load_metadata(k[0]) for k in source_folders}
    target_resolutions = {
        k: get_target_resolution(v) for k, v in video_metadatas.items()
    }
    action_sets = {
        k: transform_action_coords_list(
            combine_actions(parse_actions(v)),
            target_resolutions[k],
            video_metadatas[k],
        )
        for k, v in raw_action_sets.items()
    }
    action_segments = [
        (source_dir, segment_actions_by_time_gaps(actions))
        for source_dir, actions in action_sets.items()
    ]

    unravelled_segments = [
        (source_dir, action_chunk)
        for source_dir, segments in action_segments
        for actions in segments
        for action_chunk in reverse_chunk_actions_to_length(
            actions,
        )
    ]
    action_segment_lens_df = pd.DataFrame(
        [
            {
                "source_dir": source_dir,
                "num_actions": len(actions),
            }
            for source_dir, actions in unravelled_segments
        ]
    )
    assert len(action_segment_lens_df) > 0, (
        "No action segments found. Check if the source folders contain valid actions."
    )
    print("Action segments summary:")
    print(action_segment_lens_df.describe())

    if not unravelled_segments:
        print("No video files found in the specified folders")
        return
    if max_video_files is not None:
        unravelled_segments = unravelled_segments[:max_video_files]
        print(f"Limiting to first {max_video_files} video files")

    # Build mappings for efficiency

    # Prepare arguments for each video
    process_args: list[
        tuple[str, str, RecordingMetadata, VideoResolution, list[Action]]
    ] = [
        (
            source_dir[0],
            output_dir,
            video_metadatas[source_dir],
            target_resolutions[source_dir],
            actions,
        )
        for source_dir, actions in unravelled_segments
    ]

    # Use all CPU cores if num_processes not specified
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(
        f"Processing {len(process_args)} videos using {num_processes} processes x {threads_per_process} threads..."
    )

    # Use multiprocessing + threading pattern
    manager = Manager()
    update_queue: Queue[tuple[dict | None, str | None]] = manager.Queue()
    results_collector: ListProxy[dict] = (
        manager.list()
    )  # shared list to collect results

    # Start progress listener thread
    listener = threading.Thread(
        target=progress_listener,
        args=(len(process_args), update_queue, results_collector, output_dir),
        daemon=True,
    )
    listener.start()

    # Split video args among processes
    chunks = chunkify(process_args, num_processes)
    worker = process_worker(
        thread_worker,
        threads_per_process,
        update_queue,
    )
    processes = []
    for chunk in chunks:
        if not chunk:
            continue
        p = Process(
            target=worker,
            args=(chunk, threads_per_process, update_queue),
        )
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Interrupted, terminating child processes...")
        for p in processes:
            p.terminate()

    # Wait until listener sees all updates
    listener.join()

    # Results are now saved incrementally by progress_listener
    print(
        f"Processing completed. Results saved incrementally in batches to {output_dir}/samples_*.jsonl files"
    )


def main() -> None:
    dataset_name = "kunal_data"
    process_videos(
        [
            # # Jeffrey
            # (
            #     "gs://induction-labs-data-ext/action_capture/jeffrey/2025-08-10_133207_0V8HU",
            #     (None, None),
            # ),
            # # Jonathan
            # (
            #     "gs://induction-labs-data-ext/action_capture/jonathan/2025-07-17_093647_KZ3CG",
            #     (None, None),
            # ),
            # # Jarry
            # (
            #     "gs://induction-labs-data-ext/action_capture/Jarry/2025-07-07_002920_0SPCN",
            #     (None, None),
            # ),
            # (
            #     "gs://induction-labs-data-ext/action_capture/Jarry/2025-08-10_121140_Q4KI9",
            #     (None, None),
            # ),
            # (
            #     "gs://induction-labs-data-ext/action_capture/Jarry/2025-08-11_185116_OXFUY",
            #     (None, None),
            # ),
            # # Aryan
            # (
            #     # This one has second monitor stuffs
            #     "gs://induction-labs-data-ext/action_capture/aryan_91532/2025-07-07_170814_A2QD2",
            #     (None, None),
            # ),
            # (
            #     "gs://induction-labs-data-ext/action_capture/aryan_91532/2025-07-07_143610_SBK20",
            #     (None, None),
            # ),
            # (
            #     "gs://induction-labs-data-ext/action_capture/aryan_91532/2025-07-08_160952_VX5RU",
            #     # Filters to video 414. TODO: write auto filter based on timestamps
            #     (None, 1752017846.856214),
            # ),
            # # Joyce
            # *(
            #     (
            #         data,
            #         (None, None),
            #     )
            #     for data in [
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-04_110139_B6VYF/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-06_112602_9ZEFH/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-09_160136_WVNHY/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-12_100706_2RKVJ/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-14_111643_P725G/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-14_162035_B9PM6/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-15_100419_MAESY/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-15_101847_913IO/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-15_102313_K967C/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-18_203324_YL5VM/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-27_192513_2FNA0/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-27_192852_E59D8/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-27_193551_TX1BD/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-29_193316_LSEM8/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-30_111301_8RVWD/",
            #         "gs://induction-labs-data-ext/action_capture/joyceliu/2025-07-31_200548_0Z5EH/",
            #     ]
            # ),
            # Kunal
            *(
                (
                    data,
                    (None, None),
                )
                for data in [
                    # "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-18_101735_MIELX",
                    "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-19_121221_3A4PN",
                    "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-18_133213_FN0VR",
                    "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-18_172629_2RBSA",
                    "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-19_125156_PQS8V",
                    "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-19_150406_A27L7",
                    "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-21_111019_IF3S0",
                    "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-21_172805_4KV69",
                    "gs://induction-labs-data-ext/action_capture/Kunal/2025-07-21_195634_CHHBR",
                ]
            ),
            *(
                (
                    data,
                    (None, None),
                )
                for data in [
                    "gs://induction-labs-data-ext/action_capture/Mahdi_lumio/2025-07-19_202437_BSGP6",
                    "gs://induction-labs-data-ext/action_capture/Mahdi_lumio/2025-07-19_202545_R9AN9",
                    "gs://induction-labs-data-ext/action_capture/Mahdi_lumio/2025-07-21_145524_Y5HX2",
                    "gs://induction-labs-data-ext/action_capture/Mahdi_lumio/2025-07-24_141207_3TWWR",
                    "gs://induction-labs-data-ext/action_capture/Mahdi_lumio/2025-08-06_155356_WIGKJ",
                ]
            ),
        ],
        f"gs://induction-labs/passive_data/{datetime.datetime.now(datetime.UTC):%Y-%m-%d}/{dataset_name}-{datetime.datetime.now(datetime.UTC):%H-%M-%S}",
        # max_video_files=20,
        # num_processes=1,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
