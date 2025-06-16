from .audio_process import process_audio_info
from .video_process import (
    process_vision_info,
)
# https://github.com/QwenLM/Qwen2.5-Omni/blob/main/qwen-omni-utils/pyproject.toml


def process_mm_info(conversations, use_audio_in_video, return_video_kwargs=False):
    audios = process_audio_info(conversations, use_audio_in_video)
    vision = process_vision_info(conversations, return_video_kwargs=return_video_kwargs)
    return (audios,) + vision
