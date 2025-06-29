from modeling.utils.qwen_omni_utils.video_process import process_vision_info
from transformers.models.qwen2_5_omni import (
    Qwen2_5OmniProcessor,
    # Qwen2_5OmniThinkerForConditionalGeneration,
)

# thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-Omni-7B"
# )
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

conversations = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg",
            },
            {
                "type": "video",
                "video": [
                    "https://www.ilankelman.org/stopsigns/australia.jpg",
                    "https://www.ilankelman.org/stopsigns/australia.jpg",
                    "https://www.ilankelman.org/stopsigns/australia.jpg",
                    "https://www.ilankelman.org/stopsigns/australia.jpg",
                    "https://www.ilankelman.org/stopsigns/australia.jpg",
                    "https://www.ilankelman.org/stopsigns/australia.jpg",
                    "https://www.ilankelman.org/stopsigns/australia.jpg",
                    "https://www.ilankelman.org/stopsigns/australia.jpg",
                    "https://www.ilankelman.org/stopsigns/australia.jpg",
                ],
            },
            {
                "type": "video",
                "video": "/home/jeffrey_inductionlabs_com/documents/induction-labs/repos/synapse/test_data/screen_capture_000000.mp4",
                # "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4",
            },
        ],
    },
]

text = processor.apply_chat_template(
    conversations, add_generation_prompt=True, tokenize=False
)
print(f"{text=}")
# audios = [
#     librosa.load(
#         BytesIO(urlopen(conversations[1]["content"][1]["audio_url"]).read()),
#         sr=self.processor.feature_extractor.sampling_rate,
#     )
# ]
images, videos = process_vision_info(conversations)
print(videos[1].shape)
inputs = processor(
    text=text,
    # audios=[],
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
)

# Generate
inputs["use_audio_in_video"] = False
print(inputs)
print(inputs["pixel_values"].shape)
print(inputs["pixel_values_videos"].shape)

print(inputs["input_ids"].shape)
print(inputs["input_ids"][0][400:420])

# generation = thinker.generate(**inputs, max_new_tokens=2048)
# generate_ids = generation[:, inputs.input_ids.size(1) :]
