from __future__ import annotations

from transformers.models.qwen2_5_omni import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)


def main():
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        torch_dtype="auto",
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    assert isinstance(processor, Qwen2_5OmniProcessor)

    conversations = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hi friend"},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversations,
        load_audio_from_video=True,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        video_fps=1,
        # kwargs to be passed to `Qwen2-5-OmniProcessor`
        padding=True,
        use_audio_in_video=True,
    ).to(model.device)

    text_ids = model.generate(**inputs, use_audio_in_video=True)
    text = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(text)


if __name__ == "__main__":
    main()
