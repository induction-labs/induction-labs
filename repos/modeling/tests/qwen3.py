from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B"

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)


def main():
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Use "flash_attention_2" for better performance
        device_map="auto",  # Use "auto" for automatic device mapping
    )
    # model = model.to("cuda:0")  # Move model to GPU
    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    print(content)
    print(f"{torch.cuda.max_memory_allocated()=}")


if __name__ == "__main__":
    main()
