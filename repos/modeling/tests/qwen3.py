from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import fully_shard
import os

model_name = "Qwen/Qwen3-0.6B"

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)


def main():
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    # torch.distributed.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype="auto",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",  # Use "flash_attention_2" for better performance
        # device_map="auto",  # Use "auto" for automatic device mapping
    )
    model = fully_shard(model)
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
    print(f"{model=}")

    print(content)
    print(f"{torch.cuda.max_memory_allocated()=}")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
