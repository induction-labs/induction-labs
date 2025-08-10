"""
https://x.com/marksaroufim/status/1810695541963251924?lang=en
torchrun --standalone --nproc_per_node=2 test_fsdp_basic.py
"""

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


def main():
    torch.manual_seed(42)
    model_args = ModelArgs(
        n_layers=12,
        vocab_size=50304,
        n_heads=32,
        dim=2048,
        max_seq_len=2048,
        dropout_p=0.0,
    )
    model = Transformer(model_args)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    fsdp_cfg = {"mp_policy": mp_policy}
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            fully_shard(module, **fsdp_cfg)

    ignored_params: set[nn.Parameter] = set()
    # Doesn't work:
    # for param in model.output.parameters():
    #     ignored_params.add(param)
    fully_shard(model, ignored_params=ignored_params, **fsdp_cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inp = torch.randint(0, model_args.vocab_size, (8, 1024), device="cuda")
    loss = model(inp).sum()
    loss.backward()
    optim.step()
    print("Model training step completed successfully, loss:", loss.item())


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    rank = gpu_id
    main()
    dist.destroy_process_group()
