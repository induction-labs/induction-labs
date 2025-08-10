import torch
import os

# from torch.testing._internal.distributed._tensor.common_dtensor import (
#     DTensorTestBase,
#     with_comms,
# )
from typing import cast
import functools


# torchrun tests/deterministic/test_fsdp.py
if __name__ == "__main__":
    seed = 42
    import random
    import numpy as np

    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)  # make hash-based operations reproducible

    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # CPU RNG
    torch.cuda.manual_seed(seed)  # current GPU RNG
    torch.cuda.manual_seed_all(seed)  # all-GPU RNGs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    use_bias = True

    # # In PyTorch ≥1.8 you can also do:
    torch.use_deterministic_algorithms(True)

    # mesh & fsdp2
    from torch.distributed.device_mesh import (
        init_device_mesh,
    )  # torch version: 2.4.1
    from torch.distributed._composable.fsdp import fully_shard, FSDPModule
    import torch.nn as nn

    mesh = init_device_mesh("cuda", (1, 1), mesh_dim_names=("replicate", "shard"))

    # llama model
    from transformers.models.qwen2_5_omni import (
        Qwen2_5OmniThinkerForConditionalGeneration,
        Qwen2_5OmniThinkerConfig,
    )

    class QwenWrapped(Qwen2_5OmniThinkerForConditionalGeneration):
        def __init__(self, config: Qwen2_5OmniThinkerConfig):
            config.output_hidden_states = True
            super().__init__(config)
            self.test_last_layer = nn.Sequential(
                nn.Linear(
                    config.text_config.hidden_size,
                    8,
                    bias=use_bias,
                ),
            )

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            last_hidden_state = super().forward(*args, **kwargs).hidden_states[-1]
            return self.test_last_layer(last_hidden_state)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = Qwen2_5OmniThinkerConfig.from_pretrained(
        os.path.join(dir_path, "qwen_config.json")
    )
    model = QwenWrapped(config)
    model = model.cuda()
    model = model.train()
    config = model.config

    # model: nn.Module =
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        fused=True,
    )

    # fsdp
    fully_shard_fn = functools.partial(
        fully_shard,
        mesh=mesh,
        # reshard_after_forward? # same NaN
        # mixed precision? # same NaN
    )
    model = fully_shard_fn(model)
    # model.set_reshard_after_backward()? # same NaN

    # microbatches
    for i in range(99):
        # if self.rank == 0:

        input = torch.randint(
            low=0, high=config.text_config.vocab_size, size=(4, 4), device="cuda"
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model(input).mean()
        loss.backward()
        optimizer.step()
        print(f"[DEBUG] microbatch {i} loss: {loss.item()}")

        # check NaN grad
        fsdp_params = []
        for module in cast(nn.Module, model).modules():
            if isinstance(module, FSDPModule):
                if fsdp_param_group := module._get_fsdp_state()._fsdp_param_group:
                    fsdp_params += fsdp_param_group.fsdp_params
        for fsdp_param in fsdp_params:
            sharded_param = fsdp_param.sharded_param
            if not sharded_param.requires_grad:
                continue
            if sharded_param.grad is None:
                continue
            local_grad = sharded_param.grad._local_tensor
            assert torch.isnan(local_grad).sum().item() == 0, (
                f"{local_grad=} {fsdp_param=}"
            )
            replicate_grad = sharded_param.grad.full_tensor()
            assert torch.isnan(replicate_grad).sum().item() == 0, f"{replicate_grad}"
