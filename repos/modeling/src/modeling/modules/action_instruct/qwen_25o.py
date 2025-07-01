from __future__ import annotations

from typing import Any

from modeling.config import DatapackConfig, RunConfig
from modeling.data.video_action import ActionDataSample, ActionDatapackConfig
from modeling.modules.action_module import ActionLIT, ActionLITConfig
from modeling.utils.elapsed_timer import elapsed_timer
from .qwen_25o_actions import Qwen2_5OmniThinkerForActionModelling

from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
import torch


class Qwen25OActionLIT(ActionLIT):
    """
    Qwen-2.5O Lightning Module for text pretraining.
    Inherits from TextPretrainLIT and uses the Qwen-2.5O model.
    """

    model: Qwen2_5OmniThinkerForActionModelling

    def __init__(
        self,
        config: Qwen25OActionLITConfig,
        run_config: RunConfig,
    ):
        super().__init__(config=config, run_config=run_config)

        model = Qwen2_5OmniThinkerForActionModelling.from_pretrained(
            config.model_name,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_impl,
        ).train()
        assert isinstance(model, Qwen2_5OmniThinkerForActionModelling), (
            f"Expected model to be of type Qwen2_5OmniThinkerForActionModelling, "
            f"got {type(model)}"
        )
        self.model = model
        self.model_config = self.model.config

    def training_step(self, inputs: ActionDataSample):
        # Forward pass through the model

        with elapsed_timer() as timer:
            # Note: You CANT call self.model.forward here because it fucking doesn't trigger the FSDP hooks so weights dont gather
            outputs = self.model(
                cursor_path=inputs.cursor_path,
                action_tokens=inputs.action_tokens,
                **inputs.qwen_inputs.model_dump(),
            )
            elapsed = timer()

        assert isinstance(outputs.loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )
        # Assert that the loss is a scalar tensor
        assert outputs.loss.ndim == 0, (
            f"Expected outputs.loss to be a scalar tensor, got {outputs.loss.ndim}"
        )

        # Assert loss is not NaN or Inf
        assert not torch.isnan(outputs.loss), "Loss is NaN"
        assert not torch.isinf(outputs.loss), "Loss is Inf"
        # TODO: Add more metrics and logging (steptime, tok/s, etc.)

        if self.run_config.accelerator == "cuda":
            torch.cuda.synchronize()

            allocated_memory = torch.cuda.memory_allocated(
                device=torch.cuda.current_device()
            )
            reserved_memory = torch.cuda.memory_reserved(
                device=torch.cuda.current_device()
            )
            memory_metrics = {
                "train/allocated_memory": allocated_memory / 1e9,  # in GB
                "train/reserved_memory": reserved_memory / 1e9,  # in GB
            }

        metrics = {
            "train/step_time": elapsed,
            "train/tokens_per_second": inputs.qwen_inputs.input_ids.numel() / elapsed,
            "train/loss": outputs.loss,
            **memory_metrics,
        }

        self.log_dict(
            metrics,
            logger=True,
            on_step=True,  # Log on every step
        )
        return outputs.loss

    def configure_model(self) -> None:
        # if self.model.device.type != "meta":
        #     return  # already configured
        assert isinstance(self.device_mesh, DeviceMesh), (
            f"Expected device_mesh to be a DeviceMesh, got {type(self.device_mesh)}"
        )

        dp_mesh = self.device_mesh["data_parallel"]  # provided by ModelParallelStrategy
        # mp_policy = MixedPrecisionPolicy(
        #     param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        # )

        # fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        fsdp_config = {
            "mesh": dp_mesh,
        }

        # for layer_id, transformer_block in enumerate(self.model.model.layers):
        #     # Apply activation checkpointing

        #     # For now this is broken with HF models https://github.com/huggingface/transformers/issues/34928
        #     #             from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        #     #     checkpoint_wrapper,
        #     # )
        #     # transformer_block = checkpoint_wrapper(transformer_block)

        #     reshard_after_forward = int(layer_id) < len(self.model.model.layers) - 1
        #     fully_shard(
        #         transformer_block,
        #         **fsdp_config,
        #         reshard_after_forward=reshard_after_forward,
        #     )
        #     self.model.model.layers[layer_id] = transformer_block
        fully_shard(self.model, **fsdp_config)


class Qwen25OActionLITConfig(ActionLITConfig):
    """
    Configuration class for Qwen-2.5O Lightning Module.
    Inherits from TextPretrainLITConfig and sets the model name.
    """

    config_path: str = (
        "modeling.modules.action_instruct.qwen_25o.Qwen25OActionLITConfig"
    )
    model_name: str = "Qwen/Qwen2.5-Omni-3B"
    tokenizer_name: str = "Qwen/Qwen2.5-Omni-3B"

    def validate_datapack_compatibility(
        self, datapack_config: DatapackConfig[Any]
    ) -> ActionDatapackConfig:
        assert isinstance(datapack_config, ActionDatapackConfig), (
            f"Expected {datapack_config=} to be of type ActionDatapackConfig"
        )
        return datapack_config

    def create_module(self, run_config: RunConfig) -> Qwen25OActionLIT:
        return Qwen25OActionLIT(self, run_config)
