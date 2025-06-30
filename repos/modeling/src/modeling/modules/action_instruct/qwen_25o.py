from __future__ import annotations

from typing import Any

from modeling.config import DatapackConfig, RunConfig
from modeling.data.video_action import ActionDatapackConfig
from modeling.modules.action_module import ActionLIT, ActionLITConfig
from .qwen_25o_actions import Qwen2_5OmniThinkerForActionModelling

from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
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

    def configure_model(self) -> None:
        # if self.model.device.type != "meta":
        #     return  # already configured
        assert isinstance(self.device_mesh, DeviceMesh), (
            f"Expected device_mesh to be a DeviceMesh, got {type(self.device_mesh)}"
        )

        dp_mesh = self.device_mesh["data_parallel"]  # provided by ModelParallelStrategy
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )

        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

        for layer_id, transformer_block in enumerate(self.model.model.layers):
            # Apply activation checkpointing

            # For now this is broken with HF models https://github.com/huggingface/transformers/issues/34928
            #             from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            #     checkpoint_wrapper,
            # )
            # transformer_block = checkpoint_wrapper(transformer_block)

            reshard_after_forward = int(layer_id) < len(self.model.model.layers) - 1
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            self.model.model.layers[layer_id] = transformer_block
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
