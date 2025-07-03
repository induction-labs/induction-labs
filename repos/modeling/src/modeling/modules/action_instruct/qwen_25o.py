from __future__ import annotations

from typing import Any

from modeling.config import DatapackConfig, RunConfig, ModuleConfig
from modeling.data.video_action import ActionDataSample, ActionDatapackConfig
from modeling.modules.base_module import BaseLITModule
from .qwen_25o_actions import Qwen2_5OmniThinkerForActionModelling
from torch.distributed.fsdp import MixedPrecisionPolicy

from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
import torch

MODEL_TYPE = Qwen2_5OmniThinkerForActionModelling


class Qwen25OActionLIT(
    BaseLITModule[MODEL_TYPE, ActionDataSample, "Qwen25OActionLITConfig"]
):
    """
    Qwen-2.5O Lightning Module for text pretraining.
    Inherits from TextPretrainLIT and uses the Qwen-2.5O model.
    """

    def __init__(
        self,
        module_config: Qwen25OActionLITConfig,
        run_config: RunConfig,
    ):
        super().__init__(module_config=module_config, run_config=run_config)

        model = MODEL_TYPE.from_pretrained(
            module_config.model_name,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_impl,
        ).train()
        assert isinstance(model, MODEL_TYPE), (
            f"Expected model to be of type Qwen2_5OmniThinkerForActionModelling, "
            f"got {type(model)}"
        )
        self.model = model

    def run_training_step(self, inputs: ActionDataSample):
        # Forward pass through the model
        outputs = self.model(
            cursor_path=inputs.cursor_path,
            action_tokens=inputs.action_tokens,
            **inputs.qwen_inputs.model_dump(),
        )
        assert isinstance(outputs.loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )
        # Assert that the loss is a scalar tensor

        return outputs.loss

    def shard_model(
        self,
        *,
        mp_policy: MixedPrecisionPolicy,
        device_mesh: DeviceMesh,
    ):
        # if self.model.device.type != "meta":
        #     return  # already configured

        dp_mesh = device_mesh["data_parallel"]  # provided by ModelParallelStrategy
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
        return fully_shard(self.model, **fsdp_config)


class Qwen25OActionLITConfig(ModuleConfig):
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
