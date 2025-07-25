from __future__ import annotations

from typing import TypeVar

import numpy as np
import torch
from synapse.utils.logging import configure_logging, logging
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from modeling.config import (
    InstanceConfig,
    RunConfig,
)
from modeling.config.distributed import MeshAxis
from modeling.data.video_action import ActionDataSample
from modeling.utils.class_property import class_property

from .base_action import BaseActiionLIT, BaseActionLITConfig
from .qwen_25o_actions import (
    Qwen2_5OmniActionCausalLMOutputWithPast,
)
from .qwen_25vl_actions import (
    Qwen2_5_VLActionConfig,
    Qwen2_5_VLForActionModel,
)

logger = configure_logging(__name__, level=logging.DEBUG)

MODEL_TYPE = Qwen2_5_VLForActionModel
l2_loss = nn.MSELoss(reduce=False)

T = TypeVar("T")


def to_numpy_clean(tensor: torch.Tensor, dtype=torch.float32) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array, ensuring it is on CPU and detached.
    """
    return tensor.to(dtype=dtype, device="cpu").detach().numpy()


class Qwen25VLActionLIT(BaseActiionLIT[MODEL_TYPE, "Qwen25VLActionLITConfig"]):
    """
    Qwen-2.5O Lightning Module for text pretraining.
    Inherits from TextPretrainLIT and uses the Qwen-2.5O model.
    """

    model: MODEL_TYPE

    @class_property
    def model_cls(cls) -> type[MODEL_TYPE]:
        return MODEL_TYPE

    def call_model(
        self, inputs: ActionDataSample
    ) -> Qwen2_5OmniActionCausalLMOutputWithPast:
        """Call the model with the given inputs.
        This method should be implemented by subclasses.
        """
        return self.model(
            action_tokens=inputs.action_tokens,
            **inputs.qwen_inputs.model_dump(),
        )

    def init_model_meta(
        self,
    ):
        module_config = self.module_config
        config = Qwen2_5_VLActionConfig.from_pretrained(
            module_config.model_name,
            freeze_network=module_config.freeze_network,
            freeze_vision=module_config.freeze_vision,
            freeze_action_head=module_config.freeze_action_head,
            freeze_action_embedding=module_config.freeze_action_embedding,
            use_fun_mask=module_config.use_fun_mask,
        )

        model = MODEL_TYPE(config)
        assert isinstance(model, MODEL_TYPE), (
            f"Expected model to be of type Qwen2_5OmniThinkerForActionModelling, "
            f"got {type(model)}"
        )
        return model

    def shard_model(
        self,
        *,
        mp_policy: MixedPrecisionPolicy,
        device_mesh: DeviceMesh,
    ):
        """
        Shard the model using Fully Sharded Data Parallel (FSDP).
        This method is called during the model configuration phase.
        """

        fsdp_config = {
            "mesh": device_mesh[MeshAxis.FSDP],
            "mp_policy": mp_policy,
        }
        fully_shard(self.model.model.visual, **fsdp_config)

        for layer_id, transformer_block in enumerate(
            self.model.model.language_model.layers
        ):
            # Activation checkpointing kinda broken
            # For now this is broken with HF models https://github.com/huggingface/transformers/issues/34928

            reshard_after_forward = (
                int(layer_id) < len(self.model.model.language_model.layers) - 1
            )
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            self.model.model.language_model.layers[layer_id] = transformer_block

        return fully_shard(
            self.model,
            **fsdp_config,
        )


class Qwen25VLActionLITConfig(BaseActionLITConfig):
    """
    Configuration class for Qwen-2.5O Lightning Module.
    Inherits from TextPretrainLITConfig and sets the model name.
    """

    config_path: str = (
        "modeling.modules.action_instruct.qwen_25vl.Qwen25VLActionLITConfig"
    )
    model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B"
    # tokenizer_name: str = "Qwen/Qwen2.5-Omni-3B"

    def create_module(
        self,
        run_config: RunConfig,
        instance_config: InstanceConfig,
    ) -> Qwen25VLActionLIT:
        return Qwen25VLActionLIT(self, run_config, instance_config)

    @classmethod
    def module_cls(cls) -> type[Qwen25VLActionLIT]:
        return Qwen25VLActionLIT
