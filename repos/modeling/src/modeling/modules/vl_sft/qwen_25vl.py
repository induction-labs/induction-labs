from __future__ import annotations

from typing import Any, TypeVar

from synapse.utils.logging import configure_logging, logging
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
)

from modeling.config import (
    DatapackConfig,
    InstanceConfig,
    RunConfig,
)
from modeling.config.distributed import MeshAxis
from modeling.data.trajectory_train import VlDatapackConfig, VlDataSample
from modeling.modules.base_module import BaseLITModule, BaseModuleConfig
from modeling.utils.class_property import class_property

logger = configure_logging(__name__, level=logging.DEBUG)

T = TypeVar("T")

MODEL_TYPE = Qwen2_5_VLForConditionalGeneration


class Qwen25VLActionLIT(BaseLITModule[MODEL_TYPE, VlDataSample, "VlSftLITConfig"]):
    """
    Qwen-2.5O Lightning Module for text pretraining.
    Inherits from TextPretrainLIT and uses the Qwen-2.5O model.
    """

    model: MODEL_TYPE

    @class_property
    def model_cls(cls) -> type[MODEL_TYPE]:
        return MODEL_TYPE

    def call_model(self, inputs: VlDataSample) -> Qwen2_5_VLCausalLMOutputWithPast:
        """Call the model with the given inputs.
        This method should be implemented by subclasses.
        """
        return self.model(
            **inputs.model_dump(),
        )

    def run_training_step(self, inputs: VlDataSample):
        outputs = self.call_model(inputs)

        return outputs.loss, {}

    def run_validation_step(self, inputs: VlDataSample, global_step: int):
        # Forward pass through the model
        outputs = self.call_model(inputs)

        return outputs.loss, {}

    def init_model_meta(
        self,
    ):
        model = MODEL_TYPE.from_pretrained(
            self.module_config.model_name,
            trust_remote_code=True,
        )

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


class VlSftLITConfig(BaseModuleConfig):
    config_path: str = "modeling.modules.vl_sft.qwen_25vl.VlSftLITConfig"
    tokenizer_name: str = "Qwen/Qwen2.5-Omni-3B"

    def create_module(
        self,
        run_config: RunConfig,
        instance_config: InstanceConfig,
    ) -> Qwen25VLActionLIT:
        return Qwen25VLActionLIT(self, run_config, instance_config)

    @classmethod
    def module_cls(cls) -> type[Qwen25VLActionLIT]:
        return Qwen25VLActionLIT

    def validate_datapack_compatibility(
        self, datapack_config: DatapackConfig[Any]
    ) -> VlDatapackConfig:
        assert isinstance(datapack_config, VlDatapackConfig), (
            f"Expected {datapack_config=} to be of type VlDatapackConfig"
        )
        return datapack_config
