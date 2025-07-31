from __future__ import annotations

from typing import TypeVar

from liger_kernel.transformers import (
    apply_liger_kernel_to_qwen2_vl,
)
from synapse.utils.logging import configure_logging, logging
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
)

from modeling.config import (
    InstanceConfig,
    RunConfig,
)
from modeling.config.distributed import MeshAxis
from modeling.data.trajectory_train import VlDataSample
from modeling.modules.vl_sft.base import BaseVlSft, VlSftActionLITConfig
from modeling.utils.class_property import class_property

logger = configure_logging(__name__, level=logging.DEBUG)

T = TypeVar("T")

MODEL_TYPE = Qwen2_5_VLForConditionalGeneration


class Qwen25VLActionLIT(BaseVlSft[MODEL_TYPE, "VlSftLITConfig"]):
    """
    Qwen-2.5O Lightning Module for text pretraining.
    Inherits from TextPretrainLIT and uses the Qwen-2.5O model.
    """

    model: MODEL_TYPE

    @class_property
    def model_cls(cls) -> type[MODEL_TYPE]:
        return MODEL_TYPE

    def patch_model(self) -> Qwen2_5_VLForConditionalGeneration:
        if self.module_config.use_liger_kernel:
            logger.debug("Applying Liger Kernel to Qwen2.5 VL model")
            apply_liger_kernel_to_qwen2_vl(model=self.model)
        return super().patch_model()

    def call_model(self, inputs: VlDataSample) -> Qwen2_5_VLCausalLMOutputWithPast:
        """Call the model with the given inputs.
        This method should be implemented by subclasses.
        """
        return self.model(
            **{
                **inputs.model_dump(),
                "pixel_values": inputs.pixel_values
                if inputs.pixel_values.shape[0] != 0
                else None,
                "image_grid_thw": inputs.image_grid_thw
                if inputs.image_grid_thw.shape[0] != 0
                else None,
            },
            use_cache=False,
        )

    def init_model_meta(
        self,
    ):
        module_config = self.module_config
        config = Qwen2_5_VLConfig.from_pretrained(
            module_config.model_name,
            freeze_vision=module_config.freeze_vision,
        )

        model = MODEL_TYPE(config)
        for param in model.model.visual.parameters():
            param.requires_grad = not self.module_config.freeze_vision

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


class VlSftLITConfig(VlSftActionLITConfig):
    config_path: str = "modeling.modules.vl_sft.qwen_25vl.VlSftLITConfig"
    tokenizer_name: str = "Qwen/Qwen2.5-Omni-3B"
    use_liger_kernel: bool = True

    freeze_vision: bool = False

    def create_module(
        self,
        run_config: RunConfig,
        instance_config: InstanceConfig,
    ) -> Qwen25VLActionLIT:
        return Qwen25VLActionLIT(self, run_config, instance_config)

    @classmethod
    def module_cls(cls) -> type[Qwen25VLActionLIT]:
        return Qwen25VLActionLIT
