from __future__ import annotations
from pathlib import Path


from modeling.config import RunConfig
from modeling.modules.text_pretrain.default import TextPretrainLITConfig
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers.models.qwen2_5_omni import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from modeling.modules.text_pretrain.default import (
    TextPretrainLIT,
)
from modeling.utils.class_property import class_property


class Qwen25OLIT(
    TextPretrainLIT[Qwen2_5OmniThinkerForConditionalGeneration, "Qwen25OLITConfig"]
):
    @class_property
    def model_cls(cls) -> type[Qwen2_5OmniThinkerForConditionalGeneration]:
        return Qwen2_5OmniThinkerForConditionalGeneration

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
        # dp_mesh = device_mesh["data_parallel"]
        # fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        # for layer_id, transformer_block in enumerate(self.model.model.layers):
        #     # Apply activation checkpointing

        #     # For now this is broken with HF models https://github.com/huggingface/transformers/issues/34928
        #     from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        #         checkpoint_wrapper,
        #     )

        #     transformer_block = checkpoint_wrapper(transformer_block)

        #     # reshard_after_forward = int(layer_id) < len(self.model.model.layers) - 1
        #     # fully_shard(
        #     #     transformer_block,
        #     #     **fsdp_config,
        #     #     reshard_after_forward=reshard_after_forward,
        #     # )
        #     self.model.model.layers[layer_id] = transformer_block
        return self.model
        # return fully_shard(self.model, **fsdp_config)

    def init_model_meta(self, *args) -> Qwen2_5OmniThinkerForConditionalGeneration:
        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.module_config.model_name,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_impl,
        )


class Qwen25OLITConfig(TextPretrainLITConfig):
    """
    Configuration class for Qwen-2.5O Lightning Module.
    Inherits from TextPretrainLITConfig and sets the model name.
    """

    config_path: str = "modeling.modules.text_pretrain.qwen_25o.Qwen25OLITConfig"
    model_name: str = "Qwen/Qwen2.5-Omni-3B"
    tokenizer_name: str = "Qwen/Qwen2.5-Omni-3B"

    @property
    def get_tokenizer(self):
        processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
        assert isinstance(processor, Qwen2_5OmniProcessor)
        tokenizer = processor.tokenizer  # type: ignore[attr-defined]
        return tokenizer

    def create_module(
        self, run_config: RunConfig, tmp_dir: Path, global_state
    ) -> Qwen25OLIT:
        return Qwen25OLIT(self, run_config, tmp_dir, global_state)
