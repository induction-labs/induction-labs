from __future__ import annotations

from pathlib import Path
from modeling.config import RunConfig
from modeling.modules.base_module import GlobalState
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from modeling.modules.text_pretrain.default import (
    TextPretrainLIT,
    ConfigType,
    TextPretrainLITConfig,
)
from modeling.utils.class_property import class_property


class Llama3LIT(TextPretrainLIT[LlamaForCausalLM, ConfigType]):
    @class_property
    def model_cls(cls) -> type[LlamaForCausalLM]:
        return LlamaForCausalLM

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
        dp_mesh = device_mesh["data_parallel"]
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


class Llama3LITConfig(TextPretrainLITConfig):
    config_path: str = "modeling.modules.text_pretrain.llama3.Llama3LITConfig"
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    tokenizer_name: str = "meta-llama/Meta-Llama-3-8B"

    def create_module(
        self, run_config: RunConfig, tmp_dir: Path, global_state: GlobalState
    ) -> Llama3LIT:
        return Llama3LIT(self, run_config, tmp_dir, global_state)
