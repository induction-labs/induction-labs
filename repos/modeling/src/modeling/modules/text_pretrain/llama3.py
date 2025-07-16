from __future__ import annotations

from modeling.config import RunConfig, InstanceConfig
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


class Llama3LITConfig(TextPretrainLITConfig):
    config_path: str = "modeling.modules.text_pretrain.llama3.Llama3LITConfig"
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    tokenizer_name: str = "meta-llama/Meta-Llama-3-8B"

    def create_module(
        self,
        run_config: RunConfig,
        instance_config: InstanceConfig,
    ) -> Llama3LIT:
        return Llama3LIT(self, run_config, instance_config)
