from __future__ import annotations

from modeling.config import RunConfig, InstanceConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
)
from modeling.modules.text_pretrain.default import (
    TextPretrainLIT,
    TextPretrainLITConfig,
    ConfigType,
)
from modeling.utils.class_property import class_property


class Qwen3LIT(TextPretrainLIT[Qwen3ForCausalLM, ConfigType]):
    @class_property
    def model_cls(cls) -> type[Qwen3ForCausalLM]:
        return Qwen3ForCausalLM


class Qwen3LITConfig(TextPretrainLITConfig):
    config_path: str = "modeling.modules.text_pretrain.qwen3.Qwen3LITConfig"
    model_name: str = "Qwen/Qwen3-0.6B"
    tokenizer_name: str = "Qwen/Qwen3-0.6B"

    def create_module(
        self,
        run_config: RunConfig,
        instance_config: InstanceConfig,
    ) -> Qwen3LIT:
        return Qwen3LIT(self, run_config, instance_config)
