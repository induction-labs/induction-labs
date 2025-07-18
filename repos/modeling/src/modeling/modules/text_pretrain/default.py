from __future__ import annotations

from typing import TypeVar
from typing import Any, cast

from modeling.config import (
    DatapackConfig,
    RunConfig,
    InstanceConfig,
)
from modeling.data.text_train import TextPretrainDatapackConfig, TextPretrainDataSample
from modeling.modules.text_module import TextLIT, TextLITConfig, MODEL_TYPE
from modeling.utils.class_property import class_property
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.modeling_utils import PreTrainedModel

import torch
from torch.distributed.device_mesh import DeviceMesh

from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from synapse.utils.logging import configure_logging
from accelerate import init_empty_weights
from modeling.config.distributed import MeshAxis
import logging

logger = configure_logging(__name__, level=logging.DEBUG)


ConfigType = TypeVar("ConfigType", bound="TextPretrainLITConfig")


class TextPretrainLIT(TextLIT[MODEL_TYPE, TextPretrainDataSample, ConfigType]):
    @class_property
    def model_cls(cls) -> type[PreTrainedModel]:
        """
        Return the class of the model that this module will use.
        This should be overridden in subclasses to return the specific model class.
        """
        return PreTrainedModel

    def init_model_meta(self, *args) -> MODEL_TYPE:
        model_config = AutoConfig.from_pretrained(
            self.module_config.model_name, trust_remote_code=True
        )

        logger.debug(f"Initializing model {self.module_config.model_name}")

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                model_config, trust_remote_code=True
            )

        logger.debug(
            f"Initialized model {self.module_config.model_name} with dtype {self.dtype}"
        )
        assert isinstance(model, PreTrainedModel)
        assert model.device.type == "meta", (
            f"Expected model to be on meta device, got {model.device.type}"
        )
        return cast(MODEL_TYPE, model)

    def shard_model(
        self,
        *,
        mp_policy: MixedPrecisionPolicy,
        device_mesh: DeviceMesh,
    ):
        """
        Shard the model using Fully Sharded Data Parallel (FSDP).
        This method is called during the model configuration phase.
        (HSDP) with ``(Replicate(), Shard(0))``
        ! TODO(jl): Currently broken - DP=1 just takes more memory with HSDP even though it should be a no-op
        """
        fsdp_config = {
            "mesh": device_mesh[MeshAxis.FSDP],
            "mp_policy": mp_policy,
        }
        assert isinstance(self.model.model, PreTrainedModel) and isinstance(
            self.model.model.layers, nn.ModuleList
        )
        for layer_id, transformer_block in enumerate(self.model.model.layers):
            reshard_after_forward = int(layer_id) < len(self.model.model.layers) - 1
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            self.model.model.layers[layer_id] = transformer_block
        return fully_shard(self.model, **fsdp_config)

    def run_training_step(
        self, inputs: TextPretrainDataSample
    ) -> tuple[torch.Tensor, dict]:
        """
        Run a training step with the provided inputs.
        The inputs should be a dictionary containing the necessary data for the model.
        """
        # Forward pass through the model
        outputs = self.model(
            **inputs.model_dump(),  # inputs should contain the necessary model inputs
        )
        assert isinstance(outputs.loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )
        return outputs.loss, {}

    def run_validation_step(
        self, inputs: TextPretrainDataSample, global_step: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Run a training step with the provided inputs.
        The inputs should be a dictionary containing the necessary data for the model.
        """
        # Forward pass through the model
        outputs = self.model(
            **inputs.model_dump(),  # inputs should contain the necessary model inputs
        )

        assert isinstance(outputs.loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )
        return outputs.loss, {}


class TextPretrainLITConfig(TextLITConfig):
    config_path: str = "modeling.modules.text_pretrain.default.TextPretrainLITConfig"
    model_name: str = "Qwen/Qwen3-0.6B"
    tokenizer_name: str = "Qwen/Qwen3-0.6B"

    @property
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer_name)

    def validate_datapack_compatibility(
        self, datapack_config: DatapackConfig[Any]
    ) -> TextPretrainDatapackConfig:
        assert isinstance(datapack_config, TextPretrainDatapackConfig), (
            f"Expected {datapack_config=} to be of type TextPretrainDatapackConfig"
        )
        return datapack_config

    def create_module(
        self,
        run_config: RunConfig,
        instance_config: InstanceConfig,
    ) -> TextPretrainLIT:
        return TextPretrainLIT(self, run_config, instance_config)
