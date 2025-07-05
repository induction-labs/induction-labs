from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from modeling.config import DatapackConfig, RunConfig
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.text_module import TextLIT, TextLITConfig, MODEL_TYPE
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

logger = configure_logging(
    __name__,
    # level=logging.DEBUG
)


class TextPretrainLIT(TextLIT[MODEL_TYPE, dict, "TextPretrainLITConfig"]):
    def init_model_meta(self, *args) -> MODEL_TYPE:
        model_config = AutoConfig.from_pretrained(
            self.module_config.model_name, trust_remote_code=True
        )

        logger.debug(f"Initializing model {self.module_config.model_name} with config")

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
        dp_mesh = device_mesh["data_parallel"]  # provided by ModelParallelStrategy
        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        return fully_shard(self.model, **fsdp_config)

    def run_training_step(self, inputs: dict) -> torch.Tensor:
        """
        Run a training step with the provided inputs.
        The inputs should be a dictionary containing the necessary data for the model.
        """
        # Forward pass through the model
        outputs = self.model(
            **inputs,  # inputs should contain the necessary model inputs
        )
        assert isinstance(outputs.loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )
        return outputs.loss

    def run_validation_step(
        self, inputs: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Run a training step with the provided inputs.
        The inputs should be a dictionary containing the necessary data for the model.
        """
        # Forward pass through the model
        outputs = self.model(
            **inputs,  # inputs should contain the necessary model inputs
        )

        assert isinstance(outputs.loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )
        return outputs.loss, {}

    # def configure_model(self) -> None:
    #     # TODO(jl): make this work with cpu device (e.g. just skip it)
    #     if self.model.device.type != "meta":
    #         return  # already configured
    #     assert isinstance(self.device_mesh, DeviceMesh), (
    #         f"Expected device_mesh to be a DeviceMesh, got {type(self.device_mesh)}"
    #     )
    #     _dp_mesh = self.device_mesh[
    #         "data_parallel"
    #     ]  # provided by ModelParallelStrategy

    #     self.model.to_empty(device=torch.cuda.current_device())
    #     load_checkpoint_and_dispatch(
    #         self.model,
    #         checkpoint=self.ckpt_dir,  # hub ID or local folder
    #         device_map={"": torch.cuda.current_device()},
    #         dtype=self.model.dtype,
    #     )
    #     from accelerate import (
    #         FullyShardedDataParallelPlugin,
    #         Accelerator,
    #     )
    #     from accelerate.utils.fsdp_utils import (
    #         fsdp2_prepare_model,
    #     )

    #     fsdp_plugin = FullyShardedDataParallelPlugin(fsdp_version=2)
    #     accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    #     # TODO: This doesn't fully work yet because it doens't handle the device mesh correctly
    #     self.model = fsdp2_prepare_model(
    #         accelerator=accelerator,
    #         model=self.model,
    #     )


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

    def create_module(self, run_config: RunConfig, tmp_dir: Path) -> TextPretrainLIT:
        return TextPretrainLIT(self, run_config, tmp_dir)
