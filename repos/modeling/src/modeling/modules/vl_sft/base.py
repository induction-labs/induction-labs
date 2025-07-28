from __future__ import annotations

from abc import abstractmethod
from typing import Any

from synapse.utils.logging import configure_logging, logging
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)

from modeling.config import (
    DatapackConfig,
)
from modeling.data.trajectory_train import VlDatapackConfig, VlDataSample
from modeling.modules.base_module import BaseLITModule, BaseModuleConfig

logger = configure_logging(__name__, level=logging.DEBUG)

l2_loss = nn.MSELoss(reduce=False)


class BaseVlSft[ModelType: PreTrainedModel, ConfigType: "VlSftActionLITConfig"](
    BaseLITModule[ModelType, VlDataSample, ConfigType]
):
    """
    Qwen-2.5O Lightning Module for text pretraining.
    Inherits from TextPretrainLIT and uses the Qwen-2.5O model.
    """

    @abstractmethod
    def call_model(self, inputs: VlDataSample) -> Qwen2_5_VLCausalLMOutputWithPast:
        """Call the model with the given inputs.
        This method should be implemented by subclasses.
        """

    def run_training_step(self, inputs: VlDataSample):
        outputs = self.call_model(inputs)
        assert outputs.loss is not None, (
            f"Expected model outputs to contain a loss, but got {outputs=}"
        )

        return outputs.loss, {}

    def run_validation_step(self, inputs: VlDataSample, global_step: int):
        outputs = self.call_model(inputs)
        assert outputs.loss is not None, (
            f"Expected model outputs to contain a loss, but got {outputs=}"
        )

        return outputs.loss, {}


class VlSftActionLITConfig(BaseModuleConfig):
    """
    Configuration class for Qwen-2.5O Lightning Module.
    Inherits from TextPretrainLITConfig and sets the model name.
    """

    config_path: str = "modeling.modules.action_instruct.base_action.VlSftLITConfig"

    def validate_datapack_compatibility(
        self, datapack_config: DatapackConfig[Any]
    ) -> VlDatapackConfig:
        assert isinstance(datapack_config, VlDatapackConfig), (
            f"Expected {datapack_config=} to be of type VlSftDatapackConfig"
        )
        return datapack_config
