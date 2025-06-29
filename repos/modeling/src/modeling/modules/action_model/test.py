from __future__ import annotations

from typing import Any

from modeling.config import DatapackConfig
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.text_module import TextLIT, TextLITConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.modeling_utils import PreTrainedModel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch
from torch.distributed.device_mesh import DeviceMesh
from huggingface_hub import snapshot_download


class TestActionLIT(TextLIT):
    def __init__(self, config: TestActionLITConfig):
        super().__init__(config)

        self.model_config = AutoConfig.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self.ckpt_dir = snapshot_download(  # downloads all shards & index
            repo_id=self.config.model_name,
        )

        with init_empty_weights():  # ① on meta
            model = AutoModelForCausalLM.from_config(
                self.model_config, torch_dtype=self.dtype
            )
        assert isinstance(model, PreTrainedModel)
        assert model.device.type == "meta", (
            f"Expected model to be on meta device, got {model.device.type}"
        )
        self.model = model

    def configure_model(self) -> None:
        if self.model.device.type != "meta":
            return  # already configured
        assert isinstance(self.device_mesh, DeviceMesh), (
            f"Expected device_mesh to be a DeviceMesh, got {type(self.device_mesh)}"
        )
        _dp_mesh = self.device_mesh[
            "data_parallel"
        ]  # provided by ModelParallelStrategy

        self.model.to_empty(device=torch.cuda.current_device())
        load_checkpoint_and_dispatch(
            self.model,
            checkpoint=self.ckpt_dir,  # hub ID or local folder
            device_map={"": torch.cuda.current_device()},
            dtype=self.model.dtype,
        )
        from accelerate import (
            FullyShardedDataParallelPlugin,
            Accelerator,
        )
        from accelerate.utils.fsdp_utils import (
            fsdp2_prepare_model,
        )

        fsdp_plugin = FullyShardedDataParallelPlugin(fsdp_version=2)
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        # TODO: This doesn't fully work yet because it doens't handle the device mesh correctly
        self.model = fsdp2_prepare_model(
            accelerator=accelerator,
            model=self.model,
        )


class TestActionLITConfig(TextLITConfig):
    config_path: str = "modeling.modules.text_pretrain.default.TestActionLITConfig"
    model_name: str = "Qwen/Qwen3-0.6B"
    tokenizer_name: str = "Qwen/Qwen3-0.6B"
    lr: float = 1e-3

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

    def create_module(self) -> TestActionLIT:
        return TestActionLIT(
            self,
        )
