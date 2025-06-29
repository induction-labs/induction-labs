from __future__ import annotations

from typing import Any

from modeling.config import DatapackConfig, RunConfig
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.text_module import TextLIT, TextLITConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from huggingface_hub import snapshot_download
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
)


class Qwen3LIT(TextLIT):
    model: Qwen3ForCausalLM

    def __init__(self, config: Qwen3LITConfig, run_config: RunConfig):
        super().__init__(config, run_config)

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
        assert isinstance(model, Qwen3ForCausalLM)
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
        self.model.to_empty(device=torch.cuda.current_device())
        load_checkpoint_and_dispatch(
            self.model,
            checkpoint=self.ckpt_dir,  # hub ID or local folder
            device_map={"": torch.cuda.current_device()},
            dtype=self.model.dtype,
        )
        dp_mesh = self.device_mesh["data_parallel"]  # provided by ModelParallelStrategy

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )

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
        fully_shard(self.model, **fsdp_config)


class Qwen3LITConfig(TextLITConfig):
    config_path: str = "modeling.modules.text_pretrain.qwen3.Qwen3LITConfig"
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

    def create_module(self, run_config: RunConfig) -> Qwen3LIT:
        return Qwen3LIT(self, run_config)
