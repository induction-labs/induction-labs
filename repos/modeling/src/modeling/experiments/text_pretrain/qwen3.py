from __future__ import annotations

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    RunConfig,
    WandbConfig,
)
from modeling.data.text_train import TextPretrainDatapackConfig

from modeling.modules.text_pretrain.qwen3 import Qwen3LITConfig
from modeling.types import AttentionImplementation

from modeling.types import Accelerator, DType

Qwen3PretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="qwen3_4B_text_pretrain"),
        output_dir="output/text_pretrain",
        checkpoint=None,
    ),
    module=Qwen3LITConfig(
        model_name="Qwen/Qwen3-8B",
        tokenizer_name="Qwen/Qwen3-8B",
    ),
    datapack=TextPretrainDatapackConfig(),
    run=RunConfig(
        lr=1e-5,
        sequence_length=1024,
        batch_size=4,
        steps_per_epoch=5000,
        distributed=DistributedConfig(
            devices_per_node=8,
        ),
        attn_impl=AttentionImplementation.SDPA,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
    ),
)
Qwen3PretrainTest = Qwen3PretrainExperimentConfig.testing_config(
    num_steps=10, enable_wandb=True
)

# mdl export modeling.experiments.text_pretrain.qwen3.Qwen3PretrainExperimentConfig
# mdl export modeling.experiments.text_pretrain.qwen3.Qwen3PretrainTest
