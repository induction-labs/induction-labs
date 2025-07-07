from __future__ import annotations

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    RunConfig,
    WandbConfig,
    LinearLRSchedule,
)
from modeling.data.text_train import TextPretrainDatapackConfig

from modeling.modules.text_pretrain.qwen3 import Qwen3LITConfig
from modeling.types import AttentionImplementation
from pathlib import Path
from modeling.types import Accelerator, DType

Qwen3PretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="qwen3_4B_text_pretrain"),
        output_dir=Path("./output/qwen3_text_pretrain"),
        checkpoint=None,
    ),
    module=Qwen3LITConfig(
        model_name="Qwen/Qwen3-0.6B",
        tokenizer_name="Qwen/Qwen3-0.6B",
    ),
    datapack=TextPretrainDatapackConfig(),
    run=RunConfig(
        lr=LinearLRSchedule.constant_lr(1e-5),
        sequence_length=1024,
        batch_size=1,
        steps_per_epoch=5000,
        distributed=DistributedConfig(
            devices_per_node=1,
        ),
        attn_impl=AttentionImplementation.SDPA,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
    ),
)
Qwen3PretrainTest = Qwen3PretrainExperimentConfig.testing_config(
    num_steps=5, enable_wandb=True
)

# mdl export modeling.experiments.text_pretrain.qwen3.Qwen3PretrainExperimentConfig
# mdl export modeling.experiments.text_pretrain.qwen3.Qwen3PretrainTest
