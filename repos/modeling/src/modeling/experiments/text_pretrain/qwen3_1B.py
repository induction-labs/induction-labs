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

name = "Qwen3-1B-TextPretrain"
Qwen3_1BPretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="qwen3_1B_text_pretrain"),
        output_dir=Path("./output/qwen3_1B_text_pretrain"),
        checkpoint=None,
    ),
    module=Qwen3LITConfig(
        model_name="Qwen/Qwen3-0.6B",
        tokenizer_name="Qwen/Qwen3-0.6B",
        # compile=CompileConfig(),  # Uncomment if you want to use compilation
    ),
    datapack=TextPretrainDatapackConfig(),
    run=RunConfig(
        lr=LinearLRSchedule.constant_lr(1e-5),
        sequence_length=4096,
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
Qwen3_1BPretrainTest = Qwen3_1BPretrainExperimentConfig.testing_config(
    num_steps=5, enable_wandb=False
)

# mdl export modeling.experiments.text_pretrain.qwen3.Qwen3_1BPretrainExperimentConfig
# mdl export modeling.experiments.text_pretrain.qwen3.Qwen3_1BPretrainTest
