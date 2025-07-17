from __future__ import annotations

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    RunConfig,
    WandbConfig,
    ProfileConfig,
    LinearLRSchedule,
)
from pathlib import Path
from modeling.data.text_train import TextPretrainDatapackConfig

from modeling.modules.text_pretrain.llama3 import Llama3LITConfig
from modeling.types import AttentionImplementation

from modeling.types import Accelerator, DType

bs = 8
experiment_name = "llama3_text_pretrain.shard_test"

Llama3PretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name=experiment_name),
        output_dir=Path(f"./output/{experiment_name}"),
        checkpoint=None,
    ),
    module=Llama3LITConfig(
        model_name="meta-llama/Meta-Llama-3-8B",
        tokenizer_name="meta-llama/Meta-Llama-3-8B",
    ),
    datapack=TextPretrainDatapackConfig(),
    run=RunConfig(
        lr=LinearLRSchedule(
            peak_lr=1e-3,
            end_lr=1e-5,
            warmup_steps=16,
            end_step=500,  # 10k steps
        ),
        sequence_length=4096,
        batch_size=1,
        num_steps=5000,
        distributed=DistributedConfig(
            devices_per_node=bs, sharding=DistributedConfig.ShardingConfig(FSDP=bs)
        ),
        attn_impl=AttentionImplementation.SDPA,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
        profile=ProfileConfig(),
    ),
)
Llama3PretrainTest = Llama3PretrainExperimentConfig.testing_config(
    num_steps=5, enable_wandb=False
)

# mdl export modeling.experiments.text_pretrain.llama3.Llama3PretrainExperimentConfig
# mdl export modeling.experiments.text_pretrain.llama3.Llama3PretrainTest
