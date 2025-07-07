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

from modeling.modules.text_pretrain.llama3 import Llama3LITConfig
from modeling.types import AttentionImplementation

from modeling.types import Accelerator, DType

Llama3PretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="llama3_4B_text_pretrain"),
        output_dir="output/text_pretrain",
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
Llama3PretrainTest = Llama3PretrainExperimentConfig.testing_config(
    num_steps=5, enable_wandb=False
)

# mdl export modeling.experiments.text_pretrain.llama3.Llama3PretrainExperimentConfig
# mdl export modeling.experiments.text_pretrain.llama3.Llama3PretrainTest
