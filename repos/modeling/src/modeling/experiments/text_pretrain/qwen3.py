from __future__ import annotations

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    RunConfig,
    WandbConfig,
)
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.text_pretrain.default import TextPretrainLITConfig

Qwen3PretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="qwen3_text_pretrain"),
        output_dir="output/text_pretrain",
    ),
    module=TextPretrainLITConfig(
        model_name="Qwen/Qwen3-4B",
        tokenizer_name="Qwen/Qwen3-4B",
    ),
    datapack=TextPretrainDatapackConfig(),
    run=RunConfig(
        sequence_length=1024,  # Default sequence length
        batch_size=1,
        steps_per_epoch=1000,  # Number of steps per epoch
        distributed=DistributedConfig(
            devices_per_node=1,
        ),
    ),
)

# mdl export modeling.experiments.text_pretrain.qwen3.Qwen3PretrainExperimentConfig
