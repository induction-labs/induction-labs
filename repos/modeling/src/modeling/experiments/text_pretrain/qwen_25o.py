from __future__ import annotations

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    RunConfig,
    WandbConfig,
)
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.text_pretrain.qwen_25o import Qwen25OLITConfig

Qwen25OPretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="qwen25o_text_pretrain"),
        output_dir="output/text_pretrain",
    ),
    module=Qwen25OLITConfig(),
    datapack=TextPretrainDatapackConfig(),
    run=RunConfig(
        sequence_length=1024,  # Default sequence length
        batch_size=16,
        steps_per_epoch=1000,  # Number of steps per epoch
        distributed=DistributedConfig(
            devices_per_node=2,
        ),
    ),
)

# mdl export modeling.experiments.text_pretrain.qwen_25o.Qwen25OPretrainExperimentConfig
