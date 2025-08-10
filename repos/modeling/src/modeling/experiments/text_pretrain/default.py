from __future__ import annotations

from pathlib import Path

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    LinearLRSchedule,
    RunConfig,
    WandbConfig,
)
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.text_pretrain.default import TextPretrainLITConfig

TextPretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="text_pretrain"),
        output_dir=Path("./output/text_pretrain"),
        checkpoint=None,
    ),
    module=TextPretrainLITConfig(),
    datapack=TextPretrainDatapackConfig(),
    run=RunConfig(
        sequence_length=1024,  # Default sequence length
        batch_size=2,
        num_steps=1000,  # Number of steps per epoch
        distributed=DistributedConfig.mock_data(),
        lr=LinearLRSchedule.constant_lr(1e-5),  # Constant learning rate
    ),
)

# mdl export modeling.experiments.text_pretrain.default.TextPretrainExperimentConfig
