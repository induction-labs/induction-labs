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

Qwen3_1BPretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="qwen3_0.6B_text_pretrain"),
        # wandb=None,
        output_dir="output/text_pretrain",
        checkpoint=None,
    ),
    module=Qwen3LITConfig(
        model_name="Qwen/Qwen3-0.6B",
        tokenizer_name="Qwen/Qwen3-0.6B",
    ),
    datapack=TextPretrainDatapackConfig(),
    run=RunConfig(
        validation_every_n_steps=10,
        validation_steps=10,
        lr=1e-3,
        sequence_length=1024,  # Default sequence length
        batch_size=1,
        steps_per_epoch=1000,  # Number of steps per epoch
        distributed=DistributedConfig(
            devices_per_node=1,
            num_nodes=1,
        ),
    ),
)

# mdl export modeling.experiments.text_pretrain.qwen3_1B.Qwen3_1BPretrainExperimentConfig
