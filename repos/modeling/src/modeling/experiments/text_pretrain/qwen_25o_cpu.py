from __future__ import annotations

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    RunConfig,
    WandbConfig,
)
from modeling.types import Accelerator, DType
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
        lr=1e-3,
        sequence_length=128,
        batch_size=1,
        steps_per_epoch=1000,
        distributed=DistributedConfig(
            devices_per_node=1,
        ),
        accelerator=Accelerator.CPU,
        precision=DType.fp32,
    ),
)

# mdl export modeling.experiments.text_pretrain.qwen_25o_cpu.Qwen25OPretrainExperimentConfig
