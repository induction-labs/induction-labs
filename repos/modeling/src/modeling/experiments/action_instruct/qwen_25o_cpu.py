from __future__ import annotations

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    RunConfig,
    WandbConfig,
)
from modeling.types import Accelerator, DType
from modeling.data.video_action import ActionDatapackConfig
from modeling.modules.action_instruct.qwen_25o import Qwen25OActionLITConfig

Qwen25OActionExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="qwen25o_text_pretrain"),
        output_dir="output/text_pretrain",
    ),
    module=Qwen25OActionLITConfig(),
    datapack=ActionDatapackConfig(
        processed_data_paths=[
            f"gs://induction-labs/jonathan/synth/cursor_follow_v2/sample_{i}.zarr"
            for i in range(0, 4)
        ]
    ),
    run=RunConfig(
        lr=1e-3,
        sequence_length=1026,
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
