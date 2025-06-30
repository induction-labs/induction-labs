from __future__ import annotations

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    RunConfig,
    WandbConfig,
    AttentionImplementation,
)
from modeling.types import Accelerator, DType
from modeling.data.video_action import ActionDatapackConfig
from modeling.modules.action_instruct.qwen_25o import Qwen25OActionLITConfig

Qwen25OActionExperimentConfig_CPU = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="mouse_following", name="qwen25o_mouse_follow"),
        output_dir="output/mouse_following",
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
        sequence_length=560,
        batch_size=1,
        steps_per_epoch=1000,
        distributed=DistributedConfig(
            devices_per_node=1,
        ),
        accelerator=Accelerator.CPU,
        precision=DType.fp32,
    ),
)


Qwen25OActionExperimentConfig_GPU = Qwen25OActionExperimentConfig_CPU.model_copy(
    update={
        "run": Qwen25OActionExperimentConfig_CPU.run.model_copy(
            update={
                "accelerator": Accelerator.CUDA,
                "precision": DType.fp16,  # Using mixed precision for GPU training
                "attn_impl": AttentionImplementation.FLASH_ATTENTION_2,
            }
        )
    }
)


# mdl export modeling.experiments.action_instruct.qwen_25o.Qwen25OActionExperimentConfig_CPU
# mdl export modeling.experiments.action_instruct.qwen_25o.Qwen25OActionExperimentConfig_GPU
