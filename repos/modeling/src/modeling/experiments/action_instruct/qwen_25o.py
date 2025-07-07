from __future__ import annotations

from modeling.config import (
    AttentionImplementation,
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    GCSCheckpointConfig,
    RunConfig,
    WandbConfig,
    LinearLRSchedule,
)
from pathlib import Path
from modeling.types import Accelerator, DType
from modeling.data.video_action import RangeActionDatapackConfig
from modeling.modules.action_instruct.qwen_25o import Qwen25OActionLITConfig
from modeling.utils.cloud_path import CloudPath
from modeling.modules.base_module import CompileConfig

run_name = "qwen25o_mouse_follow_overfit"
Qwen25OActionExperimentConfig_GPU = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="mouse_following", name=run_name),
        output_dir=Path("./output") / run_name,
        # checkpoint=None,
        checkpoint=GCSCheckpointConfig(
            checkpoint_prefix=CloudPath.from_str(
                f"gs://induction-labs/checkpoints/{run_name}",
            ),
            checkpoint_frequency=1000,  # Save every 1000 steps
            checkpoint_first_step=False,  # Save the first step
            checkpoint_last_step=True,  # Save the last step
        ),
    ),
    module=Qwen25OActionLITConfig(
        # checkpoint_path=CloudPath.from_str(
        #     "gs://induction-labs/checkpoints/qwen25o_mouse_follow/test_noise_2/2025-07-04T04-05-09/step_-1"
        # )
        # freeze_action_embedding=True,
        # compile=None,
        compile=CompileConfig(),
    ),
    datapack=RangeActionDatapackConfig(
        # prefix="gs://induction-labs/jonathan/synth/garbage_cursor_follow_v1/sample_",
        prefix="gs://induction-labs/jonathan/synth/cursor_follow_v3/sample_",
        # prefix="gs://induction-labs/jonathan/synth/noise_cursor_follow_v1/sample_",
        end_index=4000,  # 60k samples
    ),
    run=RunConfig(
        lr=LinearLRSchedule(
            peak_lr=1e-3,
            end_lr=1e-5,
            warmup_steps=16,
            end_step=500,  # 10k steps
        ),
        sequence_length=1024,
        batch_size=1,
        steps_per_epoch=1000,
        validation_every_n_steps=-1,
        distributed=DistributedConfig(
            devices_per_node=1,
        ),
        attn_impl=AttentionImplementation.SDPA,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
    ),
)

Qwen25OActionGPU_Test = Qwen25OActionExperimentConfig_GPU.testing_config(
    num_steps=5, enable_wandb=False
)

Qwen25OActionExperimentConfig_CPU = Qwen25OActionExperimentConfig_GPU.model_copy(
    update={"run": Qwen25OActionExperimentConfig_GPU.run.cpu_config()}
)

# mdl export modeling.experiments.action_instruct.qwen_25o.Qwen25OActionExperimentConfig_CPU
# mdl export modeling.experiments.action_instruct.qwen_25o.Qwen25OActionExperimentConfig_GPU
# mdl export modeling.experiments.action_instruct.qwen_25o.Qwen25OActionGPU_Test
