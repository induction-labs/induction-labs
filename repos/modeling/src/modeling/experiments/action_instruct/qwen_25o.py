from __future__ import annotations

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    RunConfig,
    WandbConfig,
    GCSCheckpointConfig,
)
from modeling.types import Accelerator, DType
from modeling.data.video_action import RangeActionDatapackConfig
from modeling.modules.action_instruct.qwen_25o import Qwen25OActionLITConfig
from modeling.utils.cloud_path import CloudPath

Qwen25OActionExperimentConfig_GPU = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(
            project="mouse_following", name="qwen25o_mouse_follow_big_mlp"
        ),
        output_dir="output",
        # checkpoint=None,
        checkpoint=GCSCheckpointConfig(
            checkpoint_prefix=CloudPath.from_str(
                "gs://induction-labs/checkpoints/qwen25o_mouse_follow/test_big_mlp",
            ),
            checkpoint_frequency=1000,  # Save every 1000 steps
            checkpoint_first_step=False,  # Save the first step
            checkpoint_last_step=True,  # Save the last step
        ),
    ),
    module=Qwen25OActionLITConfig(
        checkpoint_path=CloudPath.from_str(
            "gs://induction-labs/checkpoints/qwen25o_mouse_follow/test_noise_2/2025-07-04T04-05-09/step_-1"
        )
    ),
    datapack=RangeActionDatapackConfig(
        # prefix="gs://induction-labs/jonathan/synth/garbage_cursor_follow_v1/sample_",
        prefix="gs://induction-labs/jonathan/synth/cursor_follow_v3/sample_",
        # prefix="gs://induction-labs/jonathan/synth/noise_cursor_follow_v1/sample_",
        end_index=60000,  # 60k samples
    ),
    run=RunConfig(
        lr=1e-5,
        sequence_length=4096,
        batch_size=4,
        steps_per_epoch=60000 / 4,
        distributed=DistributedConfig(
            devices_per_node=8,
        ),
        # "attn_impl": AttentionImplementation.FLASH_ATTENTION_2,
        # attn_impl=AttentionImplementation.FLASH_ATTENTION_2,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
    ),
)

Qwen25OActionGPU_Test = Qwen25OActionExperimentConfig_GPU.testing_config(num_steps=10)

Qwen25OActionExperimentConfig_CPU = Qwen25OActionExperimentConfig_GPU.model_copy(
    update={"run": Qwen25OActionExperimentConfig_GPU.run.cpu_config()}
)


# mdl export modeling.experiments.action_instruct.qwen_25o.Qwen25OActionExperimentConfig_CPU
# mdl export modeling.experiments.action_instruct.qwen_25o.Qwen25OActionExperimentConfig_GPU
# mdl export modeling.experiments.action_instruct.qwen_25o.Qwen25OActionGPU_Test
