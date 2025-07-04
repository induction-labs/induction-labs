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
            project="mouse_following", name="qwen25o_mouse_follow_vision_unfrozen_noise"
        ),
        output_dir="output",
        checkpoint=GCSCheckpointConfig(
            checkpoint_prefix=CloudPath.from_str(
                "gs://induction-labs/checkpoints/qwen25o_mouse_follow/test_noise_2",
            ),
            checkpoint_frequency=1000,  # Save every 1000 steps
            checkpoint_first_step=True,  # Save the first step
            checkpoint_last_step=True,  # Save the last step
        ),
    ),
    module=Qwen25OActionLITConfig(),
    datapack=RangeActionDatapackConfig(
        # prefix="gs://induction-labs/jonathan/synth/garbage_cursor_follow_v1/sample_",
        prefix="gs://induction-labs/jonathan/synth/cursor_follow_v2/sample_",
        # prefix="gs://induction-labs/jonathan/synth/noise_cursor_follow_v1/sample_",
        end_index=5000,  # 5k samples
    ),
    run=RunConfig(
        lr=1e-5,
        sequence_length=2048,
        batch_size=1,
        steps_per_epoch=5000,
        distributed=DistributedConfig(
            devices_per_node=1,
        ),
        # "attn_impl": AttentionImplementation.FLASH_ATTENTION_2,
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
