from __future__ import annotations

from pathlib import Path

from modeling.config import (
    AttentionImplementation,
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    GCSCheckpointConfig,
    LinearLRSchedule,
    RunConfig,
    WandbConfig,
    # ProfileConfig
)
from modeling.config.sweep import Sweep
from modeling.data.video_action import RangeActionDatapackConfig
from modeling.modules.action_instruct.qwen_25o import Qwen25OActionLITConfig
from modeling.modules.base_module import OptimizerType
from modeling.types import Accelerator, DType
from modeling.utils.cloud_path import CloudPath

# from modeling.modules.base_module import CompileConfig

run_name = "sweeps_optimizer"
num_devices = 2
Qwen25OActionExperimentConfig_GPU = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="qwen25o_7B_real_data", name=run_name),
        # wandb=None,
        output_dir=Path("./output") / run_name,
        # checkpoint=None,
        checkpoint=GCSCheckpointConfig(
            checkpoint_prefix=CloudPath.from_str(
                f"gs://induction-labs/checkpoints/{run_name}",
            ),
            checkpoint_frequency=0,  # Save every 1000 steps
            checkpoint_first_step=False,  # Save the first step
            checkpoint_last_step=True,  # Save the last step
        ),
    ),
    module=Qwen25OActionLITConfig(
        # checkpoint_path=CloudPath.from_str(
        #     "gs://induction-labs/checkpoints/qwen25o_7B_uninitialized/2025-07-17T23-05-38/step_100"
        # ),
        model_name="Qwen/Qwen2.5-Omni-7B",
        tokenizer_name="Qwen/Qwen2.5-Omni-7B",
        freeze_vision=False,
        freeze_network=False,
        freeze_action_embedding=False,
        freeze_action_head=False,
        loss_type=Qwen25OActionLITConfig.CursorPredictionLoss.L2_DISTANCE,
        # compile=None,
        # compile=CompileConfig(),
    ),
    datapack=RangeActionDatapackConfig(
        # prefix="gs://induction-labs/jonathan/synth/garbage_cursor_follow_v1/sample_",
        prefix="gs://induction-labs/jonathan/synth/cursor_follow_v3/sample_",
        # prefix="gs://induction-labs/jonathan/synth/noise_cursor_follow_v1/sample_",
        end_index=60_000,  # 60k samples
    ),
    run=RunConfig(
        lr=LinearLRSchedule(
            peak_lr=1e-4,
            end_lr=1e-5,
            warmup_steps=0,
            end_step=100,  # 10k steps
        ),
        sequence_length=4096,
        batch_size=num_devices,
        num_steps=100,
        validation_every_n_steps=20,
        distributed=DistributedConfig(
            devices_per_node=num_devices,
        ),
        attn_impl=AttentionImplementation.SDPA,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
        seed=52,
    ),
)

Qwen25OActionGPU_Test = Qwen25OActionExperimentConfig_GPU.testing_config(
    num_steps=25,
    enable_wandb=True,
    with_val=True,
    profile=False,
)

Qwen25OActionExperimentConfig_CPU = Qwen25OActionExperimentConfig_GPU.model_copy(
    update={"run": Qwen25OActionExperimentConfig_GPU.run.cpu_config()}
)


Qwen25oActionSweep = (
    Sweep(Qwen25OActionExperimentConfig_GPU)
    .sweep(
        [
            None,
            CloudPath.from_str(
                "gs://induction-labs/checkpoints/qwen25o_7B_uninitialized/2025-07-17T23-05-38/step_100"
            ),
        ],
        lambda checkpoint, exp: (
            exp.module.__setattr__("checkpoint_path", checkpoint),
            exp,
        )[-1],
    )
    .sweep(
        [
            # Qwen25OActionLITConfig.CursorPredictionLoss.L2_DISTANCE,
            (Qwen25OActionLITConfig.CursorPredictionLoss.ANALYTICAL_DISTANCE, 50),
            (Qwen25OActionLITConfig.CursorPredictionLoss.COEFFICIENTS_DISTANCE, 300),
            (Qwen25OActionLITConfig.CursorPredictionLoss.ANALYTICAL_DISTANCE, None),
            (Qwen25OActionLITConfig.CursorPredictionLoss.COEFFICIENTS_DISTANCE, None),
        ],
        lambda loss, exp: (
            exp.module.__setattr__("loss_type", loss[0]),
            exp.run.__setattr__("grad_clip", loss[1]),
            exp,
        )[-1],
    )
    .sweep(
        [
            # OptimizerType.ADAMW,
            OptimizerType.ADAGRAD,
            OptimizerType.SGD,
        ],
        lambda optimizer, exp: (
            exp.module.__setattr__("optimizer", optimizer),
            exp,
        )[-1],
    )
    # .sweep(
    #     [
    #         LinearLRSchedule(
    #             peak_lr=1e-5,
    #             end_lr=1e-5,
    #             warmup_steps=0,
    #             end_step=3_000,
    #         ),
    #         *(
    #             LinearLRSchedule(
    #                 peak_lr=peak_lr,
    #                 end_lr=1e-5,
    #                 warmup_steps=warmup_steps,
    #                 end_step=3_000,
    #             )
    #             for peak_lr, warmup_steps in Sweep.S.product(
    #                 [1e-3, 5e-4, 5e-5], [0, 20, 50, 100]
    #             )
    #         ),
    #     ],
    #     Sweep.S.lr,
    # )
)
# 0.0005


# mdl export modeling.experiments.action_instruct.qwen_25o.Qwen25OActionExperimentConfig_CPU
# mdl export modeling.experiments.action_instruct.qwen_25o.Qwen25OActionExperimentConfig_GPU
# mdl export modeling.experiments.action_instruct.qwen_25o.Qwen25OActionGPU_Test
# mdl sweep modeling.experiments.action_instruct.qwen_25o.Qwen25oActionSweep
