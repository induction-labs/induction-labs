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
)
from modeling.config.sweep import Sweep
from modeling.data.trajectory_train import VlDatapackConfig
from modeling.data.video_action import (
    VideoProcessorConfig,
)
from modeling.modules.base_module import OptimizerType
from modeling.modules.vl_sft.qwen_25vl import VlSftLITConfig
from modeling.types import Accelerator, DType
from modeling.utils.cloud_path import CloudPath

# from modeling.modules.base_module import CompileConfig

processor_config = VideoProcessorConfig.Qwen25VL("ByteDance-Seed/UI-TARS-1.5-7B")
run_name = "uitars_sft_7b"
num_devices = 1
UITarsActionExperimentConfig_GPU = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="UITars_7B_sft", name=run_name),
        # wandb=None,
        output_dir=Path("./output") / run_name,
        # checkpoint=None,
        checkpoint=GCSCheckpointConfig(
            checkpoint_prefix=CloudPath.from_str(
                f"gs://induction-labs/checkpoints/{run_name}",
            ),
            checkpoint_frequency=100,  # Save every 100 steps
            checkpoint_first_step=False,  # Save the first step
            checkpoint_last_step=True,  # Save the last step
        ),
    ),
    module=VlSftLITConfig(
        # checkpoint_path=CloudPath.from_str(
        #     "gs://induction-labs/checkpoints/UITars_7B_uninitialized/2025-07-17T23-05-38/step_100"
        # ),
        model_name="ByteDance-Seed/UI-TARS-1.5-7B",
        tokenizer_name="ByteDance-Seed/UI-TARS-1.5-7B",
        optimizer=OptimizerType.ADAMW,
    ),
    train_datapack=VlDatapackConfig(),
    validation_datapack=VlDatapackConfig(),
    run=RunConfig(
        lr=LinearLRSchedule(
            peak_lr=1e-4,
            end_lr=1e-5,
            warmup_steps=100,
            end_step=500,  # 10k steps
        ),
        sequence_length=8192,
        batch_size=num_devices,
        num_steps=500,
        validation_every_n_steps=50,
        distributed=DistributedConfig(
            devices_per_node=num_devices,
        ),
        attn_impl=AttentionImplementation.SDPA,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
        seed=42,
    ),
)

UITarsActionGPU_Test = UITarsActionExperimentConfig_GPU.testing_config(
    num_steps=5,
    enable_wandb=False,
    with_val=False,
    profile=False,
)

UITarsActionExperimentConfig_CPU = UITarsActionExperimentConfig_GPU.model_copy(
    update={"run": UITarsActionExperimentConfig_GPU.run.cpu_config()}
)
UITarsActionSweep = (
    Sweep(UITarsActionExperimentConfig_GPU)
    # .sweep(
    #     [True, False],
    #     lambda freeze_vision, exp: (
    #         exp.module.__setattr__("freeze_vision", freeze_vision),
    #         exp,
    #     )[-1],
    # )
    # .sweep(
    #     [
    #         None,
    #         CloudPath.from_str(
    #             "gs://induction-labs/checkpoints/UITars_7B_uninitialized/2025-07-17T23-05-38/step_100"
    #         ),
    #         CloudPath.from_str(
    #             "gs://induction-labs/checkpoints/sweeps_optimizer/2025-07-22T17-28-51.iGEDRrvU/step_-1"
    #         ),
    #     ],
    #     lambda checkpoint, exp: (
    #         exp.module.__setattr__("checkpoint_path", checkpoint),
    #         exp,
    #     )[-1],
    # )
    .sweep(range(20, 24), Sweep.S.seed)
    # .sweep(
    #     [
    #         # UITarsActionLITConfig.CursorPredictionLoss.L2_DISTANCE,
    #         (UITarsActionLITConfig.CursorPredictionLoss.ANALYTICAL_DISTANCE, 50),
    #         (UITarsActionLITConfig.CursorPredictionLoss.COEFFICIENTS_DISTANCE, 300),
    #         (UITarsActionLITConfig.CursorPredictionLoss.ANALYTICAL_DISTANCE, None),
    #         (UITarsActionLITConfig.CursorPredictionLoss.COEFFICIENTS_DISTANCE, None),
    #     ],
    #     lambda loss, exp: (
    #         exp.module.__setattr__("loss_type", loss[0]),
    #         exp.run.__setattr__("grad_clip", loss[1]),
    #         exp,
    #     )[-1],
    # )
    # .sweep(
    #     [
    #         # OptimizerType.ADAMW,
    #         OptimizerType.ADAGRAD,
    #         OptimizerType.SGD,
    #     ],
    #     lambda optimizer, exp: (
    #         exp.module.__setattr__("optimizer", optimizer),
    #         exp,
    #     )[-1],
    # )
    # .sweep(
    #     [
    #         # LinearLRSchedule(
    #         #     peak_lr=1e-5,
    #         #     end_lr=1e-5,
    #         #     warmup_steps=0,
    #         #     end_step=3_000,
    #         # ),
    #         *(
    #             LinearLRSchedule(
    #                 peak_lr=peak_lr,
    #                 end_lr=1e-5,
    #                 warmup_steps=warmup_steps,
    #                 end_step=3_000,
    #             )
    #             for peak_lr, warmup_steps in Sweep.S.product([5e-5], [0])
    #         ),
    #     ],
    #     Sweep.S.lr,
    # )
)
# 0.0005


# mdl export modeling.experiments.vl_sft.uitars.UITarsActionExperimentConfig_CPU
# mdl export modeling.experiments.vl_sft.uitars.UITarsActionExperimentConfig_GPU
# mdl export modeling.experiments.vl_sft.uitars.UITarsActionGPU_Test
# mdl sweep modeling.experiments.vl_sft.uitars.UITarsActionSweep
