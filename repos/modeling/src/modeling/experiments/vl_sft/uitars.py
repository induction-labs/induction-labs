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
num_devices = 8
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
            checkpoint_frequency=50,  # Save every 50 steps
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
        freeze_vision=True,
    ),
    train_datapack=VlDatapackConfig(
        dataset_path="gs://induction-labs/jonathan/sampled_trajectories/osworld_uitars_10x_en_5k/samples_correct_expanded_5_under_20_turns_train.jsonl"
        # dataset_path="gs://induction-labs/jonathan/sampled_trajectories/osworld_uitars_10x_en_5k/samples_correct_expanded_5_under_20_turns_train_only_5.jsonl",
    ),
    validation_datapack=VlDatapackConfig(
        dataset_path="gs://induction-labs/jonathan/sampled_trajectories/osworld_uitars_10x_en_5k/samples_correct_expanded_5_under_20_turns_test.jsonl"
    ),
    run=RunConfig(
        lr=LinearLRSchedule(
            peak_lr=1e-5,
            end_lr=5e-6,
            warmup_steps=20,
            end_step=185,  # 10k steps
        ),
        sequence_length=8192 * 2,
        batch_size=32,
        num_steps=185,
        validation_every_n_steps=10,
        distributed=DistributedConfig(
            devices_per_node=num_devices,
        ),
        attn_impl=AttentionImplementation.SDPA,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
        seed=52,
    ),
)

UITarsActionGPU_Test = UITarsActionExperimentConfig_GPU.testing_config(
    num_steps=5,
    enable_wandb=True,
    with_val=True,
    profile=False,
)

UITarsActionExperimentConfig_CPU = UITarsActionExperimentConfig_GPU.model_copy(
    update={"run": UITarsActionExperimentConfig_GPU.run.cpu_config()}
)
UITarsActionSweep = (
    Sweep(UITarsActionGPU_Test).sweep(
        [8192 * 3, 20_000],
        lambda num_workers, exp: (
            exp.run.__setattr__("dataloader_num_workers", num_workers),
            exp,
        )[-1],
    )
    # .sweep(
    #     [16, 8],
    #     lambda num_workers, exp: (
    #         exp.run.__setattr__("dataloader_num_workers", num_workers),
    #         exp,
    #     )[-1]
    # )
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
    # .sweep(range(20, 24), Sweep.S.seed)
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
    #                 end_lr=peak_lr * 0.5,
    #                 warmup_steps=20,
    #                 end_step=185,
    #             )
    #             for peak_lr in [5e-4] #[5e-5, 1e-4, 1e-5]
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
