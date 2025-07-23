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
from modeling.data.video_action import (
    RangeActionDatapackConfig,
    VideoProcessorConfig,
    calc_min_num_tokens_for_n_actions,
    make_raw_prompt,
)
from modeling.modules.action_instruct.qwen_25vl import Qwen25VLActionLITConfig
from modeling.modules.base_module import OptimizerType
from modeling.types import Accelerator, DType
from modeling.utils.cloud_path import CloudPath

# from modeling.modules.base_module import CompileConfig

processor_config = VideoProcessorConfig.Qwen25VL("ByteDance-Seed/UI-TARS-1.5-7B")
raw_prompt = make_raw_prompt(
    processor_config,
    suffix="",
)
run_name = "uitars_lr_sweep"
num_devices = 1
UITarsActionExperimentConfig_GPU = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="UITars_7B_lr_sweep", name=run_name),
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
    module=Qwen25VLActionLITConfig(
        # checkpoint_path=CloudPath.from_str(
        #     "gs://induction-labs/checkpoints/UITars_7B_uninitialized/2025-07-17T23-05-38/step_100"
        # ),
        model_name="ByteDance-Seed/UI-TARS-1.5-7B",
        # tokenizer_name="Qwen/Qwen2.5-Omni-7B",
        freeze_vision=True,
        freeze_network=False,
        freeze_action_embedding=False,
        freeze_action_head=False,
        loss_type=Qwen25VLActionLITConfig.CursorPredictionLoss.L2_DISTANCE,
        optimizer=OptimizerType.ADAMW,
        # compile=None,
        # compile=CompileConfig(),
    ),
    train_datapack=RangeActionDatapackConfig(
        # prefix="gs://induction-labs/jonathan/synth/garbage_cursor_follow_v1/sample_",
        prefix="gs://induction-labs/jonathan/synth/cursor_follow_v3/sample_",
        raw_prompt=raw_prompt,
        processor_config=processor_config,
        # prefix="gs://induction-labs/jonathan/synth/noise_cursor_follow_v1/sample_",
        end_index=55_000,  # 60k samples
    ),
    validation_datapack=RangeActionDatapackConfig(
        # prefix="gs://induction-labs/jonathan/synth/garbage_cursor_follow_v1/sample_",
        prefix="gs://induction-labs/jonathan/synth/cursor_follow_v3/sample_",
        raw_prompt=raw_prompt,
        processor_config=processor_config,
        # prefix="gs://induction-labs/jonathan/synth/noise_cursor_follow_v1/sample_",
        start_index=55_000,  # 60k samples
        end_index=60_000,  # 60k samples
    ),
    run=RunConfig(
        lr=LinearLRSchedule(
            peak_lr=1e-3,
            end_lr=1e-5,
            warmup_steps=200,
            end_step=3000,  # 10k steps
        ),
        sequence_length=calc_min_num_tokens_for_n_actions(
            840 * 476, 8, raw_prompt, processor_config
        ),
        batch_size=num_devices,
        num_steps=4000,
        validation_every_n_steps=100,
        distributed=DistributedConfig(
            devices_per_node=num_devices,
        ),
        attn_impl=AttentionImplementation.FLASH_ATTENTION_2,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
        seed=1,
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


# mdl export modeling.experiments.action_instruct.uitars.UITarsActionExperimentConfig_CPU
# mdl export modeling.experiments.action_instruct.uitars.UITarsActionExperimentConfig_GPU
# mdl export modeling.experiments.action_instruct.uitars.UITarsActionGPU_Test
# mdl sweep modeling.experiments.action_instruct.uitars.UITarsActionSweep
