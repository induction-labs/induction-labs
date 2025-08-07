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
from modeling.modules.base_module import CompileConfig, OptimizerType
from modeling.modules.vl_sft.qwen_25vl import VlSftLITConfig
from modeling.types import Accelerator, DType
from modeling.utils.cloud_path import CloudPath

model_name = "ByteDance-Seed/UI-TARS-1.5-7B"
# model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
processor_config = VideoProcessorConfig.Qwen25VL(model_name)
run_name = "uitars_sft_7b_yehaw_two_epoch"
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
            checkpoint_frequency=0,  # Save every 10 steps
            checkpoint_first_step=False,  # Save the first step
            checkpoint_last_step=True,  # Save the last step
        ),
    ),
    module=VlSftLITConfig(
        # checkpoint_path=CloudPath.from_str(
        #     # "gs://induction-labs/checkpoints/uitars_sft_7b_yehaw_good_nice/2025-07-31T05-18-29.nTtAsFOt/step_1170/"
        #     "gs://induction-labs/checkpoints/uitars_sft_7b_yehaw_good_nice/2025-07-31T21-46-31.9xPQKPdh/step_620"
        # ),
        model_name=model_name,
        tokenizer_name=model_name,
        optimizer=OptimizerType.ADAMW,
        compile=CompileConfig(),
        # compile=None,
        freeze_vision=True,
        # use_liger_kernel=False,
    ),
    train_datapack=VlDatapackConfig(
        dataset_path="gs://induction-labs/jonathan/halluminate_v1_synth/hard_samples_correct_trajectories_expanded_under_50_train.jsonl"
        # dataset_path="<PLACEHOLDER>",
    ),
    validation_datapack=VlDatapackConfig(
        # dataset_path="gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/samples_correct_trajectories_expanded_under_100.jsonl",
        dataset_path="gs://induction-labs/jonathan/halluminate_v1_synth/hard_samples_correct_trajectories_expanded_under_50_test.jsonl"
        # dataset_path="<PLACEHOLDER>",
    ),
    run=RunConfig(
        lr=LinearLRSchedule(
            peak_lr=1e-5,
            end_lr=5e-6,
            warmup_steps=40,
            end_step=420 * 2,
        ),
        sequence_length=1024 * 22,
        batch_size=1,
        num_steps=420 * 2,
        validation_every_n_steps=50,
        distributed=DistributedConfig(
            devices_per_node=num_devices,
        ),
        attn_impl=AttentionImplementation.FLASH_ATTENTION_2,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
        seed=8492,
    ),
)

UITarsActionGPU_Test = UITarsActionExperimentConfig_GPU.testing_config(
    num_steps=10,
    enable_wandb=True,
    with_val=True,
    profile=False,
)

UITarsActionExperimentConfig_CPU = UITarsActionExperimentConfig_GPU.model_copy(
    update={"run": UITarsActionExperimentConfig_GPU.run.cpu_config()}
)
UITarsActionSweep = (
    Sweep(UITarsActionExperimentConfig_GPU)
    .sweep(
        [2048 * 11],  # 8192*3],  # 2048*10, ],#8192 * 3, 8192 * 4],
        lambda sq, exp: (
            exp.run.__setattr__("sequence_length", sq),
            exp,
        )[-1],
    )
    .sweep(
        [
            # (
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/all_samples_correct_trajectories_expanded_under_50_train.jsonl",
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/all_samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     435*3,
            #     5e-5,
            #     True,
            #     12839,
            # ),
            # (
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/all_samples_correct_trajectories_expanded_under_50_train.jsonl",
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/all_samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     435*3,
            #     1e-4,
            #     True,
            #     128399,
            # ),
            # (
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/only_osworld_samples_correct_trajectories_expanded_under_50_train.jsonl",
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/only_osworld_samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     70*3,
            #     5e-5,
            #     True,
            #     128392,
            # ),
            # (
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/only_osworld_samples_correct_trajectories_expanded_under_50_train.jsonl",
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/only_osworld_samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     70*3,
            #     1e-5,
            #     True,
            #     1829238,
            # ),
            # (
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/only_osworld_samples_correct_trajectories_expanded_under_50_train.jsonl",
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/only_osworld_samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     70*3,
            #     1e-4,
            #     True,
            #     128390,
            # ),
            # (
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/only_osworld_samples_correct_trajectories_expanded_under_50_train.jsonl",
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/only_osworld_samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     70*3,
            #     5e-6,
            #     True,
            #     18938,
            # ),
            # --------------------
            #
            # (
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/all_samples_correct_trajectories_expanded_under_50_train.jsonl",
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/all_samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     # 620 * 2,
            #     430 * 2,
            #     1e-4,
            #     True,
            # ),
            # (
            #     "gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/together__samples_correct_trajectories_expanded_under_50_train_3x.jsonl",
            #     "gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/together__samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     620*3,
            #     5e-5,
            #     True,
            #     12839,
            # ),
            # (
            #     "gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/together__samples_correct_trajectories_expanded_under_50_train.jsonl",
            #     "gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/together__samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     620*3,
            #     1e-4,
            #     True,
            #     43920,
            # ),
            # (
            #     "gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/together_no_special_weighting_samples_correct_trajectories_expanded_under_50_train.jsonl",
            #     "gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/together_no_special_weighting_samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     735 * 2,
            #     1e-5,
            #     True,
            # ),
            # (
            #     "gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/min_3_samples_correct_trajectories_expanded_under_50_train_4x.jsonl",
            #     "gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/min_3_samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     320,
            #     1e-4,
            #     True,
            # ),
            # (
            #     "gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/min_3_samples_correct_trajectories_expanded_under_50_train_4x.jsonl",
            #     "gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/min_3_samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     320,
            #     1e-4,
            #     False,
            # ),
            # (
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/hard_samples_correct_trajectories_expanded_under_50_train.jsonl",
            #     "gs://induction-labs/jonathan/halluminate_v1_synth/hard_samples_correct_trajectories_expanded_under_50_test.jsonl",
            #     46 * 3,
            #     5e-5,
            #     True,
            #     18392,
            # ),
            (
                "gs://induction-labs/jonathan/halluminate_v1_synth/hard_samples_correct_trajectories_expanded_under_50_train.jsonl",
                "gs://induction-labs/jonathan/halluminate_v1_synth/hard_samples_correct_trajectories_expanded_under_50_test.jsonl",
                46 * 3,
                1e-4,
                True,
                40302,
            ),
        ],
        lambda values, exp: (
            exp.train_datapack.__setattr__("dataset_path", values[0]),
            exp.validation_datapack.__setattr__("dataset_path", values[1]),
            exp.run.__setattr__("num_steps", values[2]),
            exp.module.__setattr__("freeze_vision", values[4]),
            exp.metadata.__setattr__(
                "checkpoint",
                GCSCheckpointConfig(
                    checkpoint_prefix=exp.metadata.checkpoint.checkpoint_prefix,
                    checkpoint_frequency=values[2] // 3,
                    checkpoint_first_step=False,  # Save the first step
                    checkpoint_last_step=True,  # Save the last step
                ),
            ),
            exp.run.__setattr__(
                "lr",
                LinearLRSchedule(
                    peak_lr=values[3],
                    end_lr=values[3] * 0.5,
                    warmup_steps=3,
                    end_step=values[2],
                ),
            ),
            exp.run.__setattr__("seed", values[5]),
            exp,
        )[-1],
    )
    # .sweep(
    #     [32],
    #     lambda num_workers, exp: (
    #         exp.run.__setattr__("dataloader_num_workers", num_workers),
    #         exp,
    #     )[-1]
    # )
    # .sweep(
    #     [2],
    #     lambda num_workers, exp: (
    #         exp.run.__setattr__("dataloader_prefetch_factor", num_workers),
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
    #                 warmup_steps=10,
    #                 end_step=420 * 2 * 2,
    #             )
    #             for peak_lr in [5e-5]  # [1e-5, 5e-5, 5e-6, 1e-4]
    #         ),
    #     ],
    #     Sweep.S.lr,
    # )
    # .sweep(
    #     [32],
    #     Sweep.S.batch_size
    # )
    # .sweep(
    # )
    # .sweep(range(20, 24), Sweep.S.seed)
)
# 0.0005


# mdl export modeling.experiments.vl_sft.uitars.UITarsActionExperimentConfig_CPU
# mdl export modeling.experiments.vl_sft.uitars.UITarsActionExperimentConfig_GPU
# mdl export modeling.experiments.vl_sft.uitars.UITarsActionGPU_Test
# mdl sweep modeling.experiments.vl_sft.uitars.UITarsActionSweep
