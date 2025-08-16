from __future__ import annotations

from pathlib import Path

from modeling.config import (
    AttentionImplementation,
    CombinedDatapackConfig,
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
run_name = "clicks_maxxing"
num_devices = 8
UITarsActionExperimentConfig_GPU = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="UITars_7B_passive", name=run_name),
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
        freeze_vision=False,
        # use_liger_kernel=False,
    ),
    data=CombinedDatapackConfig(
        train_datapack=VlDatapackConfig(
            dataset_paths=[
                "gs://induction-labs/passive_data/smooth_brain_clicks/2025-08-15/reprocess_all_no_loading-01-36-47/samples.jsonl",
                # "gs://induction-labs/passive_data/2025-08-13/all_data_reprocessed_good-04-19-14/expanded_samples.jsonl"
            ]
            # dataset_path="<PLACEHOLDER>",
        ),
        validation_datapack=VlDatapackConfig(
            # dataset_path="gs://induction-labs/jonathan/sampled_trajectories/all_trajectories/samples_correct_trajectories_expanded_under_100.jsonl",
            dataset_paths=[
                "gs://induction-labs/passive_data/smooth_brain_clicks/2025-08-15/reprocess_all_no_loading-01-36-47/samples.jsonl",
                # "gs://induction-labs/passive_data/2025-08-13/all_data_reprocessed_good-04-19-14/expanded_samples.jsonl"
            ]
            # dataset_path="<PLACEHOLDER>",
        ),
    ),
    run=RunConfig(
        lr=LinearLRSchedule(
            peak_lr=1e-5,
            end_lr=5e-6,
            warmup_steps=40,
            end_step=420 * 2,
        ),
        sequence_length=1024 * 4,
        batch_size=1,
        num_steps=420 * 2,
        validation_every_n_steps=20,
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
            # ------- b200 ------------
            {
                "init": None,
                "data": (
                    data := [
                        # (
                        #     "gs://induction-labs/passive_data/2025-08-13/all_data_reprocessed_good-04-19-14/random_expanded_samples.jsonl",
                        #     2352,
                        # ),
                        # (
                        #     "gs://induction-labs/passive_data/2025-08-14/reprocess_all-01-20-24/expanded_samples.jsonl",
                        #     1631,
                        # ),
                        # (
                        #     "gs://induction-labs/passive_data/2025-08-14/reprocess_all-01-20-24/expanded_samples.jsonl",
                        #     3425,
                        # ),
                        # (
                        #     "gs://induction-labs/passive_data/2025-08-14/reprocess_all_no_loading-04-30-15/expanded_random_truncate.jsonl",
                        #     3398,
                        # ),
                        (
                            "gs://induction-labs/passive_data/smooth_brain_clicks/2025-08-15/reprocess_all_no_loading-01-36-47/samples.jsonl",
                            2623,
                        ),
                        *(
                            (
                                f"gs://induction-labs/passive_data/smooth_brain_clicks/2025-08-15/click_kunal_mahdi-22-13-03/samples_{i}.jsonl",
                                100,
                            )
                            for i in range(70)
                        ),
                    ]
                ),
                "epochs": 1,
                "batch_size": 64,
                "num_steps": int(sum(k[1] for k in data) / 64),
                "start_lr": 2e-5,
                "end_lr_ratio": 0.5,
                "warmup_steps": 30,
                "seed": 38298,
                "val_data": [
                    "gs://induction-labs/passive_data/smooth_brain_clicks/2025-08-15/click_kunal_mahdi-22-13-03/samples_71.jsonl",
                    "gs://induction-labs/passive_data/smooth_brain_clicks/2025-08-15/click_kunal_mahdi-22-13-03/samples_72.jsonl",
                    "gs://induction-labs/passive_data/smooth_brain_clicks/2025-08-15/click_kunal_mahdi-22-13-03/samples_73.jsonl",
                ],
            },
        ],
        lambda values, exp: (
            exp.data.train_datapack.__setattr__(
                "dataset_paths", [k[0] for k in values["data"]]
            ),
            exp.data.validation_datapack.__setattr__(
                "dataset_paths", values["val_data"]
            ),
            exp.run.__setattr__("num_steps", values["num_steps"] * values["epochs"]),
            exp.module.__setattr__("freeze_vision", False),
            exp.module.__setattr__(
                "checkpoint_path",
                CloudPath.from_str(values["init"]) if values["init"] else None,
            ),
            exp.metadata.__setattr__(
                "checkpoint",
                GCSCheckpointConfig(
                    checkpoint_prefix=exp.metadata.checkpoint.checkpoint_prefix,
                    # checkpoint_prefix=CloudPath.from_str(
                    #     "gs://induction-labs/checkpoints/uitars_sft_7b_yehaw_two_epoch/2025-08-05T11-05-09.LRJVBr96/step_480/",
                    # ),
                    checkpoint_frequency=values["num_steps"] // 3,
                    checkpoint_first_step=True,  # Save the first step
                    checkpoint_last_step=True,  # Save the last step
                ),
            ),
            exp.run.__setattr__(
                "lr",
                LinearLRSchedule(
                    peak_lr=values["start_lr"],
                    end_lr=values["start_lr"] * values["end_lr_ratio"],
                    warmup_steps=values["warmup_steps"],
                    end_step=values["num_steps"] * values["epochs"],
                ),
            ),
            exp.run.__setattr__("seed", values["seed"]),
            exp.run.__setattr__("batch_size", values["batch_size"]),
            exp,
        )[-1],
    )
)


# mdl export modeling.experiments.vl_sft.uitars.UITarsActionExperimentConfig_CPU
# mdl export modeling.experiments.vl_sft.uitars.UITarsActionExperimentConfig_GPU
# mdl export modeling.experiments.vl_sft.uitars.UITarsActionGPU_Test
# mdl sweep modeling.experiments.vl_sft.uitars.UITarsActionSweep
