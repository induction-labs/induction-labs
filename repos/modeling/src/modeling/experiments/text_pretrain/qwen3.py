from __future__ import annotations

from pathlib import Path

from modeling.config import (
    CombinedDatapackConfig,
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    GCSCheckpointConfig,
    LinearLRSchedule,
    ProfileConfig,
    RunConfig,
    WandbConfig,
)
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.base_module import CompileConfig
from modeling.modules.text_pretrain.qwen3 import Qwen3LITConfig
from modeling.types import Accelerator, AttentionImplementation
from modeling.utils.cloud_path import CloudPath

# from modeling.config.distributed import DistributedConfig, ShardingConfig
run_name = "qwen3_4B_text_pretrain"
Qwen3PretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name=run_name),
        output_dir=Path("./output/qwen3_text_pretrain"),
        checkpoint=GCSCheckpointConfig(
            checkpoint_prefix=CloudPath.from_str(
                f"gs://induction-labs/checkpoints/{run_name}",
            ),
            checkpoint_frequency=0,  # Save every 10 steps
            checkpoint_first_step=False,  # Save the first step
            checkpoint_last_step=True,  # Save the last step
        ),
    ),
    module=Qwen3LITConfig(
        model_name="Qwen/Qwen3-4B",
        tokenizer_name="Qwen/Qwen3-4B",
        # activation_checkpointing=None,  # Optional activation checkpointing config
        compile=CompileConfig(),  # Uncomment if you want to use compilation
    ),
    data=CombinedDatapackConfig(
        train_datapack=TextPretrainDatapackConfig(),
        validation_datapack=TextPretrainDatapackConfig(),
    ),
    run=RunConfig(
        lr=LinearLRSchedule.constant_lr(1e-5),
        sequence_length=4096,
        batch_size=1,
        num_steps=20,
        distributed=DistributedConfig(
            devices_per_node=1,
        ),
        attn_impl=AttentionImplementation.FLASH_ATTENTION_2,
        accelerator=Accelerator.CUDA,
        profile=ProfileConfig(),
    ),
)
Qwen3PretrainTest = Qwen3PretrainExperimentConfig.testing_config(
    num_steps=5, enable_wandb=False
)

# mdl export modeling.experiments.text_pretrain.qwen3.Qwen3PretrainExperimentConfig
# mdl export modeling.experiments.text_pretrain.qwen3.Qwen3PretrainTest
