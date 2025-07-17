from __future__ import annotations

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    RunConfig,
    WandbConfig,
)
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.text_pretrain.qwen_25o import Qwen25OLITConfig


from modeling.config import (
    LinearLRSchedule,
)

from modeling.types import AttentionImplementation
from pathlib import Path
from modeling.types import Accelerator, DType
# from modeling.modules.base_module import CompileConfig


# mdl export modeling.experiments.text_pretrain.qwen3.Qwen3PretrainExperimentConfig
# mdl export modeling.experiments.text_pretrain.qwen3.Qwen3PretrainTest

Qwen25OPretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="qwen25o_text_pretrain"),
        output_dir=Path("./output/qwen25o_text_pretrain"),
        checkpoint=None,
    ),
    module=Qwen25OLITConfig(
        # compile=CompileConfig(),
        model_name="Qwen/Qwen2.5-Omni-3B",
        tokenizer_name="Qwen/Qwen2.5-Omni-3B",
        # activation_checkpointing=None,
    ),
    datapack=TextPretrainDatapackConfig(),
    run=RunConfig(
        lr=LinearLRSchedule.constant_lr(1e-5),
        sequence_length=4096,
        batch_size=1,
        num_steps=5000,
        distributed=DistributedConfig(
            devices_per_node=1,
        ),
        attn_impl=AttentionImplementation.SDPA,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
        # profile=ProfileConfig(),
    ),
)

Qwen25OPretrainTest = Qwen25OPretrainExperimentConfig.testing_config(
    num_steps=5, enable_wandb=False
)
# mdl export modeling.experiments.text_pretrain.qwen_25o.Qwen25OPretrainExperimentConfig
# mdl export modeling.experiments.text_pretrain.qwen_25o.Qwen25OPretrainTest
