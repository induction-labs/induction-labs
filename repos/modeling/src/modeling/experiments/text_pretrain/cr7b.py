from __future__ import annotations

from modeling.config import (
    DistributedConfig,
    ExperimentConfig,
    ExperimentMetadata,
    RunConfig,
    WandbConfig,
)
from modeling.data.text_train import TextPretrainDatapackConfig

from modeling.modules.text_pretrain.llama3 import Llama3LITConfig

from modeling.types import Accelerator, DType

CR7BPretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="llama3_4B_text_pretrain"),
        output_dir="output/text_pretrain",
        checkpoint=None,
    ),
    module=Llama3LITConfig(
        model_name="CohereLabs/c4ai-command-r7b-12-2024",
        tokenizer_name="CohereLabs/c4ai-command-r7b-12-2024",
    ),
    datapack=TextPretrainDatapackConfig(),
    run=RunConfig(
        lr=1e-5,
        sequence_length=1024,
        batch_size=8,
        num_steps=5000,
        distributed=DistributedConfig(
            devices_per_node=8,
        ),
        # attn_impl=AttentionImplementation.FLASH_ATTENTION_2,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
    ),
)
CR7BPretrainTest = CR7BPretrainExperimentConfig.testing_config(
    num_steps=100, enable_wandb=True
)

# mdl export modeling.experiments.text_pretrain.cr7b.CR7BPretrainExperimentConfig
# mdl export modeling.experiments.text_pretrain.cr7b.CR7BPretrainTest
