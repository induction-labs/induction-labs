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
from modeling.types import AttentionImplementation

from modeling.types import Accelerator, DType

Llama3PretrainExperimentConfig = ExperimentConfig(
    metadata=ExperimentMetadata(
        wandb=WandbConfig(project="testing", name="llama3_4B_text_pretrain"),
        output_dir="output/text_pretrain",
        checkpoint=None,
    ),
    module=Llama3LITConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
    ),
    datapack=TextPretrainDatapackConfig(),
    run=RunConfig(
        lr=1e-5,
        sequence_length=1024,
        batch_size=8,
        steps_per_epoch=5000,
        distributed=DistributedConfig(
            devices_per_node=8,
        ),
        # attn_impl=AttentionImplementation.FLASH_ATTENTION_2,
        attn_impl=AttentionImplementation.SDPA,
        accelerator=Accelerator.CUDA,
        precision=DType.bf16,
    ),
)
Llama3PretrainTest = Llama3PretrainExperimentConfig.testing_config(
    num_steps=100, enable_wandb=True
)

# mdl export modeling.experiments.text_pretrain.llama3.Llama3PretrainExperimentConfig
# mdl export modeling.experiments.text_pretrain.llama3.Llama3PretrainTest
