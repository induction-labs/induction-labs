from __future__ import annotations

# import tempfile
# from pathlib import Path
from typing import Any

import lightning as L
import torch

# from lightning.pytorch.loggers import WandbLogger
from modeling.config import DatapackConfig
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.text_module import TextLITConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.modeling_utils import PreTrainedModel


class TextPretrainLIT(L.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, use_cache=False)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=self._dtype)
        print(f"{self._dtype=}")
        assert isinstance(model, PreTrainedModel)
        self.model = model.train()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer

    def training_step(self, inputs):
        # Forward pass through the model
        outputs = self.model.forward(**inputs)
        assert isinstance(outputs.loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )
        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, fused=True)
        return optimizer


class TextPretrainLITConfig(TextLITConfig[TextPretrainLIT]):
    config_path: str = "modeling.modules.text_pretrain.TextPretrainLITConfig"
    model_name: str = "openai-community/gpt2"
    tokenizer_name: str = "openai-community/gpt2"

    @property
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer_name)

    def validate_datapack_compatibility(
        self, datapack_config: DatapackConfig[Any]
    ) -> TextPretrainDatapackConfig:
        assert isinstance(datapack_config, TextPretrainDatapackConfig), (
            f"Expected {datapack_config=} to be of type TextPretrainDatapackConfig"
        )
        return datapack_config

    def create_module(self) -> TextPretrainLIT:
        return TextPretrainLIT(model_name=self.model_name)


# def main():
#     # Initialize the model
#     wandb_logger = WandbLogger(project="train-test")

#     model = LitAutoModel()

#     # Load and preprocess the dataset
#     args = LoadDataArgs(
#         dataset_name="tatsu-lab/alpaca",
#         seq_length=1024,
#         tokenizer=model.tokenizer,
#         batch_size=1,
#     )
#     with tempfile.TemporaryDirectory() as tmpdir:
#         datamodule = TextPretrainDataModule(args, tmpdir=Path(tmpdir))
#         # Create a PyTorch Lightning Trainer
#         trainer = L.Trainer(
#             max_epochs=1,
#             accelerator="auto",
#             devices=1,
#             logger=wandb_logger,
#             precision="bf16-true" if torch.cuda.is_bf16_supported() else "16-mixed",
#         )
#         # Train the model
#         trainer.fit(
#             model,
#             datamodule=datamodule,
#         )


# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb
# wandb_logger.experiment.config.update({key1: val1, key2: val2})

# if __name__ == "__main__":
#     main()
