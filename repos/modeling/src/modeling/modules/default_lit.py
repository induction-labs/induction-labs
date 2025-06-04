from __future__ import annotations

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from modeling.data.text_train import LoadDataArgs, load_and_preprocess_data
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.modeling_utils import PreTrainedModel


class LitAutoModel(L.LightningModule):
    def __init__(self, model_name: str = "openai-community/gpt2"):
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
        # assert isinstance(outputs, )
        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, fused=True)
        return optimizer


def main():
    # Initialize the model
    wandb_logger = WandbLogger(project="train-test")

    model = LitAutoModel()

    # Load and preprocess the dataset
    args = LoadDataArgs(
        dataset_name="tatsu-lab/alpaca",
        seq_length=1024,
        tokenizer=model.tokenizer,
        batch_size=1,
    )
    dataset = load_and_preprocess_data(args)

    # Create a PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        precision="bf16-true" if torch.cuda.is_bf16_supported() else "16-mixed",
    )
    # Train the model
    trainer.fit(
        model,
        train_dataloaders=dataset,
    )


# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb
# wandb_logger.experiment.config.update({key1: val1, key2: val2})

if __name__ == "__main__":
    main()
