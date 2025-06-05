from __future__ import annotations

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from modeling.data.text_train import LoadDataArgs, load_and_preprocess_data
from transformers import Qwen2TokenizerFast
from transformers.models.qwen2_5_omni import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
)


class LitQwen25O(L.LightningModule):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-3B"):
        super().__init__()
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        ).train()
        processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        assert isinstance(processor, Qwen2_5OmniProcessor)

        tokenizer = processor.tokenizer
        assert isinstance(tokenizer, Qwen2TokenizerFast)
        self.tokenizer = tokenizer

    def training_step(self, inputs) -> torch.FloatTensor | None:
        # Forward pass through the model
        outputs = self.model.forward(**inputs)
        assert isinstance(outputs, Qwen2_5OmniThinkerCausalLMOutputWithPast)
        self.log(
            "train_loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, fused=True)
        return optimizer


def main():
    # Initialize the model
    wandb_logger = WandbLogger(project="train-test")

    model = LitQwen25O()

    # Load and preprocess the dataset
    args = LoadDataArgs(
        dataset_name="tatsu-lab/alpaca",
        seq_length=1024,
        tokenizer=model.tokenizer,
        batch_size=8,
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
