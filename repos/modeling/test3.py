import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial

# 1. Import a suiting wrapping policy from PyTorch
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# 2. Configure the policy
# policy = partial(size_based_auto_wrap_policy, min_num_params=10000)
from torch.distributed.fsdp.api import ShardingStrategy

import lightning as L
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.demos import Transformer, WikiText2


class LanguageModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(  # 1B parameters
            vocab_size=vocab_size,
            nlayers=32,
            nhid=4096,
            ninp=1024,
            nhead=64,
        )

    def training_step(self, batch):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)


L.seed_everything(42)

# Data
dataset = WikiText2()
train_dataloader = DataLoader(dataset)

# Model
model = LanguageModel(vocab_size=dataset.vocab_size)


my_auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=20_000_000)
strat = FSDPStrategy(
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # or GRAD_SHARD or NO_SHARD
    auto_wrap_policy=my_auto_wrap_policy,
)

# Trainer
trainer = L.Trainer(accelerator="cuda", devices=8, limit_train_batches=2, max_epochs=1)
trainer.fit(model, train_dataloader)
trainer.print(torch.cuda.memory_summary())
