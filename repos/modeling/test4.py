import torch
import torch.nn.functional as F

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.demos import Transformer, WikiText2

fabric = L.Fabric(accelerator="cuda", devices=2, strategy=FSDPStrategy())
fabric.launch()

fabric.seed_everything(42)

with fabric.rank_zero_first():
    dataset = WikiText2()

# 1B parameters
model = Transformer(
    vocab_size=dataset.vocab_size, nlayers=32, nhid=4096, ninp=1024, nhead=64
)

model = fabric.setup(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
optimizer = fabric.setup_optimizers(optimizer)


for i in range(2):
    input, target = fabric.to_device(dataset[i])
    output = model(input.unsqueeze(0), target.unsqueeze(0))
    loss = F.nll_loss(output, target.view(-1))
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    fabric.print(loss.item())

fabric.print(torch.cuda.memory_summary())
