import os
import random

import numpy as np
import torch


def fix_rng(seed: int, device: torch.device | None = None) -> torch.Generator:
    os.environ["PYTHONHASHSEED"] = str(seed)  # make hash-based operations reproducible
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for reproducibility in cuBLAS

    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # CPU RNG
    torch.cuda.manual_seed(seed)  # current GPU RNG
    torch.cuda.manual_seed_all(seed)  # all-GPU RNGs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # # In PyTorch ≥1.8 you can also do:
    torch.use_deterministic_algorithms(True)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
