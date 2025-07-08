from __future__ import annotations

import torch.distributed as dist
from synapse.utils.logging import configure_logging
from modeling.config import DistributedConfig
import logging

logger = configure_logging(__name__, level=logging.INFO)


def init_distributed(config: DistributedConfig) -> None:
    """
    Initialize torch.distributed for distributed training.
    This should be called before any other distributed operations.

    Args:
        config: DistributedConfig containing distributed training settings
    """
    if dist.is_available() and not dist.is_initialized():
        # Initialize the process group
        # Uses environment variables for configuration:
        # RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
        dist.init_process_group(backend="nccl")
        logger.info(
            f"Initialized distributed training: rank {dist.get_rank()}/{dist.get_world_size()}"
        )
    elif dist.is_initialized():
        logger.info(
            f"Distributed already initialized: rank {dist.get_rank()}/{dist.get_world_size()}"
        )
    else:
        logger.info("Distributed training not available")


def is_distributed() -> bool:
    """
    Check if distributed training is initialized and available.

    Returns:
        True if distributed training is initialized, False otherwise
    """
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """
    Get the rank of the current process in distributed training.

    Returns:
        Rank of the current process, or 0 if not in distributed mode
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get the total number of processes in distributed training.

    Returns:
        Total number of processes, or 1 if not in distributed mode
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """
    Check if the current process is the main process (rank 0).

    Returns:
        True if this is the main process, False otherwise
    """
    return get_rank() == 0
