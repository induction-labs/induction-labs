from __future__ import annotations

import torch.distributed as dist
from synapse.utils.logging import configure_logging
from modeling.config import DistributedConfig, InstanceConfig
from modeling.config.distributed import MeshAxis
import logging
import torch
from contextlib import contextmanager
from typing import Iterator

logger = configure_logging(__name__, level=logging.INFO)


@contextmanager
def init_distributed(
    distributed_config: DistributedConfig, instance_config: InstanceConfig
) -> Iterator[torch.distributed.device_mesh.DeviceMesh]:
    """
    Initialize torch.distributed for distributed training as a context manager.
    This should be called before any other distributed operations.
    Properly cleans up the process group on exit.

    Args:
        distributed_config: DistributedConfig containing distributed training settings
        instance_config: InstanceConfig containing instance-specific settings

    Yields:
        DeviceMesh: The initialized device mesh for distributed training
    """
    assert dist.is_available(), (
        "Distributed training is not available in this PyTorch installation"
    )
    assert not dist.is_initialized(), "Distributed training is already initialized"

    torch.cuda.set_device(instance_config.device_rank)

    logger.info(
        f"Initializing distributed training on rank {distributed_config.global_rank(instance_config)}"
    )

    try:
        # Use env:// for now
        torch.distributed.init_process_group(
            # backend=distributed_config.backend,
            # init_method=distributed_config.init_method,
            # world_size=distributed_config.world_size,
            # rank=instance_config.device_rank,
        )

        # Validate the initialized process group
        assert dist.get_rank() == distributed_config.global_rank(instance_config), (
            f"Expected rank {distributed_config.global_rank(instance_config)}, but got {dist.get_rank()}"
        )
        assert dist.get_world_size() == distributed_config.world_size, (
            f"Expected world size {distributed_config.world_size}, but got {dist.get_world_size()}"
        )
        assert (
            local_rank := torch.distributed.distributed_c10d.get_node_local_rank()
        ) == instance_config.node_rank, (
            f"Expected node rank {instance_config.node_rank}, but got {local_rank}"
        )

        # Initialize device mesh
        s = distributed_config.sharding
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (s.DP, s.FSDP, s.TP),
            mesh_dim_names=(MeshAxis.DP, MeshAxis.FSDP, MeshAxis.TP),
        )

        logger.info(
            f"Distributed training initialized successfully: rank {dist.get_rank()}/{dist.get_world_size()}"
        )

        yield mesh

    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        raise
    finally:
        # Clean up the process group
        if dist.is_initialized():
            logger.debug(f"Destroying process group on rank {dist.get_rank()}")
            dist.destroy_process_group()
            logger.debug("Process group destroyed successfully")


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
