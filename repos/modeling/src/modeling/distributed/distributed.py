from __future__ import annotations

import torch.distributed as dist
from synapse.utils.logging import configure_logging
from modeling.config import DistributedConfig, InstanceConfig
from modeling.config.distributed import MeshAxis
import logging
import torch
from contextlib import contextmanager
from typing import Iterator
from synapse.elapsed_timer import elapsed_timer

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
    global_rank = distributed_config.global_rank(instance_config)

    logger.info(
        f"Initializing distributed training on rank {distributed_config.global_rank(instance_config)}"
    )

    try:
        with elapsed_timer(
            f"Distributed training initialization on rank {distributed_config.global_rank(instance_config)}"
        ) as timer:
            # Use env:// for now
            torch.distributed.init_process_group(
                init_method="env://",
                backend="nccl",
                device_id=instance_config.device,
                # backend=distributed_config.backend,
                # init_method=distributed_config.init_method,
                # world_size=distributed_config.world_size,
                # rank=instance_config.device_rank,
            )
            logger.debug(f"Process group initialized on rank {dist.get_rank()}. ")

            # Validate the initialized process group
            assert dist.get_rank() == global_rank, (
                f"Expected rank {global_rank}, but got {dist.get_rank()}"
            )
            assert dist.get_world_size() == distributed_config.world_size, (
                f"Expected world size {distributed_config.world_size}, but got {dist.get_world_size()}"
            )

            # Initialize device mesh
            s = distributed_config.sharding
            mesh = torch.distributed.device_mesh.init_device_mesh(
                "cuda",
                (s.DP, s.FSDP, s.TP),
                mesh_dim_names=(MeshAxis.DP, MeshAxis.FSDP, MeshAxis.TP),
            )
            dist.barrier()

        logger.info(
            f"Distributed training initialized in {timer.elapsed:.2f}s: rank {dist.get_rank()}/{dist.get_world_size()}"
        )

        yield mesh

    except Exception as e:
        logger.critical(
            f"Failed to initialize distributed training on rank {global_rank}: {e}"
        )
        raise e
    finally:
        # Clean up the process group
        if dist.is_initialized():
            logger.debug(f"Destroying process group on rank {dist.get_rank()}")
            dist.destroy_process_group()
            logger.debug("Process group destroyed successfully")


@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()
