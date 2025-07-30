from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import torch
import torch.distributed as dist
from pydantic import AnyUrl, UrlConstraints
from synapse.elapsed_timer.elapsed_timer import elapsed_timer
from synapse.utils.logging import configure_logging

from modeling.config import DistributedConfig, InstanceConfig
from modeling.config.distributed import MeshAxis

logger = configure_logging(__name__, level=logging.INFO)


class TorchUrl(AnyUrl):
    """ """

    host_required = True

    @property
    def host(self) -> str:
        """The required URL host."""
        return cast(str, self._url.host)  # pyright: ignore[reportAttributeAccessIssue]

    _constraints = UrlConstraints(allowed_schemes=["tcp"])


@contextmanager
def init_distributed(
    distributed_config: DistributedConfig,
    instance_config: InstanceConfig,
    rank0_address: TorchUrl,
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

    global_rank = distributed_config.global_rank(instance_config)

    logger.info(
        f"Initializing distributed training on rank {global_rank=} {torch.cuda.device_count()=} {rank0_address=} "
    )

    logger.debug(
        f"Environment variables: {os.environ['LD_PRELOAD']=} {os.environ['LD_LIBRARY_PATH']=}"
    )
    try:
        torch.cuda.set_device(instance_config.device)
        with elapsed_timer(
            f"Distributed training initialization on rank {distributed_config.global_rank(instance_config)}"
        ) as timer:
            #! NOTE: Currently on B200s there is a bug where if you specify `device_id=torch.device("cuda:0")`
            #! Then we get the error
            #! ncclUnhandledCudaError: Call to CUDA function failed. Cuda failure 'PTX JIT compiler library not found'
            #! Bruh moment.
            torch.distributed.init_process_group(
                init_method=rank0_address.encoded_string(),
                # init_method="tcp://127.0.0.1:29500",
                backend="nccl",
                # device_id=instance_config.device,
                rank=global_rank,
                world_size=distributed_config.world_size,
            )
            logger.debug(
                f"Process group initialized on rank {dist.get_rank()}. tcp://127.0.0.1:29500 "
            )

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
