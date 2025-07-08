from modeling.config import InstanceConfig
import functools
import os


@functools.lru_cache(maxsize=1)
def get_rank():
    if os.environ.get("LOCAL_RANK") is not None:
        # If LOCAL_RANK is set, use it to determine the rank
        assert os.environ["LOCAL_RANK"].isdigit(), "LOCAL_RANK must be an integer"
        return int(os.environ["LOCAL_RANK"])
    return 0


def get_distributed_env() -> InstanceConfig:
    """
    Get the distributed environment configuration.

    Returns:
        DistributedInstanceConfig: The distributed configuration.
    """
    # TODO: Inititalize through ray

    return InstanceConfig(device_rank=get_rank(), node_rank=0)
