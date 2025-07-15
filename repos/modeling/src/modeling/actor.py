from modeling.utils.typed_remote import (
    remote_method,
    BaseActor,
)
from pydantic import BaseModel
from modeling.config.distributed import InstanceConfig


class ActorArgs(BaseModel):
    instance_config: InstanceConfig


class ExperimentActor(BaseActor[ActorArgs]):
    """
    An example actor that can be used to run experiments in a distributed manner.
    """

    @remote_method
    def health_check(self) -> None:
        """
        Run a health check on the experiment.
        """
        # Implement the logic to run the experiment here
        print(f"Running health check on experiment with {self.args=}")
