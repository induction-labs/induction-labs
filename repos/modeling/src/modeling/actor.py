from modeling.utils.typed_remote import (
    remote_method,
    actor_class,
    BaseActor,
    RemoteArgs,
)
from pydantic import BaseModel
from modeling.config.distributed import InstanceConfig


class ActorArgs(BaseModel):
    instance_config: InstanceConfig


class ActorState(BaseModel):
    """
    State of the actor, can be used to store any data that needs to be shared between methods.
    This is not the same as the actor's state, which is managed by Ray.
    """

    pass


@actor_class(RemoteArgs(num_gpus=1.0))
class ExperimentActor(BaseActor[ActorArgs, ActorState]):
    """
    An example actor that can be used to run experiments in a distributed manner.
    """

    @remote_method
    def health_check(self) -> None:
        """
        Run a health check on the experiment.
        """
        # Implement the logic to run the experiment here
        print(f"Running health check on experiment with f{self.args=}")

    async def _configure_state(self) -> ActorState:
        return ActorState()
