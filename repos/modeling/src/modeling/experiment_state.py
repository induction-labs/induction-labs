from __future__ import annotations

from wandb.sdk.wandb_run import Run

from pydantic import BaseModel, ConfigDict


class ExperimentState(BaseModel):
    """
    A class to represent the mutable state of an experiment instance.
    It contains properties that can change during the experiment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    wandb: Run | None = None
    global_step: int = 0
