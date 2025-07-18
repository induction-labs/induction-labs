from . import ExperimentConfig, LinearLRSchedule

from typing import Any, TypeVar, Sequence, Callable
from typing import Self
from dataclasses import dataclass, field


T = TypeVar("T")


@dataclass
class Sweep:
    experiment: ExperimentConfig
    _sweeps: list[
        tuple[Sequence[Any], Callable[[Any, ExperimentConfig], ExperimentConfig]]
    ] = field(default_factory=list)

    class S:
        @staticmethod
        def lr(lr: LinearLRSchedule, exp: ExperimentConfig) -> ExperimentConfig:
            exp.run.lr = lr
            return exp

        @staticmethod
        def seed(seed: int, exp: ExperimentConfig) -> ExperimentConfig:
            exp.run.seed = seed
            return exp

    def sweep(
        self, inputs: Sequence[T], f: Callable[[T, ExperimentConfig], ExperimentConfig]
    ) -> Self:
        self._sweeps.append((inputs, f))
        return self

    def collect(self) -> list[ExperimentConfig]:
        # start from the base experiment
        configs: list[ExperimentConfig] = [self.experiment]

        # for each sweep, apply it across all current configs
        for inputs, fn in self._sweeps:
            next_configs: list[ExperimentConfig] = []
            for cfg in configs:
                for inp in inputs:
                    next_configs.append(fn(inp, cfg.model_copy(deep=True)))
            configs = next_configs

        return configs
