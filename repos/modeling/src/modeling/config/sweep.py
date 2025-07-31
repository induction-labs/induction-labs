from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Self, TypeVar

from . import ExperimentConfig, LinearLRSchedule

T = TypeVar("T")

A = TypeVar("A")
B = TypeVar("B")


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

        @staticmethod
        def batch_size(batch_size: int, exp: ExperimentConfig) -> ExperimentConfig:
            exp.run.batch_size = batch_size
            return exp

        @staticmethod
        def product(list_a: list[A], list_b: list[B]) -> list[tuple[A, B]]:
            """
            Returns the Cartesian product of list_a and list_b as a list of tuples.

            Example:
                >>> product([1, 2], ['a', 'b'])
                [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
            """
            return [(a, b) for a in list_a for b in list_b]

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
