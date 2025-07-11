from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from typing import Self

from pydantic import BaseModel

from synapse.utils.logging import configure_logging, logging

timing_logger = configure_logging(__name__, level=logging.DEBUG)


class TimingStats(BaseModel):
    total_duration: float
    first_enter: float
    last_exit: float

    def __str__(self) -> str:
        return (
            f"total_duration={self.total_duration:.6f}, "
            f"first_enter={self.first_enter:.6f}, "
            f"last_exit={self.last_exit:.6f})"
        )

    def merge(self, other: TimingStats) -> TimingStats:
        """Merge another TimingStats into this one"""
        return TimingStats(
            total_duration=self.total_duration + other.total_duration,
            first_enter=min(self.first_enter, other.first_enter),
            last_exit=max(self.last_exit, other.last_exit),
        )


def merge_timing_stats_dict(
    d1: dict[str, TimingStats],
    d2: dict[str, TimingStats],
) -> dict[str, TimingStats]:
    """Merge two dictionaries of TimingStats"""
    merged = defaultdict(
        lambda: TimingStats(
            total_duration=0.0, first_enter=float("inf"), last_exit=float("-inf")
        )
    )
    for key, stats in d1.items():
        merged[key] = merged[key].merge(stats)
    for key, stats in d2.items():
        merged[key] = merged[key].merge(stats)
    return dict(merged)


class TimingContext:
    def __init__(self, name: str):
        self.name = name
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.children: dict[str, list[TimingContext]] = defaultdict(list)

    def add_child(self, child: TimingContext):
        """Add a child context to this context. Create if it does not exist."""
        self.children[child.name].append(child)

    def __enter__(self) -> Self:
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        self.end_time = time.time()
        return False

    @property
    def fstart_time(self) -> float:
        """Get start time of this context"""
        assert self.start_time is not None, (
            "Context must be entered before accessing start time"
        )
        return self.start_time

    @property
    def fend_time(self) -> float:
        """Get end time of this context"""
        assert self.end_time is not None, (
            "Context must be exited before accessing end time"
        )
        return self.end_time

    @property
    def elapsed(self) -> float:
        """Calculate elapsed time for this context"""
        assert self.start_time is not None, (
            "Context must be entered before calculating elapsed time"
        )

        return (
            (self.end_time - self.start_time)
            if self.end_time
            else (time.time() - self.start_time)
        )

    def get_timing_tree(self) -> dict[str, TimingStats]:
        """Get timing stats for this context and its children"""
        stats = {
            self.name: TimingStats(
                total_duration=self.elapsed,
                first_enter=self.fstart_time,
                last_exit=self.fend_time,
            )
        }
        for child_name, child_contexts in self.children.items():
            # Aggregate stats for each child context
            # Perform reduce operation to merge all child contexts
            child_stats: dict[str, TimingStats] = {}
            for child_context in child_contexts:
                child_stats = merge_timing_stats_dict(
                    child_stats,
                    child_context.get_timing_tree(),
                )
            child_stats = {
                f"{self.name}.{child_name}": stats
                for child_name, stats in child_stats.items()
            }
            stats = merge_timing_stats_dict(stats, child_stats)

        return stats

    def print_timing_tree(self, logger: logging.LoggerAdapter | None = None) -> None:
        """Print the timing tree in a formatted way"""
        print_timing_tree(self.get_timing_tree(), logger=logger)


def print_timing_tree(
    timing_tree: dict[str, TimingStats], logger: logging.LoggerAdapter | None = None
) -> None:
    if logger is None:
        logger = timing_logger
    """Print the timing tree in a formatted way"""
    if not timing_tree:
        logger.info("No timing data recorded")
        return

    # Calculate column widths
    name_width = max(len("Name"), max(len(name) for name in timing_tree))
    duration_width = max(len("Duration"), 12)  # Accommodate decimal places
    first_width = max(len("First Enter"), 12)
    last_width = max(len("Last Exit"), 12)

    # Print header
    header = f"{'Name':<{name_width}} | {'Duration':<{duration_width}} | {'First Enter':<{first_width}} | {'Last Exit':<{last_width}}"
    logger.info(header)
    logger.info("-" * len(header))

    # Print data rows
    for name, stats in timing_tree.items():
        logger.info(
            f"{name:<{name_width}} | {stats.total_duration:<{duration_width}.6f} | {stats.first_enter:<{first_width}.6f} | {stats.last_exit:<{last_width}.6f}"
        )


class TimingProfiler:
    def __init__(self):
        self.context_stack: list[TimingContext] = []

    @contextmanager
    def __call__(self, name: str) -> Generator[TimingContext, None, None]:
        """Context manager for timing operations"""
        new_context = TimingContext(name)
        if self.context_stack:
            # Add to the parent context
            self.context_stack[-1].add_child(new_context)
        # Push new context onto the stack
        self.context_stack.append(new_context)
        new_context.__enter__()
        try:
            yield new_context
        finally:
            assert self.context_stack[-1] == new_context, (
                "Context stack is inconsistent, last context should match the new one"
            )
            # Pop from context stack
            new_context.__exit__(None, None, None)
            self.context_stack.pop()


# Global profiler instance
elapsed_timer = TimingProfiler()
