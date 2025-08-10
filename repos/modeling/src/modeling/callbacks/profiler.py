from __future__ import annotations

import logging
from pathlib import Path

from synapse.utils.logging import configure_logging

from modeling.config import UnifiedExperimentConfig

logger = configure_logging(__name__, level=logging.INFO)


def tensorboard_trace_handler(
    dir_name: Path,
    worker_name: str | None = None,
    use_gzip: bool = False,
):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import os
    import socket
    import time

    def handler_fn(prof) -> None:
        nonlocal worker_name
        print(f"Exporting profiling trace to {dir_name}...")
        dir_name.mkdir(parents=True, exist_ok=True)
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        output_path = dir_name / file_name
        prof.export_chrome_trace(str(output_path))
        logger.info(f"Exported profiling trace to {output_path}")

    return handler_fn


class DummyProfiler:
    def step(self):
        pass


class ProfileWrapper:
    """
    A wrapper for the profiler to handle the context management and profiling actions.
    Using this means handling profile cleanup manually.
    """

    def __init__(self, config: UnifiedExperimentConfig):
        self.config = config
        if self.config.run.profile is None:
            self.profiler = DummyProfiler()
        else:
            from torch.profiler import ProfilerActivity, profile, schedule

            profile_dir = self.config.metadata.output_dir / "profiler"
            logger.debug(f"Profiler config: {self.config.run.profile}")
            logger.info("Starting profiler...")
            profile_schedule = schedule(
                wait=self.config.run.profile.wait,
                warmup=self.config.run.profile.warmup,
                active=self.config.run.profile.active,
                repeat=self.config.run.profile.repeat,
            )
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=profile_schedule,
                on_trace_ready=tensorboard_trace_handler(
                    profile_dir,
                ),
                record_shapes=True,
                with_stack=True,
                with_flops=True,
                profile_memory=True,
                with_modules=True,
            )
            self.profiler.start()

    def step(self):
        assert self.profiler is not None, "Profiler not started."
        if isinstance(self.profiler, DummyProfiler):
            return
        self.profiler.step()

    def stop(self):
        assert self.profiler is not None, "Profiler not started."
        if isinstance(self.profiler, DummyProfiler):
            return
        self.profiler.stop()
        self.profiler = None
