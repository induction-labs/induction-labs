from __future__ import annotations
from typing import Any, Generator, Optional
from modeling.initialization import UnifiedExperimentConfig
from synapse.utils.logging import configure_logging
import logging
from contextlib import contextmanager

from pathlib import Path

from torch.profiler.profiler import profile

logger = configure_logging(__name__, level=logging.INFO)


def tensorboard_trace_handler(
    dir_name: Path, worker_name: Optional[str] = None, use_gzip: bool = False
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


@contextmanager
def profiler_context(
    config: UnifiedExperimentConfig,
) -> Generator[DummyProfiler | profile, Any, None]:  # object is torch.profiler.profile
    # TODO: Wrap profiler in a class instead of union
    """
    Context manager to handle profiling during the experiment.
    This is a placeholder for actual profiling logic.
    """
    from torch.profiler import profile, ProfilerActivity, schedule

    try:
        if config.run.profile is None:
            yield DummyProfiler()  # Replace with actual profiling logic
        else:
            profile_dir = config.metadata.output_dir / "profiler"
            profile_schedule = schedule(
                wait=config.run.profile.wait,
                warmup=config.run.profile.warmup,
                active=config.run.profile.active,
                repeat=config.run.profile.repeat,
            )
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=profile_schedule,
                on_trace_ready=tensorboard_trace_handler(
                    profile_dir,  # type: ignore[arg-type]
                ),
                record_shapes=True,
                with_stack=True,
                with_flops=True,
                profile_memory=True,
                with_modules=True,
            ) as prof:
                # prof.start()
                yield prof
                # prof.stop()
    except Exception as e:
        logger.error(f"Profiling error: {e}")
        raise e
    finally:
        pass
