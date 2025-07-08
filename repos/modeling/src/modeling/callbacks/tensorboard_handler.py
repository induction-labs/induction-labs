from __future__ import annotations
from typing import Optional
from synapse.utils.logging import configure_logging
import logging

from pathlib import Path

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
