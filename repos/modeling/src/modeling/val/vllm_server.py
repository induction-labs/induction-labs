from __future__ import annotations

import asyncio
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from synapse.utils.async_typer import AsyncTyper
from synapse.utils.logging import configure_logging, logging

logger = configure_logging(__name__, level=logging.INFO)
app = AsyncTyper()


@app.async_command(name="start")
async def start_vllm_servers(
    model: Annotated[
        str, typer.Argument(help="Model to serve with vLLM")
    ] = "ByteDance-Seed/UI-TARS-1.5-7B",
    base_port: Annotated[
        int, typer.Option("--base-port", "-p", help="Starting port number")
    ] = 8000,
    num_gpus: Annotated[
        int, typer.Option("--num-gpus", "-n", help="Number of GPUs to use")
    ] = 8,
):
    """Start multiple vLLM servers using subprocess instead of tmux."""

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent / "logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logs will be saved to: {log_dir}")

    # Environment variables needed by vLLM / CUDA
    env = os.environ.copy()
    # env["NIX_LDFLAGS"] = "-L/usr/lib/x86_64-linux-gnu"
    # env["NIX_CFLAGS_COMPILE"] = "-I/usr/local/cuda/include"

    processes: list[subprocess.Popen] = []

    logger.info(f"Starting {num_gpus} vLLM servers...")

    # Start vLLM servers for each GPU
    for gpu in range(num_gpus):
        port = base_port + gpu

        # Set CUDA_VISIBLE_DEVICES for this specific process
        process_env = env.copy()
        process_env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            "vllm",
            "serve",
            model,
            "--port",
            str(port),
            "--enable-prefix-caching",
            "--tensor-parallel-size",
            "1",
        ]

        logger.info(f"Starting vLLM server on GPU {gpu}, port {port}")

        # Create log files for this GPU
        stdout_log = log_dir / f"gpu_{gpu}_stdout.log"
        stderr_log = log_dir / f"gpu_{gpu}_stderr.log"

        try:
            with (
                open(stdout_log, "w") as stdout_file,
                open(stderr_log, "w") as stderr_file,
            ):
                process = subprocess.Popen(
                    cmd,
                    env=process_env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                )
                processes.append(process)
                logger.info(f"Started process {process.pid} for GPU {gpu}")
                logger.info(f"  stdout: {stdout_log}")
                logger.info(f"  stderr: {stderr_log}")

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to start vLLM server for GPU {gpu}: {e}")
            continue

    logger.info(f"Started {len(processes)} vLLM servers")
    logger.info(
        f"Servers running on ports: {[base_port + i for i in range(len(processes))]}"
    )

    try:
        # Wait for all processes to complete
        await asyncio.gather(
            *[asyncio.to_thread(process.wait) for process in processes]
        )

    except KeyboardInterrupt:
        logger.info("\nShutting down servers...")
        # Terminate all processes
        for i, process in enumerate(processes):
            if process.poll() is None:  # Process is still running
                logger.info(f"Terminating process {process.pid} (GPU {i})")
                process.terminate()

        # Wait a bit for graceful shutdown
        await asyncio.sleep(2)

        # Force kill if still running
        for i, process in enumerate(processes):
            if process.poll() is None:
                logger.warning(f"Force killing process {process.pid} (GPU {i})")
                process.kill()

        logger.info("All servers shut down")


# uv run python -m modeling.val.vllm_server start --help
if __name__ == "__main__":
    app()
