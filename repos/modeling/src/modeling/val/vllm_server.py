from __future__ import annotations

import asyncio
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from synapse.elapsed_timer import elapsed_timer
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

    logger.info(
        f"Starting vllm {model=} on {num_gpus=} GPUs, starting at port {base_port}"
    )

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
        f"Servers running on ports: {[base_port + i for i in range(len(processes))]}, waiting for them to be ready..."
    )

    # Wait for all servers to be ready
    with elapsed_timer("vllm_servers_startup") as startup_timer:
        await wait_for_servers_ready(processes, base_port, num_gpus)

    logger.info(
        f"All {len(processes)} vLLM servers are ready! Startup took {startup_timer.elapsed:.2f} seconds"
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


async def wait_for_servers_ready(
    processes: list[subprocess.Popen], base_port: int, num_gpus: int
):
    """Wait for all vLLM servers to be ready by polling their health endpoints."""
    import aiohttp

    async def check_server_ready(port: int, gpu: int) -> bool:
        """Check if a single vLLM server is ready."""
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"http://localhost:{port}/health",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as response,
            ):
                if response.status == 200:
                    logger.info(f"GPU {gpu} server (port {port}) is ready")
                    return True
        except Exception:
            # Server not ready yet, continue polling
            pass
        return False

    ready_servers = set()
    max_attempts = 300  # 5 minutes with 1 second intervals
    attempt = 0

    while len(ready_servers) < len(processes) and attempt < max_attempts:
        attempt += 1

        # Check all servers that aren't ready yet
        for i, process in enumerate(processes):
            if i in ready_servers:
                continue

            # Check if process is still running
            if process.poll() is not None:
                logger.error(
                    f"GPU {i} process (PID {process.pid}) has terminated unexpectedly"
                )
                continue

            port = base_port + i
            if await check_server_ready(port, i):
                ready_servers.add(i)

        if len(ready_servers) < len(processes):
            await asyncio.sleep(1)

    if len(ready_servers) < len(processes):
        not_ready = set(range(len(processes))) - ready_servers
        logger.warning(
            f"Timeout waiting for servers on GPUs {list(not_ready)} to be ready"
        )

    return len(ready_servers) == len(processes)


# uv run python -m modeling.val.vllm_server start --help
if __name__ == "__main__":
    app()
