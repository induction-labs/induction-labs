from __future__ import annotations

import ast
import asyncio
import os
import signal
import subprocess
import traceback
from time import sleep
from typing import Annotated

import typer
from synapse.utils.async_typer import AsyncTyper
from synapse.utils.logging import configure_logging, logging

logger = configure_logging(__name__, level=logging.INFO)
app = AsyncTyper()


def cleanup_processes(processes: list[subprocess.Popen], timeout=5):
    """Clean up running processes with graceful shutdown."""
    if processes:
        logger.info(f"Shutting down {len(processes)} processes...")
        # Terminate all processes
        for index, process in enumerate(processes):
            if process.poll() is None:  # Process is still running
                logger.info(f"Terminating {process.pid=} {index=}")
                try:
                    process.terminate()
                except Exception as e:
                    logger.warning(f"Failed to terminate {process.pid=} {index=}: {e}")

        # Wait a bit for graceful shutdown

        for _ in range(timeout):
            if any(process.poll() is None for process in processes):
                logger.info("Waiting for processes to shut down gracefully...")
                sleep(1)
            else:
                logger.info("All processes shut down gracefully")
                return

        # Force kill if still running
        for index, process in enumerate(processes):
            if process.poll() is None:
                logger.warning(f"Force killing {process.pid=} {index=}")
                try:
                    process.kill()
                except Exception as e:
                    logger.warning(f"Failed to kill {process.pid=} {index=}: {e}")

        logger.info("All processes shut down")


@app.async_command(name="start")
async def run_processes(
    commands: Annotated[
        list[str], typer.Argument(help="Commands to run as subprocesses")
    ],
):
    """Run multiple commands as subprocesses with proper cleanup handling."""

    # Environment variables
    env = os.environ.copy()
    processes: list[subprocess.Popen] = []

    # Set up signal handler for graceful shutdown
    shutting_down = False

    def signal_handler(signum, _):
        nonlocal shutting_down
        if shutting_down:
            logger.info(f"\nReceived signal {signum} again, force exiting...")
            os._exit(1)
        logger.info(f"\nReceived signal {signum}, shutting down processes...")
        shutting_down = True
        cleanup_processes(processes, 30)
        os._exit(signum)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    exit_code = 1

    try:
        logger.info(f"Starting {len(commands)} processes...")

        # Start each command as a subprocess
        for i, command in enumerate(commands):
            logger.info(f"Starting process {i}: {command}")

            # Parse command string into list
            try:
                cmd = ast.literal_eval(command)
                if not isinstance(cmd, list):
                    raise ValueError("Command must be a list")
            except (ValueError, SyntaxError) as e:
                logger.error(f"Failed to parse command {i}: {command}. Error: {e}")
                continue

            try:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                )
                processes.append(process)
                logger.info(f"Started process {process.pid} for command {i}")

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to start process for command {i}: {e}")
                continue

        logger.info(f"Started {len(processes)} processes")

        # wait for any process to finish
        pid, status = os.wait()  # blocks until any child process ends
        first = next(p for p in processes if p.pid == pid)
        exit_code = os.WEXITSTATUS(status)
        print(f"Process {first.pid} finished first with exit code {exit_code}")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        cleanup_processes(processes, 30)
        # Restore default signal handlers to prevent hanging
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os._exit(exit_code)


if __name__ == "__main__":
    app()
