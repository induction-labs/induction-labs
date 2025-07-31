from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Annotated

import typer
from synapse.elapsed_timer import elapsed_timer
from synapse.utils.async_typer import AsyncTyper
from synapse.utils.logging import configure_logging, logging

from modeling.checkpoints.load import download_cloud_dir
from modeling.eval.vllm_utils import wait_for_servers_ready
from modeling.utils.cloud_path import CloudPath
from modeling.utils.max_timeout import max_timeout

logger = configure_logging(__name__, level=logging.INFO)
app = AsyncTyper()


def cleanup_process_and_tmp(
    processes: list[subprocess.Popen], model_tmpdir: Path | None
):
    if processes:
        logger.info(f"Shutting down {len(processes)} servers...")
        # Terminate all processes
        for i, process in enumerate(processes):
            if process.poll() is None:  # Process is still running
                logger.info(
                    f"Terminating process {process.pid} (GPU {i if i < len(processes) - 1 else 'load balancer'})"
                )
                try:
                    process.terminate()
                except Exception as e:
                    logger.warning(f"Failed to terminate process {process.pid}: {e}")

        # Wait a bit for graceful shutdown
        try:
            sleep(5)  # Wait for 5 seconds
        except TimeoutError:
            logger.warning("Timeout during graceful shutdown wait")

        # Force kill if still running
        for i, process in enumerate(processes):
            if process.poll() is None:
                logger.warning(
                    f"Force killing process {process.pid} (GPU {i if i < len(processes) - 1 else 'load balancer'})"
                )
                try:
                    process.kill()
                except Exception as e:
                    logger.warning(f"Failed to kill process {process.pid}: {e}")

        logger.info("All servers shut down")
        # Clean up model tmpdir if it was created
    if model_tmpdir and model_tmpdir.exists():
        logger.info(f"Cleaning up model tmpdir: {model_tmpdir}")
        try:
            shutil.rmtree(model_tmpdir)
            logger.info("Model tmpdir cleaned up successfully")
        except Exception as e:
            logger.warning(f"Failed to clean up model tmpdir {model_tmpdir}: {e}")


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
    load_balancer_port: Annotated[
        int, typer.Option("--load-balancer-port", "-lb", help="Port for load balancer")
    ] = 8080,
):
    """Start multiple vLLM servers using subprocess instead of tmux."""

    # Create timestamped log directory in tmpdir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(tempfile.gettempdir()) / "vllm_logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logs will be saved to: {log_dir}")

    # Environment variables needed by vLLM / CUDA
    env = os.environ.copy()

    processes: list[subprocess.Popen] = []
    model_tmpdir: Path | None = None

    # Set up signal handler for graceful shutdown
    shutdown_event = asyncio.Event()
    shutting_down = False

    def signal_handler(signum, _):
        nonlocal shutting_down
        if shutting_down:
            logger.info(f"\nReceived signal {signum} again, force exiting...")
            os._exit(1)
        logger.info(f"\nReceived signal {signum}, shutting down servers...")
        shutting_down = True
        cleanup_process_and_tmp(processes, model_tmpdir)
        os._exit(signum)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Handle model download if it's a gs:// path
        if model.startswith("gs://"):
            model_tmpdir = Path(tempfile.mkdtemp(prefix="vllm_model_"))
            logger.info(f"Model is a GCS path, downloading to {model_tmpdir=}...")
            cloud_path = CloudPath.from_str(model)

            with elapsed_timer("model_download") as download_timer:
                download_cloud_dir(cloud_path, model_tmpdir)

            logger.info(
                f"Model downloaded to {model_tmpdir} in {download_timer.elapsed:.2f} seconds"
            )

            # Copy config files from eval/uitars_config to the model directory
            config_dir = Path(__file__).parent / "uitars_config"
            if config_dir.exists():
                logger.info(f"Copying config files from {config_dir} to {model_tmpdir}")
                for config_file in config_dir.iterdir():
                    if config_file.is_file():
                        dest_file = model_tmpdir / config_file.name
                        shutil.copy2(config_file, dest_file)
                        logger.debug(f"Copied {config_file.name}")
                logger.info("Config files copied successfully")
            else:
                logger.warning(f"Config directory {config_dir} not found")

            # Use the local path for vLLM
            model = str(model_tmpdir)

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
        backend_urls = [
            f"http://localhost:{base_port + i}" for i in range(len(processes))
        ]
        with elapsed_timer("vllm_servers_startup") as startup_timer:
            await max_timeout(
                wait_for_servers_ready(backend_urls),
                timedelta(minutes=5),
                "Timeout waiting for vLLM servers to be ready",
            )

        logger.info(
            f"All {len(processes)} vLLM servers are ready! Startup took {startup_timer.elapsed:.2f} seconds"
        )

        # Start the load balancer FastAPI server
        logger.info(f"Starting load balancer on port {load_balancer_port}")
        lb_stdout_log = log_dir / "load_balancer_stdout.log"
        lb_stderr_log = log_dir / "load_balancer_stderr.log"

        try:
            with (
                open(lb_stdout_log, "w") as lb_stdout_file,
                open(lb_stderr_log, "w") as lb_stderr_file,
            ):
                # Generate backend URLs for the load balancer

                lb_cmd = [
                    "eve",
                    "lb",
                    "serve",
                    "--port",
                    str(load_balancer_port),
                    "--default-model",
                    model,
                ]
                # Add each backend URL as a separate --backend argument
                for url in backend_urls:
                    lb_cmd.extend(["--backend", url])

                lb_process = subprocess.Popen(
                    lb_cmd,
                    env=env,
                    stdout=lb_stdout_file,
                    stderr=lb_stderr_file,
                    text=True,
                )
                processes.append(lb_process)
                logger.info(f"Started load balancer process {lb_process.pid}")
                logger.info(f"  stdout: {lb_stdout_log}")
                logger.info(f"  stderr: {lb_stderr_log}")

                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to start load balancer: {e}")

        # Wait for load balancer to be ready
        lb_url = f"http://127.0.0.1:{load_balancer_port}"
        logger.info(f"Waiting for load balancer at {lb_url} to be ready...")

        with elapsed_timer("load_balancer_startup") as lb_timer:
            await max_timeout(
                wait_for_servers_ready([lb_url]),
                timedelta(minutes=2),
                "Timeout waiting for load balancer to be ready",
            )

        logger.info(
            f"Load balancer is ready! Setup took {lb_timer.elapsed:.2f} seconds"
        )

        logger.info("To test the setup, you can run:")
        logger.info(test_command)
        logger.info("------------------------------------\n")

        # Wait for shutdown signal
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        cleanup_process_and_tmp(processes, model_tmpdir)
        # Restore default signal handlers to prevent hanging
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


# eve vllm start --help
# eve vllm start "gs://induction-labs/checkpoints/uitars_sft_7b_yehaw_good_nice/2025-07-31T05-18-29.nTtAsFOt/step_-1" --num-gpus 1

if __name__ == "__main__":
    app()

test_command = """
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {
          "role": "user", 
          "content": [
            {
              "type": "text",
              "text": "Hello, how are you today?"
            }
          ]
        }
      ],
      "max_tokens": 100,
      "temperature": 0.7
    }' | jq
"""
