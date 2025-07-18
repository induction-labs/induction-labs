from __future__ import annotations

import subprocess
import typer
from pathlib import Path
from synapse.utils.logging import configure_logging
import logging
from tqdm import tqdm
from datetime import datetime
import signal
import sys

logger = configure_logging(
    __name__,
    level=logging.DEBUG,
)

queue_app = typer.Typer()


@queue_app.command()
def run(
    directory: str = typer.Argument(
        ..., help="Directory containing experiment configuration files"
    ),
):
    """
    Run queued experiments from a directory.

    Args:
        directory (str): Path to directory containing .toml config files
    """
    config_dir = Path(directory)

    if not config_dir.exists():
        logger.error(f"Directory {config_dir} does not exist")
        raise typer.Exit(1)

    if not config_dir.is_dir():
        logger.error(f"{config_dir} is not a directory")
        raise typer.Exit(1)

    # Find all .toml files in the directory
    config_files = list(config_dir.glob("*.toml"))

    if not config_files:
        logger.error(f"No .toml files found in {config_dir}")
        raise typer.Exit(1)

    # Sort files for consistent ordering
    config_files.sort()

    logger.info(f"Found {len(config_files)} configuration files to run")

    # Import here to avoid circular import
    from modeling.config.serde import build_experiment_config

    # Create timestamp for this queue run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Variable to track current process for signal handling
    current_process = None

    def signal_handler(signum, frame):
        """Handle Ctrl-C by terminating the current subprocess"""
        if current_process:
            logger.info("Received interrupt signal, terminating current experiment...")
            current_process.terminate()
            # Give it a moment to terminate gracefully
            try:
                current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.info("Process didn't terminate gracefully, killing it...")
                current_process.kill()
        sys.exit(1)

    # Set up signal handler for Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)

    for config_file in tqdm(config_files, desc="Running experiments"):
        log_file = None  # Initialize to fix unbound variable issue

        try:
            # Load experiment config to get output_dir
            experiment_config = build_experiment_config(Path(config_file))
            base_output_dir = experiment_config.metadata.output_dir

            # Create log directory: output_dir / timestamp
            log_dir = base_output_dir / timestamp
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create single log file path
            log_file = log_dir / f"{config_file.stem}.log"

            tqdm.write(f"Running {config_file.name} - Log: {log_file}")

            # Run mdl run <config_file> -rhw with single log file for both stdout and stderr
            with open(log_file, "w") as log_file_handle:
                current_process = subprocess.Popen(
                    ["mdl", "run", str(config_file), "-rhw"],
                    stdout=log_file_handle,
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                    text=True,
                )

                # Wait for process to complete
                return_code = current_process.wait()
                current_process = None

                if return_code != 0:
                    raise subprocess.CalledProcessError(
                        return_code, ["mdl", "run", str(config_file), "-rhw"]
                    )

            tqdm.write(f"✅ Successfully completed {config_file.name}")

        except subprocess.CalledProcessError as e:
            tqdm.write(f"❌ Failed to run {config_file.name}: {e}")
            tqdm.write(f"Return code: {e.returncode}")
            if log_file:
                tqdm.write(f"Check {log_file} for error details")
            # Continue with next config instead of stopping
            continue
        except Exception as e:
            tqdm.write(f"❌ Unexpected error running {config_file.name}: {e}")
            continue
        finally:
            current_process = None

    logger.info("Queue processing complete")
