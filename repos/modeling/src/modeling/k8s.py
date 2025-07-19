from __future__ import annotations

import logging
import re
import subprocess

import typer
from synapse.utils.logging import configure_logging

logger = configure_logging(
    __name__,
    level=logging.DEBUG,
)

k8s_app = typer.Typer()


@k8s_app.command()
def bake(
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Hide depot build output"),
):
    """
    Build Docker image using depot and extract image reference.
    """
    try:
        logger.info("Starting depot bake process...")

        # Run depot bake --save command
        process = subprocess.Popen(
            ["depot", "bake", "--save"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        output_lines = []
        assert process.stdout is not None, "Process stdout should not be None"
        for line in process.stdout:
            if not quiet:
                print(line, end="")  # Print in real-time
            output_lines.append(line)

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, ["depot", "bake", "--save"]
            )

        output = "".join(output_lines)

        # Look for image reference pattern (e.g., registry.depot.dev/v2tbx2d1w1:snsd7rnnn4-remote)
        image_pattern = r"registry\.depot\.dev/[a-zA-Z0-9]+:[a-zA-Z0-9\-]+"
        matches = re.findall(image_pattern, output)

        if matches:
            image_ref = matches[-1]  # Take the last match (most recent)
            logger.info(f"Successfully built image: {image_ref}")
            print(f"Built image: {image_ref}")
            return image_ref
        else:
            logger.error("Could not extract image reference from depot output")
            print("Error: Could not extract image reference from depot output")
            raise typer.Exit(1)

    except subprocess.CalledProcessError as e:
        logger.error(f"Depot bake failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        print(f"Error: Depot bake failed - {e.stderr}")
        raise typer.Exit(1)
    except FileNotFoundError:
        logger.error("depot command not found. Please ensure depot CLI is installed")
        print("Error: depot command not found. Please ensure depot CLI is installed")
        raise typer.Exit(1)


@k8s_app.command()
def submit(
    config_path: str = typer.Argument(..., help="File to submit"),
):
    """
    Submit runs to kubernetes kueue.
    """
