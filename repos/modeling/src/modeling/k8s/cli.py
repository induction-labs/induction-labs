from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Annotated

import typer
import yaml
from synapse.utils.logging import configure_logging

from modeling.utils.gen_id import gen_id

logger = configure_logging(
    __name__,
    level=logging.INFO,
)


# TODO: Add -i interactive option to run k8s interactively
def load_k8s_template() -> dict:
    import yaml

    assert K8S_TEMPLATE_PATH.exists(), f"Template file not found: {K8S_TEMPLATE_PATH}"

    with open(K8S_TEMPLATE_PATH) as f:
        job_template = yaml.safe_load(f)
    logger.debug(f"Loaded job template from: {K8S_TEMPLATE_PATH}")
    return job_template


k8s_app = typer.Typer()


@k8s_app.command()
def bake(
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Hide depot build output")
    ] = False,
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
            assert isinstance(image_ref, str), "Image reference should be a string"
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
        raise typer.Exit(1) from e
    except FileNotFoundError as e:
        logger.error("depot command not found. Please ensure depot CLI is installed")
        print("Error: depot command not found. Please ensure depot CLI is installed")
        raise typer.Exit(1) from e


K8S_TEMPLATE_PATH = (
    Path(__file__).parent.parent.parent.parent / "k8s" / "induction-labs" / "mdl.yaml"
)


@k8s_app.command()
def submit(
    config_path: Annotated[str, typer.Argument(help="File to submit")],
    image: Annotated[
        str | None,
        typer.Option(
            "--image",
            help="Docker image to use. If not provided, will run bake to build image",
        ),
    ] = None,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Hide output from the command")
    ] = False,
    context: Annotated[
        str | None,
        typer.Option(
            "--context",
            help="Kubernetes context to use. If not provided, uses current context",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Perform a dry run without actually submitting the job",
        ),
    ] = False,
):
    """
    Submit runs to kubernetes kueue.
    """
    # Get the image - either from parameter or by running bake
    if image is None:
        logger.info("No image provided, running bake to build image...")
        image = bake(quiet=quiet)

    else:
        logger.info(f"Using provided image: {image}")

    logger.info(f"Submitting job with config: {config_path} and image: {image}")

    from kubernetes import client

    from modeling.k8s.context import load_kubernetes_config

    load_kubernetes_config(context=context)

    # Load the YAML template
    job_template = load_k8s_template()

    # Load and validate the experiment config

    from modeling.config.serde import build_experiment_config

    config_path_obj = Path(config_path)
    assert config_path_obj.exists() and config_path_obj.is_file(), (
        f"{config_path} is not a file"
    )

    experiment_config = build_experiment_config(config_path_obj)
    logger.info(f"Loaded experiment config from: {config_path}")

    # Calculate resources based on experiment config
    distributed_config = experiment_config.run.distributed
    num_gpus = distributed_config.devices_per_node * distributed_config.num_nodes

    # Resource calculation: 16GB memory per GPU + 4 CPUs per GPU + 16 CPUs for head node
    memory_gb = (num_gpus * 16) + 16
    cpu_count = (num_gpus * 4) + 4

    logger.info(
        f"Calculated resources: {num_gpus} GPUs, {memory_gb}Gi memory, {cpu_count} CPUs"
    )

    # Modify the job template
    container = job_template["spec"]["template"]["spec"]["containers"][0]

    # Update image
    container["image"] = image
    logger.debug(f"Updated container image to: {image}")

    # Update resources
    container["resources"]["requests"]["nvidia.com/gpu"] = str(num_gpus)
    container["resources"]["requests"]["memory"] = f"{memory_gb}Gi"
    container["resources"]["requests"]["cpu"] = str(cpu_count)

    container["resources"]["limits"]["nvidia.com/gpu"] = str(num_gpus)
    container["resources"]["limits"]["memory"] = f"{memory_gb}Gi"
    container["resources"]["limits"]["cpu"] = str(cpu_count)

    logger.debug(
        f"Updated resources: {num_gpus} GPUs, {memory_gb}Gi memory, {cpu_count} CPUs"
    )

    # Update the command args to use the provided config_path
    container["args"] = [f"mdl run {config_path} -rhw"]
    logger.debug(f"Updated command args to use config: {config_path}")

    # Save the modified k8s yaml config beside the original config toml

    config_path_obj = Path(config_path)
    yaml_config_path = config_path_obj.with_suffix(f".{gen_id(6)}.yaml")

    try:
        with open(yaml_config_path, "w") as f:
            yaml.dump(job_template, f, default_flow_style=False)
        logger.info(f"Saved k8s config to: {yaml_config_path}")
    except Exception as e:
        logger.error(f"Failed to save k8s config: {e}")
        raise typer.Exit(1) from e

    # Create batch API client
    batch_v1 = client.BatchV1Api()

    # Get namespace from the job template
    namespace = job_template["metadata"]["namespace"]

    try:
        # Submit the job
        if dry_run:
            response = batch_v1.create_namespaced_job(
                namespace=namespace, body=job_template, dry_run="All"
            )
            logger.info("Dry run successful - job would be created")
        else:
            response = batch_v1.create_namespaced_job(
                namespace=namespace, body=job_template
            )
            job_name = response.metadata.name  # type: ignore[attr-defined]
            logger.info(f"Successfully submitted job: {job_name}")

    except Exception as e:
        logger.error(f"Failed to submit job to Kubernetes: {e}")
        raise typer.Exit(1) from e


def get_sweep_tomls(directory: str) -> list[Path]:
    """
    Get all .toml files in the specified directory.
    """
    config_dir = Path(directory)

    assert config_dir.exists(), f"{config_dir} does not exist"
    assert config_dir.is_dir(), f"{config_dir} is not a directory"

    # Find all .toml files in the directory
    config_files = list(config_dir.glob("*.toml"))

    assert len(config_files) > 0, f"No .toml files found in {config_dir}"
    return config_files


@k8s_app.command()
def sweep(
    directory: Annotated[
        str, typer.Argument(help="Directory containing experiment configuration files")
    ],
    image: Annotated[
        str | None,
        typer.Option(
            "--image",
            help="Docker image to use. If not provided, will run bake to build image",
        ),
    ] = None,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Hide output from the command")
    ] = False,
    context: Annotated[
        str | None,
        typer.Option(
            "--context",
            help="Kubernetes context to use. If not provided, uses current context",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Perform a dry run without actually submitting the job",
        ),
    ] = False,
):
    """
    Submit runs to kubernetes kueue.
    """
    # Get the image - either from parameter or by running bake

    if image is None:
        logger.info("No image provided, running bake to build image...")
        image = bake(quiet=quiet)

    else:
        logger.info(f"Using provided image: {image}")

    sweep_tomls = get_sweep_tomls(directory)
    logger.info(f"Submitting sweep tomls at {directory} and image: {image}")
    for config_path in sweep_tomls:
        logger.debug(f"Sweep Submitting config: {config_path}")
        try:
            submit(
                config_path=config_path.as_posix(),
                image=image,
                quiet=quiet,
                context=context,
                dry_run=dry_run,
            )
        except Exception as e:
            logger.error(f"Failed to submit config {config_path}: {e}")
            if not quiet:
                print(f"Error submitting {config_path}: {e}")
            continue
