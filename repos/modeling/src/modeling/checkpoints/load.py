# 1) Install dependencies
# pip install google-cloud-storage transformers safetensors

from pathlib import Path

from huggingface_hub import snapshot_download
from synapse.utils.logging import configure_logging, logging

from modeling.utils.cloud_path import CloudPath

logger = configure_logging(__name__, level=logging.INFO)


def download_model_checkpoint(
    local_dir: Path, model_name: str, cloud_path: CloudPath | None
) -> None:
    if cloud_path:
        logger.info(f"Downloading model {model_name} from {cloud_path} to {local_dir}")
        return download_cloud_dir(cloud_path, local_dir)
    else:
        logger.info(f"Downloading model {model_name} to {local_dir}")
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
        )


def download_gcs_folder(bucket_name: str, prefix: str, local_dir: Path) -> None:
    """
    Download a folder from Google Cloud Storage using gcloud storage cp.

    Args:
        bucket_name: GCS bucket name
        prefix: Path prefix within the bucket
        local_dir: Local destination directory
    """
    logger.debug(
        f"Downloading from GCS bucket '{bucket_name}' with prefix '{prefix}' to {local_dir}"
    )

    # Create the local directory if it doesn't exist
    local_dir.mkdir(parents=True, exist_ok=True)

    # Construct the GCS source path
    if prefix:
        source = f"gs://{bucket_name}/{prefix.rstrip('/')}/*"
    else:
        source = f"gs://{bucket_name}/*"

    dest = str(local_dir).rstrip("/") + "/"

    command = [
        "gcloud",
        "storage",
        "cp",
        "-r",
        source,
        dest,
    ]

    logger.info("Downloading %s → %s", source, dest)
    logger.info("Running command: %s", " ".join(command))

    import subprocess

    try:
        result = subprocess.run(
            command,
            capture_output=True,  # grab both streams
            text=True,  # str not bytes
            check=True,  # raise if non-zero
        )

        # Success ─ log everything that gcloud printed
        if result.stdout:
            logger.debug(result.stdout.rstrip())
        if result.stderr:  # gcloud progress / warnings
            logger.debug(result.stderr.rstrip())

    except subprocess.CalledProcessError as e:
        # Command failed ─ dump its output *before* re-raising
        logger.error("gcloud exited with status %s", e.returncode)

        # e.stdout / e.stderr contain the captured streams
        if e.stdout:
            logger.error("stdout:\n%s", e.stdout.rstrip())
        if e.stderr:
            logger.error("stderr:\n%s", e.stderr.rstrip())

        raise  # keep the original stack trace

    logger.info(f"Downloaded files from gs://{bucket_name}/{prefix} to {local_dir}")


def download_cloud_dir(cloud_path: CloudPath, local_dir: Path) -> None:
    """
    Download a directory from cloud storage to local filesystem.
    Currently supports GCP (gs://) only.
    """
    if cloud_path.cloud != CloudPath.Cloud.GS:
        raise NotImplementedError(
            f"Cloud storage {cloud_path.cloud} not supported. Only GCP (gs://) is implemented."
        )

    # Parse GCP path: gs://bucket/path/to/dir
    path_parts = cloud_path.path.parts
    if not path_parts:
        raise ValueError(f"Invalid GCP path: {cloud_path}")

    bucket_name = path_parts[0]
    # Remove bucket name from path to get the prefix
    prefix = "/".join(path_parts[1:]) if len(path_parts) > 1 else ""

    download_gcs_folder(bucket_name, prefix, local_dir)
