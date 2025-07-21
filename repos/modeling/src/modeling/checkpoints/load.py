# 1) Install dependencies
# pip install google-cloud-storage transformers safetensors

from pathlib import Path

from google.cloud import storage
from huggingface_hub import snapshot_download
from synapse.utils.logging import configure_logging
from tqdm import tqdm

from modeling.utils.cloud_path import CloudPath

logger = configure_logging(__name__)


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
    Download a folder from Google Cloud Storage with progress bar.

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

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # First pass: collect all blobs to get total count
    logger.debug("Scanning files to download...")
    blobs_to_download = []
    for blob in bucket.list_blobs(prefix=prefix):
        # Skip directories (blobs ending with '/')
        if blob.name.endswith("/"):
            continue

        # Calculate relative path from prefix
        if prefix:
            # Remove the prefix from the blob name to get relative path
            if blob.name.startswith(prefix):
                relative_path = blob.name[len(prefix) :].lstrip("/")
            else:
                # This shouldn't happen with list_blobs(prefix=...), but handle it
                relative_path = blob.name
        else:
            relative_path = blob.name

        # Skip if relative path is empty (shouldn't happen)
        if not relative_path:
            continue

        blobs_to_download.append((blob, relative_path))

    if not blobs_to_download:
        logger.warning(f"No files found in gs://{bucket_name}/{prefix}")
        return

    logger.info(f"Found {len(blobs_to_download)} files to download")

    # Second pass: download with progress bar
    with tqdm(
        total=len(blobs_to_download), desc="Downloading files", unit="file"
    ) as pbar:
        for blob, relative_path in blobs_to_download:
            local_file_path = local_dir / relative_path

            # Create parent directories if they don't exist
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Update progress bar description with current file
            pbar.set_postfix_str(f"Downloading {relative_path}")
            logger.debug(f"Downloading {blob.name} to {local_file_path}")

            # Download the blob
            blob.download_to_filename(local_file_path)
            pbar.update(1)

    logger.info(
        f"Downloaded {len(blobs_to_download)} files from gs://{bucket_name}/{prefix} to {local_dir}"
    )


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
