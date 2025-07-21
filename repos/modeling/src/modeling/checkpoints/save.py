import logging
import os
import shutil
from pathlib import Path

from google.cloud import storage
from synapse.utils.logging import configure_logging
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel

logger = configure_logging(__name__, level=logging.INFO)


def save_checkpoint_to_tmpdir(model: PreTrainedModel, local_dir: Path) -> None:
    # I don't even care at this point. This is called on every single device, i guess somehow huggingface
    # magically figures out what rank its running on and saves the model only once. Doesn't say it in the code but whatever.
    # Actually hate huggingface
    os.makedirs(local_dir, exist_ok=True)
    model.save_pretrained(local_dir)


def upload_to_gcs(local_dir: Path, gcs_bucket: str, gcs_prefix: Path) -> None:
    """
    Uploads the contents of a local directory to a Google Cloud Storage bucket.

    Args:
        local_dir: Path to the local directory containing files to upload.
        gcs_bucket: Name of the GCS bucket to upload files to.
        gcs_prefix: Path prefix within the bucket where files will be uploaded.
    """
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    all_files = [
        (root, fname) for root, _, files in os.walk(local_dir) for fname in files
    ]
    total_files = len(all_files)

    logger.info(
        f"Uploading {total_files} files from {local_dir} to gs://{gcs_bucket}/{gcs_prefix}"
    )
    for root, fname in tqdm(
        all_files,
        total=total_files,
        desc="Uploading",
    ):
        local_path = os.path.join(root, fname)
        # compute the destination path in the bucket
        rel_path = os.path.relpath(local_path, local_dir)
        gcs_path = (gcs_prefix / rel_path).as_posix()
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
    # Empty the local directory after upload
    shutil.rmtree(local_dir, ignore_errors=True)


# def save_checkpoint_to_gcs(
#     model: PreTrainedModel,
#     gcs_bucket: str,
#     gcs_prefix: Path,
#     local_dir: Path,
#     is_rank0: bool,
# ) -> None:
#     """
#     1) Save HF model & tokenizer to local_dir
#     2) Upload every file under local_dir to gs://{gcs_bucket}/{gcs_prefix}/...

#     Args:
#         model:          a HuggingFace PreTrainedModel instance
#         tokenizer:      the corresponding PreTrainedTokenizer
#         gcs_bucket:     name of your GCS bucket (must already exist)
#         gcs_prefix:     folder prefix inside the bucket, e.g. "checkpoints/run1"
#         local_dir:      temporary local directory to save to
#     """
#     # 1) Save locally
#     # Performs unsharded saving
#     # Gather the parameters on all ranks

#     # tokenizer.save_pretrained(local_dir)

#     # 2) Upload to GCS
#     client = storage.Client()  # Assumes GOOGLE_APPLICATION_CREDENTIALS is set
#     bucket = client.bucket(gcs_bucket)

#     for root, _, files in os.walk(local_dir):
#         for fname in files:
#             local_path = os.path.join(root, fname)
#             # compute the destination path in the bucket
#             rel_path = os.path.relpath(local_path, local_dir)
#             gcs_path = os.path.join(gcs_prefix, rel_path).replace("\\", "/")
#             blob = bucket.blob(gcs_path)
#             blob.upload_from_filename(local_path)
#     # Empty the local directory after upload
#     shutil.rmtree(local_dir, ignore_errors=True)


def ensure_empty_gcs_prefix(bucket_name: str, output_dir: str) -> None:
    """
    Raises a RuntimeError if there are already any objects under `output_dir/`
    in the given bucket. Otherwise, creates an empty placeholder so that
    listing the "directory" will show up in some UIs.

    Args:
        bucket_name:   your GCS bucket name
        output_dir:    desired prefix (can include or omit trailing slash)
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Normalize prefix to always end with "/"
    prefix = output_dir.rstrip("/") + "/"

    # 1) Check that it doesn't already exist
    existing = list(bucket.list_blobs(prefix=prefix, max_results=1))
    if existing:
        raise RuntimeError(
            f"Output directory gs://{bucket_name}/{prefix} already exists"
        )
