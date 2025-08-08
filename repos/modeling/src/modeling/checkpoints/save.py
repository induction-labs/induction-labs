import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from google.cloud import storage
from synapse.utils.logging import configure_logging
from transformers.modeling_utils import PreTrainedModel

logger = configure_logging(__name__, level=logging.INFO)


def save_checkpoint_to_tmpdir(model: PreTrainedModel, local_dir: Path) -> None:
    # I don't even care at this point. This is called on every single device, i guess somehow huggingface
    # magically figures out what rank its running on and saves the model only once. Doesn't say it in the code but whatever.
    # Actually hate huggingface
    os.makedirs(local_dir, exist_ok=True)
    model.save_pretrained(local_dir)


logger = logging.getLogger(__name__)


def upload_to_gcs(local_dir: Path, gcs_bucket: str, gcs_prefix: Path) -> None:
    dest = f"gs://{gcs_bucket}/{gcs_prefix.as_posix().rstrip('/')}/"
    command = [
        "gcloud",
        "storage",
        "cp",
        "-r",
        f"{str(local_dir).rstrip('/')}/*",
        dest,
    ]

    logger.info("Uploading %s → %s ", local_dir, dest)
    logger.info("Running command: %s", " ".join(command))
    try:
        result = subprocess.run(
            command,
            capture_output=True,  # grab both streams
            text=True,  # str not bytes
            check=True,  # raise if non-zero
        )

        # Success ─ log everything that gcloud printed
        if result.stdout:
            logger.info(result.stdout.rstrip())
        if result.stderr:  # gcloud progress / warnings
            logger.warning(result.stderr.rstrip())

    except subprocess.CalledProcessError as e:
        # Command failed ─ dump its output *before* re-raising
        logger.error("gcloud exited with status %s", e.returncode)

        # e.stdout / e.stderr contain the captured streams
        if e.stdout:
            logger.error("stdout:\n%s", e.stdout.rstrip())
        if e.stderr:
            logger.error("stderr:\n%s", e.stderr.rstrip())

        raise  # keep the original stack trace

    finally:
        shutil.rmtree(local_dir, ignore_errors=True)


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


BUCKET = "induction-labs"
BASE_PREFIX = Path("jeffrey/testing")


@pytest.mark.integration
def test_upload_to_gcs_including_nested(tmp_path: Path):
    """
    Requires:
      - gcloud CLI installed
      - Authenticated with access to the bucket
    Run with: pytest -v -m integration
    """
    # Arrange: create top-level and nested files
    (tmp_path / "hello.txt").write_text("top-level\n")
    nested_dir = tmp_path / "nest_dir"
    nested_dir.mkdir()
    (nested_dir / "test2.txt").write_text("nested\n")

    # Use a unique sub-prefix under jeffrey/testing to avoid collisions
    run_id = uuid.uuid4().hex[:8]
    prefix = BASE_PREFIX / f"integration-{run_id}"

    # Act: upload directory contents
    upload_to_gcs(tmp_path, BUCKET, prefix)

    # Assert: both files exist at the expected (non-nested-root) paths
    gs_top = f"gs://{BUCKET}/{prefix.as_posix()}/hello.txt"
    gs_nested = f"gs://{BUCKET}/{prefix.as_posix()}/nest_dir/test2.txt"

    r1 = subprocess.run(
        ["gcloud", "storage", "cat", gs_top], capture_output=True, text=True, check=True
    )
    r2 = subprocess.run(
        ["gcloud", "storage", "cat", gs_nested],
        capture_output=True,
        text=True,
        check=True,
    )

    assert r1.stdout == "top-level\n"
    assert r2.stdout == "nested\n"

    # Cleanup: remove the uploaded test artifacts
    subprocess.run(
        ["gcloud", "storage", "rm", "-r", f"gs://{BUCKET}/{prefix.as_posix()}/"],
        check=True,
    )


if __name__ == "__main__":
    # Run the test directly if this file is executed
    pytest.main([__file__, "-v", "-m", "integration"])
# uv run python -m modeling.checkpoints.save
