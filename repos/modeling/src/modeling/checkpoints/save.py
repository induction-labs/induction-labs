import logging
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from modeling.utils.cloud_path import CloudPath
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from modeling.modules.base_module import BaseLITModule
from modeling.config import GCSCheckpointConfig, ExperimentConfig

import os
from google.cloud import storage
from smart_open import open as smart_open
import tempfile
import shutil

from transformers.modeling_utils import PreTrainedModel
from synapse.utils.logging import configure_logging

logger = configure_logging(__name__, logging.DEBUG)


def save_checkpoint_to_gcs(
    model: PreTrainedModel, gcs_bucket: str, gcs_prefix: Path, local_dir: Path
) -> None:
    """
    1) Save HF model & tokenizer to local_dir
    2) Upload every file under local_dir to gs://{gcs_bucket}/{gcs_prefix}/...

    Args:
        model:          a HuggingFace PreTrainedModel instance
        tokenizer:      the corresponding PreTrainedTokenizer
        gcs_bucket:     name of your GCS bucket (must already exist)
        gcs_prefix:     folder prefix inside the bucket, e.g. "checkpoints/run1"
        local_dir:      temporary local directory to save to
    """
    # 1) Save locally
    os.makedirs(local_dir, exist_ok=True)
    model.save_pretrained(local_dir)
    # tokenizer.save_pretrained(local_dir)

    # 2) Upload to GCS
    client = storage.Client()  # Assumes GOOGLE_APPLICATION_CREDENTIALS is set
    bucket = client.bucket(gcs_bucket)

    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            # compute the destination path in the bucket
            rel_path = os.path.relpath(local_path, local_dir)
            gcs_path = os.path.join(gcs_prefix, rel_path).replace("\\", "/")
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
    # Empty the local directory after upload
    shutil.rmtree(local_dir, ignore_errors=True)
    os.makedirs(local_dir, exist_ok=True)  # Recreate empty directory


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


class GCSCheckpointCallback(Callback):
    def __init__(self, exp_config: ExperimentConfig):
        self.exp_config = exp_config
        self.tmp_dir: Path | None = None

    @property
    def ckpt_config(self) -> GCSCheckpointConfig:
        """Return the GCS checkpoint configuration."""
        assert self.exp_config.metadata.checkpoint is not None, (
            "Checkpoint configuration is not set in the experiment metadata"
        )
        return self.exp_config.metadata.checkpoint

    @property
    def ckpt_path(self) -> CloudPath:
        return self.ckpt_config.checkpoint_path

    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        # Set up the temporary directory for saving checkpoints
        assert self.tmp_dir is None, "Temporary directory already set up"
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="gcs-checkpoint-"))
        bucket, gcs_prefix = self.ckpt_config.bucket_and_path
        ensure_empty_gcs_prefix(bucket_name=bucket, output_dir=str(gcs_prefix))
        with smart_open(self.ckpt_config.ckpt_config_path.to_str(), "w") as f:
            f.write(self.exp_config.serialize_to_toml())

    def teardown(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        """Called when fit, validate, test, predict, or tune ends."""
        # Clean up the temporary directory
        assert self.tmp_dir is not None, "Temporary directory not set up"
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.

        """
        assert isinstance(pl_module, BaseLITModule), (
            f"Expected pl_module to be an instance of BaseLITModule, got {type(pl_module)}"
        )
        assert self.tmp_dir is not None, (
            "Temporary directory for checkpoints is not set up"
        )

        step_num = trainer.global_step
        model = pl_module.model
        assert isinstance(model, PreTrainedModel)
        # TODO: figure out why this isn't auto inferred from covariant bound
        if self.ckpt_config.should_checkpoint(step_num):
            # Save the model and tokenizer to the temporary directory
            logger.info(
                f"Saving checkpoint to GCS at step {step_num} to {self.ckpt_path}"
            )
            save_checkpoint_to_gcs(
                model=model,
                gcs_bucket=self.ckpt_config.bucket_and_path[0],
                gcs_prefix=self.ckpt_config.bucket_and_path[1] / f"step_{step_num}",
                local_dir=self.tmp_dir,
            )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Save the initial model checkpoint
        assert isinstance(pl_module, BaseLITModule), (
            f"Expected pl_module to be an instance of BaseLITModule, got {type(pl_module)}"
        )
        assert self.tmp_dir is not None, (
            "Temporary directory for checkpoints is not set up"
        )

        model = pl_module.model
        assert isinstance(model, PreTrainedModel)
        if self.ckpt_config.checkpoint_first_step:
            logger.info(f"Saving first checkpoint to {self.ckpt_path}")
            save_checkpoint_to_gcs(
                model=model,
                gcs_bucket=self.ckpt_config.bucket_and_path[0],
                gcs_prefix=self.ckpt_config.bucket_and_path[1] / "step_0",
                local_dir=self.tmp_dir,
            )

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when fit ends."""
        # Save the final model checkpoint
        assert isinstance(pl_module, BaseLITModule), (
            f"Expected pl_module to be an instance of BaseLITModule, got {type(pl_module)}"
        )
        assert self.tmp_dir is not None, (
            "Temporary directory for checkpoints is not set up"
        )

        model = pl_module.model
        assert isinstance(model, PreTrainedModel)
        if self.ckpt_config.checkpoint_last_step:
            logger.info(f"Saving last checkpoint to {self.ckpt_path}")
            save_checkpoint_to_gcs(
                model=model,
                gcs_bucket=self.ckpt_config.bucket_and_path[0],
                gcs_prefix=self.ckpt_config.bucket_and_path[1] / "step_-1",
                local_dir=self.tmp_dir,
            )
