from __future__ import annotations

from typing import Any

import tensorstore as ts
import torch
import zarr
from fsspec.asyn import AsyncFileSystem
from pydantic import BaseModel
from zarr.storage import FsspecStore

context = ts.Context(
    {
        # 2 GB read-back cache in RAM
        "cache_pool": {"total_bytes_limit": 2_000_000_000},
        # Optional: raise GCS concurrency if you hit throughput limits
        "gcs_request_concurrency": {"limit": 16},
    }
)


def get_zarr_root(fs: AsyncFileSystem, path: str) -> zarr.Group:
    """
    Get the Zarr root group for the specified path.
    If the group does not exist, it will be created.
    """
    store = FsspecStore(fs, path=path, read_only=False)
    root = zarr.open_group(store, mode="a")  # "a" = read/write, create if needed
    return root


class ZarrArrayAttributes(BaseModel):
    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    dtype: Any
    path: str
    metadata: dict | None = None

    def __str__(self):
        return f"ZarrArrayAttributes(shape={self.shape}, chunk_shape={self.chunk_shape}, dtype={self.dtype})"


def get_kvstore_config(path: str) -> dict:
    """
    Get the appropriate kvstore configuration based on the path.

    Args:
        path: Path to the zarr file (local or cloud storage)

    Returns:
        Dict containing the kvstore configuration for tensorstore
    """
    if path.startswith("gs://"):
        # Parse GCS path: gs://bucket/path/to/file
        parts = path[5:].split("/", 1)  # Remove 'gs://' and split on first '/'
        bucket = parts[0]
        gcs_path = parts[1] if len(parts) > 1 else ""
        return {"driver": "gcs", "bucket": bucket, "path": gcs_path}
    elif path.startswith("s3://"):
        # Parse S3 path: s3://bucket/path/to/file
        parts = path[5:].split("/", 1)  # Remove 's3://' and split on first '/'
        bucket = parts[0]
        s3_path = parts[1] if len(parts) > 1 else ""
        return {"driver": "s3", "bucket": bucket, "path": s3_path}
    else:
        # Local file path
        return {"driver": "file", "path": path}


def get_tensorstore_spec(path: str, attributes: dict | None = None) -> dict:
    # ---- JSON spec -----------------------------------------------------------
    assert path.startswith("gs://induction-labs/"), (
        f"Path must start with 'gs://induction-labs/', got {path}"
    )

    # Get rid of gs://induction-labs/
    path = path[len("gs://induction-labs/") :]
    return {
        "driver": "zarr3",  #  "zarr" (v2) or "zarr3"
        "kvstore": {  #  Any TensorStore KvStore works here
            "driver": "gcs",  #  <- tells TensorStore to talk to GCS
            "bucket": "induction-labs",  #  GCS bucket name
            "path": path,  # ends with "/" so keys nest
        },
        # (Optional) array-level metadata; if omitted, supply them as kwargs
        "metadata": {  #  for the v2 driver
            "codecs": [
                {
                    "name": "blosc",
                    "configuration": {
                        "cname": "zstd",
                        "clevel": 3,
                        "shuffle": "bitshuffle",
                        "typesize": 1,
                    },
                }
            ],
            "attributes": attributes or {},
        },
    }


async def create_zarr_array(atr: ZarrArrayAttributes):
    """
    Create a Zarr array with the specified parameters if it does not already exist.
    """
    spec = get_tensorstore_spec(path=atr.path, attributes=atr.metadata)
    store = await ts.open(
        spec,
        dtype=atr.dtype,  # required if not in metadata
        shape=atr.shape,  # idem
        chunk_layout=ts.ChunkLayout(chunk_shape=atr.chunk_shape),
        create=True,  # create if missing
        open=True,  # succeed if it already exists
        context=context,
    )

    return store


async def append_batch(z: Any, batch: torch.Tensor, chunk_start: int):
    """
    batch: Tensor of shape (T, 3, H, W), dtype=uint8
    """
    # assert batch.ndim == 4, "Batch must be a 4D tensor (T, C, H, W)"
    # assert batch.shape[1:] == z.shape[1:], (
    #     f"Batch shape must match the Zarr array shape (C, H, W), {batch.shape=}, {z.shape=}"
    # )
    T = batch.shape[0]
    # 2b) Grow the array by T along axis 0
    assert z.shape[0] >= chunk_start + T, (
        f"Zarr array must have enough space to append {T} frames at index {chunk_start}, "
        f"current shape is {z.shape[0]} frames."
    )
    # await z.resize(exclusive_max=[chunk_start + T, 3, H, W])
    await z[chunk_start : chunk_start + T].write(batch.cpu().numpy())
