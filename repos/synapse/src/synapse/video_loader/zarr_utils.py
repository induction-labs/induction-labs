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
        "gcs_request_concurrency": {"limit": 32},
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


def get_tensorstore_spec(path: str, attributes: dict | None = None) -> dict:
    # ---- JSON spec -----------------------------------------------------------
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
        dtype=ts.uint8,  # required if not in metadata
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
    assert batch.ndim == 4, "Batch must be a 4D tensor (T, C, H, W)"
    # assert batch.shape[1:] == z.shape[1:], (
    #     f"Batch shape must match the Zarr array shape (C, H, W), {batch.shape=}, {z.shape=}"
    # )
    T, _, H, W = batch.shape
    # start = z.shape[0]
    # 2b) Grow the array by T along axis 0
    await z.resize(exclusive_max=[chunk_start + T, 3, H, W])
    await z[chunk_start : chunk_start + T].write(batch.cpu().numpy())
