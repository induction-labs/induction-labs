from __future__ import annotations

import gcsfs
import tensorstore as ts
import torch
from numcodecs.zarr3 import Zstd
from zarr.storage import FsspecStore

from synapse.elapsed_timer import elapsed_timer
from synapse.qwen_omni_utils.video_process import fetch_video

fs = gcsfs.GCSFileSystem(project="induction-labs", asynchronous=False)  # Auth via ADC

store = FsspecStore(fs, path="induction-labs/jeffrey/test_vid4.zarr")

codec = Zstd(level=3)  # any numcodecs codec


# ---- JSON spec -----------------------------------------------------------
spec = {
    "driver": "zarr3",  #  "zarr" (v2) or "zarr3"
    "kvstore": {  #  Any TensorStore KvStore works here
        "driver": "gcs",  #  <- tells TensorStore to talk to GCS
        "bucket": "induction-labs",  #  GCS bucket name
        "path": "jeffrey/frames/test4/",  # ends with "/" so keys nest
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
        "attributes": {
            # "fps": 30,
            # "source_camera": "GoPro-11",
            # "preprocessing": "resize_224_bicubic",
            # "sha256": "d2c7…",
        },
    },
}


# compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
async def main():
    context = ts.Context(
        {
            # 2 GB read-back cache in RAM
            "cache_pool": {"total_bytes_limit": 2_000_000_000},
            # Optional: raise GCS concurrency if you hit throughput limits
            "gcs_request_concurrency": {"limit": 32},
        }
    )
    video_data = {
        "type": "video",
        "video": TEST_VIDEO,
        "max_pixels": 854 * 480,
    }
    with elapsed_timer() as timer:
        video_tensor, video_fps = fetch_video(video_data, return_video_sample_fps=True)
    print(size_repr(video_tensor))
    print(f"Video fetch took {timer():.2f} seconds")
    (frames, channel, height, width) = video_tensor.shape
    with elapsed_timer() as timer:
        store = await ts.open(
            spec,
            dtype=ts.uint8,  # required if not in metadata
            shape=[frames, channel, height, width],  # idem
            chunk_layout=ts.ChunkLayout(chunk_shape=[32, channel, height, width]),
            create=True,  # create if missing
            open=True,  # succeed if it already exists
            context=context,
        )
        await store.write(video_tensor)  #  handles chunking & parallel upload
    print(f"TensorStore write took {timer():.2f} seconds")


def size_repr(self):
    return (
        f"tensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"
    )


async def get_video_numpy():
    smol_spec = {k: v for k, v in spec.items() if k not in ("metadata")}
    store = await ts.open(
        smol_spec,
        open=True,  # just open - don't recreate
    )

    # ‼️  This is the whole-array read - it returns a NumPy ndarray
    video_np = await store.read()  # (T, H, W, C), dtype=uint8
    print(store.spec())
    return video_np


# Override the __repr__ method
# torch.Tensor.__str__ = torch.Tensor.__repr__
torch.Tensor.__repr__ = size_repr

# model_config = Qwen2_5OmniThinkerConfig.from_pretrained("Qwen/Qwen2.5-Omni-3B")
# print(model_config)
TEST_VIDEO = "test_data/video_003.mp4"
TEST_IMAGE = "https://www.ilankelman.org/stopsigns/australia.jpg"


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

    k = asyncio.run(get_video_numpy())
    print(size_repr(k))
    # number of bytes:
    print(
        f"Video numpy shape: {k.shape}, dtype: {k.dtype}, size: {k.nbytes / 1e6:.2f} MB"
    )
    # main2()
