from __future__ import annotations

import av
import matplotlib.pyplot as plt
import numpy as np
from video_reader import PyVideoReader

# Open the input file
container = av.open("test_data/output.mp4")

# Grab the first video stream
video_stream = container.streams.video[0]

# Decode and iterate over each video frame
for i, frame in enumerate(container.decode(video_stream)):
    pts = frame.pts
    # Convert PTS to seconds using the stream's time_base
    time_in_seconds = float(pts * video_stream.time_base)
    print(f"Frame : PTS={pts}, time={time_in_seconds:.3f}s")
    if i > 100:
        break


async def render_zarr_frame(frame: np.ndarray) -> np.ndarray:
    assert frame.ndim == 3, "Input should be a 4D numpy array with shape (C, H, W)"

    # Convert from (C, H, W) to (H, W, C) format for display
    frame = np.transpose(frame, (1, 2, 0))

    # Display the frame
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)
    # plt.title(f"Frame {frame_number}")
    plt.axis("off")
    plt.show()

    return frame


def decode_single_frame_to_array(
    video_path: str, target_frame_idx: int = 400, fmt: str = "rgb24"
) -> np.ndarray:
    """
    Decode the frame at index `target_frame_idx` (zero-based) from `video_path`
    and return it as an HxWxC NumPy array in the specified pixel format (default 'rgb24').

    Raises:
        IndexError: if the video has fewer than target_frame_idx+1 frames.
    """
    container = av.open(video_path)

    for idx, frame in enumerate(container.decode(video=0)):
        if idx == target_frame_idx:
            # Convert to HxWxC NumPy array
            arr = frame.to_ndarray(format=fmt)
            container.close()
            # Convert to CHW format if needed
            arr = np.transpose(arr, (2, 0, 1)) if fmt == "rgb24" else arr
            return arr

    container.close()
    raise IndexError(
        f"Video contains only {idx + 1} frames; cannot get frame {target_frame_idx}."
    )


def decode_single_frame_pyvideo(
    video_path: str, target_frame_idx: int = 400
) -> np.ndarray:
    """
    Decode the frame at index `target_frame_idx` (zero-based) from `video_path`
    using PyVideoReader and return it as an HxWxC NumPy array in RGB format.

    Raises:
        IndexError: if the video has fewer than target_frame_idx+1 frames.
    """
    vr = PyVideoReader(video_path)
    # total_frames = vr.get_info()["frame_count"]

    # if target_frame_idx >= total_frames:
    #     raise IndexError(
    #         f"Video contains only {total_frames} frames; cannot get frame {target_frame_idx}."
    #     )

    frame = vr.get_batch([target_frame_idx], True)[0]
    return frame
