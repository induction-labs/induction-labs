from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def render_zarr_frame(frame: np.ndarray) -> np.ndarray:
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
