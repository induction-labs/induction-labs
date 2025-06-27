from __future__ import annotations

import json

import imageio.v2 as iio


class Recorder:
    def __init__(self, out_mp4: str, out_meta: str, fps: int = 60):
        self.writer = iio.get_writer(out_mp4, fps=fps, codec="libx264", quality=9)
        self.meta_f = open(out_meta, "w")  # noqa: SIM115
        self.fps = fps
        self.frame_idx = 0

    def capture(self, frame, action):
        self.writer.append_data(frame)
        self.meta_f.write(
            json.dumps(
                {
                    "frame": self.frame_idx,
                    "ts": action.ts,
                    "cursor": action.cursor,
                    "key": action.key,
                }
            )
            + "\n"
        )
        self.frame_idx += 1

    def close(self):
        self.writer.close()
        self.meta_f.close()
