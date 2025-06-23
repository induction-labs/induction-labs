# legacy screenshotter that uses mss to grab screenshots
from __future__ import annotations

import concurrent.futures
import subprocess
import time
from multiprocessing import Process, Queue

import mss
import numpy as np

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)


def write_npz(filename, frames, timestamps):
    np.savez_compressed(filename, frames=frames, timestamps=timestamps)


def start_ffmpeg(width: int, height: int, fps: int, outfile: str = "capture_%03d.mp4"):
    """
    Launch ffmpeg so it reads raw BGRA frames from stdin and encodes H.264.
    `-use_wallclock_as_timestamps 1` tells the demuxer to build PTS/DTS off the
    wall-clock time of each write() call, so the timestamps match your
    `CLOCK_MONOTONIC` captures.
    """
    cmd = (
        "ffmpeg",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgra",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps}",  # input option ➜ **before** -i
        "-use_wallclock_as_timestamps",
        "1",
        "-i",
        "pipe:0",  # read raw frames from stdin
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-copyts",
        "-muxdelay",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "segment",
        "-segment_time",
        "10",
        "-fps_mode",
        "vfr",
        f"{outfile}",
    )
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def grab(queue: Queue, fps: int = 12):
    interval = 1.0 / fps

    with mss.mss() as sct:
        mon = sct.monitors[0]

        next_time = time.perf_counter() + interval
        try:
            while True:
                timestamp_before = time.time_ns() * 1e-9
                frame = sct.grab(mon)
                timestamp_after = time.time_ns() * 1e-9
                avg_frame_time = (timestamp_after + timestamp_before) / 2
                queue.put((frame, avg_frame_time))

                now = time.perf_counter()
                sleep_time = next_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_time += interval
        except KeyboardInterrupt:
            pass
        finally:
            queue.put(None)


BUFFER_SIZE = 128


def save(queue: Queue, fps: int = 12, outfile: str = "capture.mp4") -> None:
    ffmpeg = None
    try:
        with open("frame_times.log", "w") as f:
            while True:
                item = queue.get()
                if item is None:  # sentinel → stop
                    break

                frame, timestamp_ns = item
                arr = np.asarray(frame)  # (H, W, 4) BGRA

                if ffmpeg is None:
                    h, w, _ = arr.shape
                    ffmpeg = start_ffmpeg(w, h, fps, outfile)

                # Write raw bytes directly; BGRA is already tightly packed
                ffmpeg.stdin.write(arr.tobytes())
                f.write(f"{timestamp_ns}\n")
    finally:
        if ffmpeg:
            ffmpeg.stdin.close()  # flush & send EOF to ffmpeg
            ffmpeg.wait()


if __name__ == "__main__":
    print("starting")
    queue: Queue = Queue(maxsize=BUFFER_SIZE * 3)

    p1 = Process(target=grab, args=(queue,))  # producer
    p2 = Process(target=save, args=(queue, 12, "screen_%03d.mp4"))  # consumer/encoder

    p1.start()
    p2.start()
    p1.join()
    queue.put(None)  # make sure the saver sees the sentinel
    p2.join()
    print("Recording finished.")
