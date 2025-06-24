# legacy screenshotter that uses mss to grab screenshots
from __future__ import annotations

import concurrent.futures
import time
from multiprocessing import Process, Queue

import mss
import numpy as np

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)


def write_npz(filename, frames, timestamps):
    np.savez_compressed(filename, frames=frames, timestamps=timestamps)


def grab(queue: Queue, fps: int = 10):
    interval = 1.0 / fps

    with mss.mss() as sct:
        mon = sct.monitors[0]

        next_time = time.perf_counter() + interval
        try:
            while True:
                timestamp_before = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                frame = sct.grab(mon)
                timestamp_after = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                avg_frame_time = round((timestamp_after + timestamp_before) / 2)
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


def save(queue: Queue) -> None:
    timestamps = []

    idx = 0
    inc = 0

    frame_shape = None

    while True:
        img = queue.get()
        if img is None:
            break

        frame, timestamp = img

        array = np.asarray(frame)

        if frame_shape is None:
            frame_shape = array.shape
            buffer = np.empty((BUFFER_SIZE, *frame_shape), dtype=np.uint8)

        buffer[idx] = array
        timestamps.append(timestamp)
        idx += 1

        if idx >= BUFFER_SIZE:
            print(f"creating file with {buffer.shape[0]} frames...")
            all_timestamps = np.array(timestamps)
            start = time.perf_counter()
            # f = gzip.GzipFile(f"tmp/output_{inc:06d}.npz.gz", "w")
            # np.savez(f, frames=all_buffers, timestamps=all_timestamps)
            # f.close()
            fn = f"tmp/output_{inc:06d}.npz"
            frames_copy = buffer.copy()
            ts_copy = all_timestamps.copy()
            executor.submit(write_npz, fn, frames_copy, ts_copy)

            # np.savez_compressed(f"tmp/output_{inc:06d}.npz", frames=buffer, timestamps=all_timestamps)
            print(time.perf_counter() - start)

            idx = 0
            inc += 1
            timestamps.clear()


if __name__ == "__main__":
    # The screenshots queue
    queue: Queue = Queue(maxsize=BUFFER_SIZE * 3)

    # 2 processes: one for grabbing and one for saving PNG files
    p1 = Process(target=grab, args=(queue,))
    p2 = Process(target=save, args=(queue,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print("Recording finished.")

    executor.shutdown(wait=True)
