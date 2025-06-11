# code to correct color space of macOS screenshots if needed

from multiprocessing import Process, Queue
import numpy as np

import subprocess, mss
import mss.tools
import time

import time
import cv2
import numpy as np
import mss
from PIL import Image, ImageCms

import ctypes, mss, cv2, numpy as np
from PIL import Image, ImageCms
import Quartz, CoreFoundation, time

import ctypes, Quartz, CoreFoundation

def get_main_display_icc() -> bytes:
    display_id = Quartz.CGMainDisplayID()
    cs        = Quartz.CGDisplayCopyColorSpace(display_id)
    if cs is None:
        raise RuntimeError("Display has no colour‑space")

    cfdata = Quartz.CGColorSpaceCopyICCData(cs)
    size   = CoreFoundation.CFDataGetLength(cfdata)

    # allocate a C buffer and have CoreFoundation fill it
    buf = (ctypes.c_ubyte * size)()                       # C array
    CoreFoundation.CFDataGetBytes(cfdata, (0, size), buf) # CFRange = (loc,len)

    return bytes(buf)                                     # → real Python bytes

# one‑time setup -------------------------------------------------
icc_bytes = get_main_display_icc()       # <- now uses CFDataGetBytes
icc_path = "/tmp/display.icc"
with open(icc_path, "wb") as f:
    f.write(icc_bytes)

monitor_profile = ImageCms.getOpenProfile(icc_path)
srgb_profile    = ImageCms.createProfile("sRGB")
to_srgb = ImageCms.buildTransform(
    monitor_profile, srgb_profile,
    inMode="RGB", outMode="RGB")

def record_raw_video(queue: Queue, fps: int =30):
    interval = 1.0 / fps

    with mss.mss() as sct:
        mon = sct.monitors[0]
        initial = sct.grab(mon)
        width, height = initial.size

        next_frame_time = time.time()
        avg_time_per_write = 0.0

        try:
            while True:
                start = time.time()
                timestamp_before = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                frame = sct.grab(mon)
                timestamp_after = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                queue.put((frame, avg_frame_time))
                end = time.time()

                avg_frame_time = (timestamp_after + timestamp_before) / 2
                avg_time_per_write = 0.9 * avg_time_per_write + 0.1 * (end - start)

                next_frame_time += interval - avg_time_per_write
                sleep = next_frame_time - time.time()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    # if we’re behind, skip sleeping to catch up
                    next_frame_time = time.time()
        except KeyboardInterrupt:
            pass
        finally:
            queue.put(None)

def save(queue: Queue) -> None:
    np_buffer = []
    while (img := queue.get()):
        frame, timestamp = img
        np_buffer.append(frame.raw)

import cv2

with mss.mss() as sct:
    mon = sct.monitors[0]
    initial = sct.grab(mon)
    width, height = initial.size
    frame = np.asarray(initial)

    start = time.time()
    w, h = frame.shape[1], frame.shape[0]

    img_p = Image.frombuffer(
        "RGBA",                 # legal Pillow mode
        (w, h),
        frame,                  # the NumPy BGRA buffer
        "raw",                  # use the “raw” decoder
        "BGRA",                 # how those bytes are actually laid out
        0, 1                    # stride = 0 (default), top‑to‑bottom
    ).convert("RGB")            # strip alpha + keep LittleCMS happy

    # wide‑gamut  ➜  sRGB
    img_srgb = ImageCms.applyTransform(img_p, to_srgb)
    end = time.time()
    print(end-start)
    img_srgb.save("img1.png", "PNG")
