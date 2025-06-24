from __future__ import annotations

import bisect
import json

import av
import cv2

VIDEO_PATH = "tmp/2025-06-10_231949_ABTR8/screen_capture_000000.mp4"
ACTIONS_PATH = "tmp/2025-06-10_231949_ABTR8/action_capture_000000.jsonl"

# Load actions (one JSON record per line)
actions = []
with open(ACTIONS_PATH) as f:
    for line in f:
        rec = json.loads(line)
        act = rec["action"]
        if act["action"] == "mouse_move":
            # timestamp assumed in milliseconds
            actions.append(
                {"timestamp": rec["timestamp"], "x": act["x"] * 2, "y": act["y"] * 2}
            )

# Sort actions by timestamp
actions.sort(key=lambda a: a["timestamp"])
timestamps = [a["timestamp"] for a in actions]

# Open video with PyAV
container = av.open(VIDEO_PATH)
video_stream = container.streams.video[0]

min_ms = float(video_stream.start_time * video_stream.time_base) * 1000.0
range_ms = float(video_stream.duration * video_stream.time_base) * 1000.0
max_ms = min_ms + range_ms
frame_time_ms = float(video_stream.time_base) * 1000.0


# Convenience function to grab and display a frame at a given absolute timestamp (ms)
def show_frame(ts_ms):
    # ts_sec = ts_ms / 1000.0
    # seek_pts = int(ts_sec / float(video_stream.time_base))
    # print(seek_pts)
    seek_pts = int((ts_ms - min_ms) / video_stream.time_base)
    container.seek(seek_pts, any_frame=True, backward=True, stream=video_stream)

    for frame in container.decode(video_stream):
        frame_time = float(frame.pts * video_stream.time_base) * 1000.0
        img = frame.to_ndarray(format="bgr24")
        break

    idx = bisect.bisect_left(timestamps, frame_time)
    if idx >= len(actions):
        idx = len(actions) - 1
    action = actions[idx]

    cv2.circle(img, (int(action["x"]), int(action["y"])), 10, (0, 0, 255), 2)
    cv2.putText(
        img,
        f"Frame: {frame_time:.1f} ms",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"Action: {action['timestamp']:.1f} ms",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Video with Cursor Overlay", img)


# Track current trackbar position
current_pos = 0


def set_position(pos):
    global current_pos
    current_pos = pos
    abs_ts = min_ms + pos * frame_time_ms
    show_frame(abs_ts)


# Setup window and trackbar
cv2.namedWindow("Video with Cursor Overlay", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video with Cursor Overlay", 1280, 720)
cv2.createTrackbar(
    "Time (ms)",
    "Video with Cursor Overlay",
    0,
    int(range_ms),
    lambda v: set_position(v),
)

# Display the initial frame
set_position(0)

# Main loop: arrow keys to step frames, ESC to exit
print("Controls: Left/Right arrows to step 1 ms, ESC or 'q' to quit.")
while True:
    key = cv2.waitKey(30)
    # Debug: uncomment to print actual key codes
    # print(f"Key pressed: {key}")

    if key in (27, ord("q")):  # ESC or 'q'
        break
    # Left arrow (Windows/Linux/mac codes)
    elif key in (81, 2424832, 65361, 2424832, 2):
        new_pos = max(0, current_pos - 1)
        cv2.setTrackbarPos("Time (ms)", "Video with Cursor Overlay", new_pos)
        set_position(new_pos)
    # Right arrow (Windows/Linux/mac codes)
    elif key in (83, 2555904, 65363, 2555904, 3):
        new_pos = min(range_ms, current_pos + 1)
        cv2.setTrackbarPos("Time (ms)", "Video with Cursor Overlay", new_pos)
        set_position(new_pos)

cv2.destroyAllWindows()
