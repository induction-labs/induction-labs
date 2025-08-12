from __future__ import annotations

import datetime
import os
import random
import re
import socket
import string
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import typer
from tqdm import tqdm

from actioncollector.record_actions import ActionRecorder
from actioncollector.utils import recording_metadata, upload_to_gcs_and_delete


@lru_cache(maxsize=1)
def get_bundled_executable(name: str) -> str:
    """Get path to bundled executable, fallback to system version"""
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle
        bundle_dir = Path(sys._MEIPASS)
        bundled_path = bundle_dir / name
        if bundled_path.exists():
            return str(bundled_path)

    try:
        proc = subprocess.run(
            ["which", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        path = proc.stdout.strip()
        if proc.returncode == 0 and path:
            return path
    except Exception:
        pass

    # Last-resort fallback
    return f"bin/{name}"


app = typer.Typer()


def start_screen_record(
    output_path: str, segment_time: int = 30, framerate: int = 30, device_index: int = 1
):
    # run this cmd
    # ffmpeg -f avfoundation -framerate 30 \
    # -use_wallclock_as_timestamps 1 -capture_cursor 1 -i "1:none" \
    # -c:v libx264 -g 15 -c:a aac -preset veryfast -crf 17 \
    # -copyts -muxdelay 0 \
    # -segment_time 10 -f segment \
    # -r 30 tmp/video_%03d.mp4

    ffmpeg_name = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    ffmpeg_path = get_bundled_executable(ffmpeg_name)
    if sys.platform.startswith("win"):
        cmd = [
            ffmpeg_path,
            "-f",
            "gdigrab",  # <-- Windows screen-capture device
            # "-init_hw_device", "qsv=hw,child_device_type=dxva2", "-filter_hw_device", "hw", "-f", "lavfi", "-i", "ddagrab",
            "-framerate",
            str(framerate),
            "-use_wallclock_as_timestamps",
            "1",
            "-draw_mouse",
            "1",  # <-- equivalent to -capture_cursor 1
            "-vsync",
            "vfr",
            "-i",
            "desktop",  # <-- no device index needed on Windows
            "-c:v",
            "libx264",
            "-g",
            "15",
            "-c:a",
            "aac",  # harmless if no audio stream is present
            "-preset",
            "ultrafast",
            "-crf",
            "17",
            "-copyts",
            "-muxdelay",
            "0",
            "-f",
            "segment",
            "-segment_time",
            str(segment_time),
            output_path,
        ]
    else:
        cmd = [
            ffmpeg_path,
            "-f",
            "avfoundation",
            "-framerate",
            str(framerate),
            "-use_wallclock_as_timestamps",
            "1",
            "-capture_cursor",
            "1",
            "-vsync",
            "vfr",
            "-i",
            f"{device_index}:none",
            "-c:v",
            "libx264",
            "-g",
            "15",
            "-c:a",
            "aac",
            "-preset",
            "ultrafast",
            "-crf",
            "17",
            "-copyts",
            "-muxdelay",
            "0",
            "-f",
            "segment",
            "-segment_time",
            str(segment_time),
            # "-r", str(framerate),
            output_path,
        ]
    # open stdin so we can send 'q' to stop cleanly
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW
        if sys.platform.startswith("win")
        else 0,
    )


def ffmpeg_list_video_devices():
    """
    Returns a list of tuples (index, device_name) for all video devices
    as enumerated by ffmpeg's AVFoundation input.
    """
    # Run ffmpeg to list devices; ffmpeg writes device lists to stderr
    ffmpeg_path = get_bundled_executable("ffmpeg")
    result = subprocess.run(
        [ffmpeg_path, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = result.stderr.splitlines()

    devices = []
    start_index = None

    # Locate the start of the video device list
    for i, line in enumerate(lines):
        if "AVFoundation video devices" in line:
            start_index = i
            break

    if start_index is None:
        raise RuntimeError(
            "Could not find 'AVFoundation video devices' in ffmpeg output"
        )

    # Parse lines until the audio devices section
    for line in lines[start_index + 1 :]:
        if "AVFoundation audio devices" in line:
            break
        match = re.match(r".*\[(\d+)\]\s*(.+)", line)
        if match:
            idx = int(match.group(1))
            name = match.group(2).strip()
            devices.append((idx, name))

    return devices


def get_default_device(devices: list[tuple[int, str]]) -> int:
    for idx, name in devices:
        if "Capture screen" in name:
            return idx


def test_screen_recording_permissions(device_index: int = 1) -> bool:
    """Test if ffmpeg can access screen recording on macOS by attempting a short capture"""
    if sys.platform != "darwin":
        return True  # Only relevant for macOS

    ffmpeg_path = get_bundled_executable("ffmpeg")
    test_output = "/tmp/screen_test.mp4"

    # Attempt a very short screen recording (1 second)
    cmd = [
        ffmpeg_path,
        "-f",
        "avfoundation",
        "-framerate",
        "1",
        "-t",
        "1",  # Record for 1 second only
        "-i",
        f"{device_index}:none",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-y",  # Overwrite output file
        test_output,
    ]

    try:
        # Run with timeout to prevent hanging
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )

        # Clean up test file if it was created
        if os.path.exists(test_output):
            os.remove(test_output)

        # Check stderr for permission-related errors
        stderr_lower = result.stderr.lower()
        permission_errors = [
            "operation not permitted",
            "permission denied",
            "screen capture not allowed",
            "cgwindowlistcopywindowinfo",
        ]

        if any(error in stderr_lower for error in permission_errors):
            return False

        # If return code is 0 or the process completed without permission errors
        return result.returncode == 0 or not any(
            error in stderr_lower for error in permission_errors
        )

    except subprocess.TimeoutExpired:
        print("[warning] Permission test timed out")
        return False
    except Exception as e:
        print(f"[warning] Permission test failed: {e}")
        return False


def test_accessibility_permissions() -> bool:
    """Test if accessibility permissions are granted for action logging on macOS"""
    if sys.platform != "darwin":
        return True  # Only relevant for macOS

    try:
        import HIServices

        is_trusted = HIServices.AXIsProcessTrusted()
        return is_trusted

    except ImportError:
        print("[warning] pynput not available for accessibility test")
        return True
    except Exception as e:
        print(f"[warning] Accessibility test failed: {e}")
        return True  # Assume permissions are okay if test fails for unknown reasons


def on_segment_finished(filename: str, gs_file_path: str, callback=None):
    upload_to_gcs_and_delete(filename, gs_file_path + filename.split("/")[-1])
    if callback:
        callback()


@app.command()
def run(
    username: str | None = None,
    output_bucket: str = "induction-labs-data-ext",
    video_segment_buffer_length: float = 30,
):
    if username is None:
        username = os.getenv("USER", "unknown_user")

    filename_session_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    random_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    file_path = f"action_capture/{username}/{filename_session_start_time}_{random_str}/"
    tmp_file_path = "/tmp/" + file_path
    # Create the directory if it doesn't exist
    os.makedirs(tmp_file_path, exist_ok=True)

    gs_file_path = f"gs://{output_bucket}/{file_path}"
    framerate = 6

    metadata = recording_metadata(
        username, video_segment_buffer_length, gs_file_path, framerate
    )

    # Save metadata to a file
    with open(tmp_file_path + "metadata.json", "w") as f:
        import json

        json.dump(metadata, f, indent=4)

    # write metadata to GCS
    upload_to_gcs_and_delete(
        tmp_file_path + "metadata.json", gs_file_path + "metadata.json"
    )

    print("[info] recording metadata:")
    print(json.dumps(metadata, indent=4))

    executor = ThreadPoolExecutor(max_workers=5)

    device_id = 0
    if sys.platform == "darwin":
        print("[info] testing accessibility permissions...")
        if not test_accessibility_permissions():
            print("[error] Accessibility permissions not granted.")
            print("[error] Please go to:")
            print(
                "[error]   System Preferences > Security & Privacy > Privacy > Accessibility"
            )
            print("[error]   or System Settings > Privacy & Security > Accessibility")
            print("[error] and grant permission to this application.")
            executor.shutdown(wait=False)
            return
        print("[info] accessibility permissions verified.")

        devices = ffmpeg_list_video_devices()
        device_id = get_default_device(devices)
        print("[info] available video devices:")
        for idx, name in devices:
            print(f"  [{idx}] {name}")
        print(
            "[info] using device index:", device_id, "which is", devices[device_id][1]
        )

        # Test screen recording permissions
        print("[info] testing screen recording permissions...")
        if not test_screen_recording_permissions(device_id):
            print("[error] Screen recording permissions not granted.")
            print("[error] Please go to:")
            print(
                "[error]   System Preferences > Security & Privacy > Privacy > Screen Recording"
            )
            print(
                "[error]   or System Settings > Privacy & Security > Screen Recording"
            )
            print("[error] and grant permission to this application.")
            executor.shutdown(wait=False)
            return
        print("[info] screen recording permissions verified.")

    # Test accessibility permissions for action logging

    print("[info] starting screen recording…")
    ffmpeg_proc = start_screen_record(
        tmp_file_path + "screen_capture_%06d.mp4",
        segment_time=video_segment_buffer_length,
        device_index=device_id,
        framerate=framerate,
    )

    # Check if ffmpeg process started successfully
    time.sleep(1)  # Give ffmpeg a moment to start
    if ffmpeg_proc.poll() is not None:
        print(
            "[error] ffmpeg process failed to start. This may be due to missing screen recording permissions."
        )
        print(
            "[error] On macOS, go to System Preferences > Security & Privacy > Privacy > Screen Recording"
        )
        print(
            "[error] and ensure this application has permission to record the screen."
        )
        ffmpeg_proc.terminate()
        executor.shutdown(wait=False)
        return

    print("[info] recording screen now.")
    pat = re.compile(r"\[segment @ [^\]]+\] Opening '([^']+)' for writing")
    action_recorder = ActionRecorder(
        tmp_file_path + "action_capture_{i:06d}.jsonl",
        thread_pool=executor,
        gs_file_path=gs_file_path,
        uploaded_callback=lambda: print("[info] action file uploaded to GCS."),
    )
    action_recorder.start()

    done_bar = tqdm(
        total=None,
        desc="Video Segments Done",
        dynamic_ncols=True,
    )

    if sys.platform.startswith("win"):

        def shutdown(server_sock):
            print("[child] shutdown request received")
            server_sock.close()  # breaks the accept loop

            ffmpeg_proc.stdin.write("q\n")
            ffmpeg_proc.stdin.flush()

        def serve_control(port=50572):
            srv = socket.socket()
            srv.bind(("127.0.0.1", port))
            srv.listen(1)
            print(f"[child] PID {os.getpid()} control on port {port}")
            conn, _ = srv.accept()  # blocks until parent connects
            conn.close()
            shutdown(srv)

        t = threading.Thread(target=serve_control, daemon=True)
        t.start()

    try:
        file_in_progress = None
        for line in ffmpeg_proc.stderr:
            # Check if process is still running
            if ffmpeg_proc.poll() is not None:
                print("[warning] ffmpeg process terminated unexpectedly")
                break

            m = pat.search(line)
            if m:
                if file_in_progress:
                    executor.submit(
                        on_segment_finished,
                        file_in_progress,
                        gs_file_path,
                        lambda: done_bar.update(1),
                    )

                file_in_progress = m.group(1)
    except KeyboardInterrupt:
        print("[info] KeyboardInterrupt received, stopping recording…")
        if ffmpeg_proc.poll() is None:
            ffmpeg_proc.stdin.write("q\n")
            ffmpeg_proc.stdin.flush()
    except Exception as e:
        print(f"[error] Error during recording: {e}")
        if ffmpeg_proc.poll() is None:
            print("[info] Terminating ffmpeg process...")
            ffmpeg_proc.terminate()

    # Wait for ffmpeg process with timeout handling
    try:
        ffmpeg_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("[warning] ffmpeg process did not terminate cleanly, force killing...")
        ffmpeg_proc.kill()
        time.sleep(1)
        if ffmpeg_proc.poll() is None:
            print("[error] Failed to kill ffmpeg process")

    action_recorder.stop()

    print("[info] saving last segment…")

    def log_final_done():
        done_bar.update(1)
        print("[info] all segments done.")

    executor.submit(
        on_segment_finished,
        file_in_progress,
        gs_file_path,
        lambda: log_final_done(),
    )

    executor.shutdown(wait=True)
    done_bar.close()

    # delete tmp files
    # for file in os.listdir(tmp_file_path):
    #     file_path = os.path.join(tmp_file_path, file)
    #     if os.path.isfile(file_path):
    #         print(f"[warning] deleting file: {file_path}")
    #         os.remove(file_path)

    print("[info] recording finished. All files uploaded to GCS. Exiting.")


if __name__ == "__main__":
    app()
