import subprocess
from tqdm import tqdm
from typing import Optional
import os
import random
import string
import datetime
import re
from actioncollector.utils import upload_to_gcs_and_delete, recording_metadata
from actioncollector.record_actions import ActionRecorder
from concurrent.futures import ThreadPoolExecutor

import typer

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

    cmd = [
        "ffmpeg",
        "-f", "avfoundation",
        "-framerate", str(framerate),
        "-use_wallclock_as_timestamps", "1",
        "-capture_cursor", "1",
        "-vsync", "vfr",
        "-i", f"{device_index}:none",
        "-c:v", "libx264",
        "-g", "15",
        "-c:a", "aac",
        "-preset", "veryfast",
        "-crf", "17",
        "-copyts", "-muxdelay", "0",
        "-f", "segment",
        "-segment_time", str(segment_time),
        # "-r", str(framerate),
        output_path,
    ]
    # open stdin so we can send 'q' to stop cleanly
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def on_segment_finished(filename: str, gs_file_path: str, callback=None):
    upload_to_gcs_and_delete(filename, gs_file_path + filename.split("/")[-1])
    if callback:
        callback()


@app.command()
def run(
    username: Optional[str] = None,
    output_bucket: str = "induction-labs",
    video_segment_buffer_length: float = 30,
):
    if username is None:
        username = os.getenv("USER", "unknown_user")

    metadata = recording_metadata(
        username, video_segment_buffer_length
    )

    filename_session_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    random_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    file_path = f"action_capture/{username}/{filename_session_start_time}_{random_str}/"
    tmp_file_path = "tmp/" + file_path
    # Create the directory if it doesn't exist
    os.makedirs(tmp_file_path, exist_ok=True)

    gs_file_path = f"gs://{output_bucket}/{file_path}"

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

    print("[info] starting screen recording…")
    ffmpeg_proc = start_screen_record(
        tmp_file_path + "screen_capture_%06d.mp4",
        segment_time=video_segment_buffer_length,
    )
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

    try:
        file_in_progress = None
        for line in ffmpeg_proc.stderr:
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
        ffmpeg_proc.stdin.write("q\n")
        ffmpeg_proc.stdin.flush()

    ffmpeg_proc.wait()
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
