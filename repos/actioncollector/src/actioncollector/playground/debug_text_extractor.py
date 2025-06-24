#!/usr/bin/env python3
"""
Debug script to extract and print all text from action capture logs.

Reads action_capture_*.jsonl files from a gsutil bucket and prints all keystroke text.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

try:
    import gcsfs

    HAS_GCSFS = True
except ImportError:
    HAS_GCSFS = False


def extract_text_from_actions(actions: list[dict[str, Any]]) -> str:
    """Extract all text from keystroke actions."""
    keystrokes = []

    for action in actions:
        if (
            action.get("action", {}).get("action") == "key_button"
            and action.get("action", {}).get("is_down") is True
        ):
            key = action.get("action", {}).get("key", "")

            # Convert special keys to readable text
            if key == "space":
                keystrokes.append(" ")
            elif key == "enter" or key == "return":
                keystrokes.append("\n")
            elif key == "tab":
                keystrokes.append("\t")
            elif key not in [
                "shift",
                "cmd",
                "ctrl",
                "alt",
                "option",
                "backspace",
                "delete",
            ]:
                keystrokes.append(key)

    return "".join(keystrokes)


def debug_files(path: str):
    """Process all action_capture_*.jsonl files from GCS bucket or local directory."""
    if path.startswith("gs://"):
        if not HAS_GCSFS:
            print("Error: gcsfs not installed. Install with: pip install gcsfs")
            return
        debug_gcs_files(path)
    else:
        debug_local_files(path)


def debug_gcs_files(bucket_path: str):
    """Process files from GCS bucket."""
    fs = gcsfs.GCSFileSystem()

    # Remove gs:// prefix
    bucket_path = bucket_path[5:]

    try:
        files = fs.glob(f"{bucket_path}/action_capture_*.jsonl")
    except Exception as e:
        print(f"Error listing files from {bucket_path}: {e}")
        return

    for file_path in files:
        filename = file_path.split("/")[-1]

        print(f"\n{'=' * 60}")
        print(f"FILE: {filename}")
        print(f"{'=' * 60}")

        try:
            with fs.open(file_path) as f:
                actions = []
                for line in f:
                    line = line.strip()
                    if line:
                        actions.append(json.loads(line))

            process_actions(actions, filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


def debug_local_files(directory: str):
    """Process files from local directory."""
    pattern = os.path.join(directory, "action_capture_*.jsonl")
    files = glob.glob(pattern)

    if not files:
        print(f"No action_capture_*.jsonl files found in {directory}")
        return

    for file_path in files:
        filename = os.path.basename(file_path)

        print(f"\n{'=' * 60}")
        print(f"FILE: {filename}")
        print(f"{'=' * 60}")

        try:
            with open(file_path) as f:
                actions = []
                for line in f:
                    line = line.strip()
                    if line:
                        actions.append(json.loads(line))

            process_actions(actions, filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


def process_actions(actions: list[dict[str, Any]], filename: str):
    """Extract and print text from actions."""
    text = extract_text_from_actions(actions)

    if text.strip():
        print(f"TEXT CONTENT ({len(text)} characters):")
        print("-" * 40)
        print(repr(text))  # Print with escape sequences visible
        print("-" * 40)
        print("READABLE TEXT:")
        print(text)
    else:
        print("No text content found (only mouse/modifier keys)")


def main():
    parser = argparse.ArgumentParser(
        description="Debug script to extract all text from action capture logs"
    )
    parser.add_argument(
        "path", help="GCS bucket path (gs://bucket-name/folder) or local directory path"
    )

    args = parser.parse_args()

    debug_files(args.path)


if __name__ == "__main__":
    main()
