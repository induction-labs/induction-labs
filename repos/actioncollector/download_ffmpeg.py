#!/usr/bin/env python3
"""
Script to download and extract ffmpeg binary for macOS packaging.
This script downloads the latest ffmpeg binary from the official site
and extracts it to a bin/ directory for PyInstaller bundling.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


def download_ffmpeg_mac():
    """Download and extract ffmpeg binary for macOS"""
    print("Downloading ffmpeg for macOS...")

    # Create bin directory
    bin_dir = Path("bin")
    bin_dir.mkdir(exist_ok=True)

    # FFmpeg download URL for macOS (Intel)
    # Using a reliable source for ffmpeg binaries
    # ffmpeg_url = "https://evermeet.cx/ffmpeg/ffmpeg-7.1.1.zip"
    ffmpeg_url = (
        "https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffmpeg.zip"
    )

    try:
        # Download ffmpeg
        print(f"Downloading from {ffmpeg_url}")
        urllib.request.urlretrieve(ffmpeg_url, "ffmpeg.zip")

        # Extract ffmpeg
        print("Extracting ffmpeg...")
        with zipfile.ZipFile("ffmpeg.zip", "r") as zip_ref:
            zip_ref.extractall(".")

        # Move ffmpeg binary to bin directory
        if os.path.exists("ffmpeg"):
            shutil.move("ffmpeg", bin_dir / "ffmpeg")
            # Make executable
            os.chmod(bin_dir / "ffmpeg", 0o755)
            print(f"FFmpeg binary installed to {bin_dir / 'ffmpeg'}")
        else:
            print("Error: ffmpeg binary not found after extraction")
            return False

        # Clean up
        os.remove("ffmpeg.zip")

        # Verify installation
        result = subprocess.run(
            [str(bin_dir / "ffmpeg"), "-version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("FFmpeg installation verified successfully")
            return True
        else:
            print("Error: FFmpeg verification failed")
            return False

    except Exception as e:
        print(f"Error downloading ffmpeg: {e}")
        return False


def check_ffmpeg_exists():
    """Check if ffmpeg binary already exists"""
    bin_dir = Path("bin")
    ffmpeg_path = bin_dir / "ffmpeg"
    if not ffmpeg_path.exists():
        print(f"FFmpeg binary not found at {ffmpeg_path}, downloading...")
        return False
    # Check if the file is executable
    if ffmpeg_path.is_file() and os.access(ffmpeg_path, os.X_OK):
        print(f"FFmpeg binary exists and is executable at {ffmpeg_path}")
        return True
    result = subprocess.run(
        [str(ffmpeg_path), "-version"], capture_output=True, text=True
    )
    if result.returncode == 0:
        print("FFmpeg binary exists and is executable")
        return True
    # If the file exists but is not executable, we will download again
    print(f"FFmpeg binary exists but is not executable at {ffmpeg_path}")
    return False


if __name__ == "__main__":
    if not check_ffmpeg_exists():
        success = download_ffmpeg_mac()
        if not success:
            print("Failed to download ffmpeg")
            sys.exit(1)
    else:
        print("FFmpeg already available, skipping download")
