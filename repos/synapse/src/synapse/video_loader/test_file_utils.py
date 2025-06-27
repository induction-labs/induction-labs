from __future__ import annotations

import pytest

from synapse.video_loader.file_utils import validate_sequential_files


def test_validate_sequential_files_valid_sequence():
    """Test with valid sequential files."""
    file_paths = [
        "/path/screen_capture_000000.mp4",
        "/path/screen_capture_000001.mp4",
        "/path/screen_capture_000002.mp4",
    ]
    result = validate_sequential_files(file_paths, "screen_capture_")
    expected = [
        "/path/screen_capture_000000.mp4",
        "/path/screen_capture_000001.mp4",
        "/path/screen_capture_000002.mp4",
    ]
    assert result == expected


def test_validate_sequential_files_missing_index():
    """Test with missing index in sequence."""
    file_paths = [
        "/path/screen_capture_000000.mp4",
        "/path/screen_capture_000002.mp4",  # Missing 000001
        "/path/screen_capture_000003.mp4",
    ]
    with pytest.raises(AssertionError, match="Missing video file indices: \\[1\\]"):
        validate_sequential_files(file_paths, "screen_capture_")


def test_validate_sequential_files_unordered_input():
    """Test with unordered input files."""
    file_paths = [
        "/path/screen_capture_000002.mp4",
        "/path/screen_capture_000000.mp4",
        "/path/screen_capture_000001.mp4",
    ]
    result = validate_sequential_files(file_paths, "screen_capture_")
    expected = [
        "/path/screen_capture_000000.mp4",
        "/path/screen_capture_000001.mp4",
        "/path/screen_capture_000002.mp4",
    ]
    assert result == expected
