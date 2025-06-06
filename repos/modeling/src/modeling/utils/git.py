from __future__ import annotations

import logging
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def get_git_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("Failed to get git commit SHA. Returning 'unknown'.")
        return "unknown"


@lru_cache(maxsize=1)
def get_git_commit_sha_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("Failed to get short git commit SHA. Returning 'unknown'.")
        return "unknown"
