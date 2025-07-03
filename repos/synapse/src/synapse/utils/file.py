from __future__ import annotations

import gcsfs


def write_to_path(path: str, data: str) -> None:
    """
    Write data to a specified path. If the path starts with 'gs://', it writes to Google Cloud Storage.
    Otherwise, it writes to the local filesystem.

    Args:
        path (str): The path where the data should be written.
        data (str): The data to write.

    Raises:
        ValueError: If the path is not a valid GCS or local path.
    """
    if path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "w") as f:
            f.write(data)
    else:
        with open(path, "w") as f:
            f.write(data)


def read_from_path(path: str) -> str:
    """
    Read data from a specified path. If the path starts with 'gs://', it reads from Google Cloud Storage.
    Otherwise, it reads from the local filesystem.

    Args:
        path (str): The path from which to read the data.

    Returns:
        str: The data read from the specified path.
    """
    if path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "r") as f:
            s = f.read()
            if isinstance(s, str):
                return s
            return s.decode("utf-8")
    else:
        with open(path) as f:
            return f.read()
