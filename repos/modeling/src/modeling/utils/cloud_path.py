from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, computed_field, field_serializer


def path_validator(path: Any) -> Path:
    """
    Validate and convert a string or Path to a pathlib.Path.
    """
    if isinstance(path, str):
        return Path(path)
    elif isinstance(path, Path):
        return path
    else:
        raise ValueError(f"Expected str or Path, got {type(path)}")


class CloudPath(BaseModel):
    """
    A pathlib-like path for cloud and local URIs (e.g., s3://, gs://, file://)

    Attributes:
        cloud: one of 's3', 'gs', or 'file'
        path: a pathlib.Path representing the path portion
    """

    @field_serializer("path")
    def serialize_dt(self, path: Path, _info):
        return path.as_posix()

    class Cloud(StrEnum):
        S3 = "s3"
        GS = "gs"
        FILE = "file"

    cloud: Cloud
    path: Annotated[Path, BeforeValidator(path_validator)]

    @computed_field
    @property
    def uri(self) -> str:
        """
        Return the full URI as a string.
        """
        if self.cloud == "file":
            return self.path.as_posix()
        return f"{self.cloud}://{self.path.as_posix()}"

    @classmethod
    def from_str(cls: type[CloudPath], uri: str) -> CloudPath:
        """
        Parse a URI like 's3://bucket/folder/file.txt' or '/local/path'.
        """
        if "://" in uri:
            scheme, rest = uri.split("://", 1)
            cloud = CloudPath.Cloud(scheme)
            # for file:// URIs, Path accepts absolute and relative
            path = Path(rest)
        else:
            cloud = CloudPath.Cloud.FILE
            path = Path(uri)
        return cls(cloud=cloud, path=path)

    @property
    def bucket_and_path(self) -> tuple[str, Path]:
        """
        Extract the bucket name and path from the CloudPath.
        Returns a tuple of (bucket_name, path_in_bucket).
        """
        if self.cloud == CloudPath.Cloud.S3:
            # S3 URIs are of the form s3://bucket/path
            parts = self.path.parts
            return parts[0], Path(*parts[1:])
        elif self.cloud == CloudPath.Cloud.GS:
            # GCS URIs are of the form gs://bucket/path
            parts = self.path.parts
            return parts[0], Path(*parts[1:])
        else:
            raise ValueError(f"Unsupported cloud type: {self.cloud}")

    def __truediv__(self, other: str | Path) -> CloudPath:
        """
        Support the / operator for path joining.

        Examples:
            prefix = CloudPath.from_str('s3://mybucket/data')
            file = prefix / 'file.txt'  # s3://mybucket/data/file.txt
        """
        other_path = Path(other)
        new_path = self.path / other_path
        return CloudPath(cloud=self.cloud, path=new_path)

    def __str__(self) -> str:
        return self.uri

    def __repr__(self) -> str:
        return f"CloudPath(cloud={self.cloud!r}, path={str(self.path)!r})"


# Example usage
if __name__ == "__main__":
    cp = CloudPath.from_str("gs://mybucket/folder")
    print(cp)  # gs://mybucket/folder
    cp2 = cp / "subdir" / "file.txt"
    print(cp2.uri)  # gs://mybucket/folder/subdir/file.txt
    local = CloudPath.from_str("/tmp/data")
    print(local)  # /tmp/data
    print(local / "notes.md")  # /tmp/data/notes.md
