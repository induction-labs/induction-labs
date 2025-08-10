import shutil
import tempfile
from contextvars import ContextVar, Token
from pathlib import Path

from synapse.utils.logging import configure_logging

logger = configure_logging(__name__)

_tmp_dir_context: ContextVar[Path] = ContextVar("_tmp_dir_context")


class TmpDirContext:
    def __init__(
        self,
    ):
        self._token: Token | None = None
        self.tmpdir: Path | None = None

    def __enter__(self):
        # set() returns a Token we can use to restore the old value
        self.tmpdir = Path(tempfile.mkdtemp())
        logger.debug(f"Created temporary directory: {self.tmpdir}")
        self._token = _tmp_dir_context.set(self.tmpdir)
        return self, self.tmpdir

    def __exit__(self, exc_type, exc, tb):
        # restore the previous value (or clear it if there wasn't one)
        assert self._token is not None
        assert self.tmpdir is not None, "Temporary directory was not set"
        logger.debug(f"Cleaning up temporary directory: {self.tmpdir}")
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        logger.debug(f"Temporary directory cleaned up: {self.tmpdir}")
        _tmp_dir_context.reset(self._token)


def use_tmp_dir() -> Path:
    """
    Context manager to temporarily set the _tmp_dir_context variable.
    This is useful for ensuring that temporary directories are used correctly
    within a specific context.
    """
    try:
        tmpdir = _tmp_dir_context.get()
        logger.debug(f"Using temporary directory: {tmpdir}")
        return tmpdir
    except LookupError as e:
        logger.debug("No TmpDirContext is active")
        raise RuntimeError("use_tmp_dir(): no TmpDirContext is active") from e
