import os
from contextlib import contextmanager


@contextmanager
def temp_env(environ: dict[str, str | None]):
    """
    Context manager that temporarily sets environment variables.
    It restores the original values after the context is exited.
    """
    # Store the original values

    original_environ = {key: os.environ.get(key, None) for key in environ.keys()}

    try:
        # Set the new path
        for key, value in environ.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        # Restore the original values
        for key, original_value in original_environ.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
