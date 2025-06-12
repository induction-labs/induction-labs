from __future__ import annotations

import time
from contextlib import contextmanager


# Alternative function-based approach using @contextmanager
@contextmanager
def elapsed_timer():
    start_time = time.perf_counter()

    def get_elapsed():
        return time.perf_counter() - start_time

    yield get_elapsed
