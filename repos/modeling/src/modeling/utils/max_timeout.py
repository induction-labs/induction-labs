import asyncio
from collections.abc import Awaitable
from datetime import timedelta
from typing import TypeVar

T = TypeVar("T")


class MaxTimeoutError(asyncio.TimeoutError):
    """Raised when the coroutine runs longer than the allowed timeout."""


async def max_timeout(
    promise: Awaitable[T],
    timeout: timedelta = timedelta(seconds=60),
    err_msg: str = "Operation exceeded maximum timeout",
) -> T:
    """
    Decorator that enforces ``timeout`` seconds on every invocation of the
    wrapped coroutine.

    Usage
    -----
    @max_timeout(5.0)
    async def do_work(x):
        ...

    await do_work(123)  # Raises MaxTimeoutError if it runs longer than 5 s.
    """
    if timeout.total_seconds() <= 0:
        raise ValueError("timeout must be > 0 seconds")

    try:
        return await asyncio.wait_for(promise, timeout.total_seconds())
    except TimeoutError as exc:
        raise MaxTimeoutError(f"[{timeout=}] {err_msg}") from exc
