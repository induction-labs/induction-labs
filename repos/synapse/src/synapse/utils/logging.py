from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

NODE_RANK = "NODE_RANK"
LOCAL_RANK = "LOCAL_RANK"


class _PrefixAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # prefix and original message are combined here
        import os

        rank = int(os.environ[LOCAL_RANK]) if LOCAL_RANK in os.environ else None
        node_rank = int(os.environ[NODE_RANK]) if NODE_RANK in os.environ else None
        prefix = ""
        if rank is not None:
            assert node_rank is not None, (
                f"{NODE_RANK} environment variable is not set. "
                "Make sure to run your script with torchrun or set NODE_RANK manually."
            )
            prefix = f"[{node_rank}:{rank}] "

        return f"{prefix}{msg}", kwargs


# https://chatgpt.com/c/68658dcb-54b0-8006-b406-ae483cadaedd
def configure_logging(
    name: str,
    level: int = logging.INFO,
    *,
    use_rich: bool = True,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.LoggerAdapter:
    """
    Create a logger that prints either plain-text or rich-formatted logs.

    Args:
        name: logger name
        level: log level
        use_rich: if True, use RichHandler for colored output
        fmt: classic logging format string (only used if use_rich=False)
        datefmt: datetime format string (only used if use_rich=False)

    Returns:
        Configured Logger instance.
    """

    # Include rank in logger name if it exists
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid double logs if root logger is configured

    # Remove any existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    if use_rich:
        # RichHandler automatically formats with colors, tracebacks, etc.
        console = Console(soft_wrap=False)  # disable wrapping
        console_handler = RichHandler(
            console=console,
            level=level,
            markup=True,  # enable markup in messages
            rich_tracebacks=True,  # pretty-print exceptions
            tracebacks_show_locals=False,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        # Include rank in format if it exists
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return _PrefixAdapter(logger)
