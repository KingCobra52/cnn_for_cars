"""Logging setup shared by the CLI and every entry point."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install a single stderr handler on the root logger.

    Idempotent: calling it twice does not duplicate handlers, which matters when
    the CLI, Hydra and a notebook all try to configure logging in one process.

    Args:
        level: Logging level, as a level number or its name.
    """
    root = logging.getLogger()
    if any(getattr(h, "_carvision", False) for h in root.handlers):
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handler._carvision = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Args:
        name: Usually ``__name__``.
    """
    return logging.getLogger(name)
