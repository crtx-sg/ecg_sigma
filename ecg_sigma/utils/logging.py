"""Centralised logging configuration.

The pipeline writes a single line per significant action — record loaded,
event extracted, file written — so that a long batch is auditable from
the log alone.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_DEFAULT_FMT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
_CONFIGURED = False


def configure_logging(level: str = "INFO", fmt: str = _DEFAULT_FMT) -> None:
    """Configure the root logger once. Subsequent calls are no-ops.

    Idempotent so that importing the package or calling the script multiple
    times does not duplicate handlers (which would cause duplicate log lines).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with the package's preferred prefix."""
    return logging.getLogger(name or "ecg_sigma")
