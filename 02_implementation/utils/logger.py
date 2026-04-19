from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger.  Safe to call multiple times — handlers are only
    added once, so callers in different modules sharing the same name get the
    same logger without duplicate output.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
