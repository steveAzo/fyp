import logging
from pathlib import Path

_LOG_FILE = Path(__file__).resolve().parent.parent / "results" / "run.log"

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Console scrollback is easy to lose on a multi-hour run (terminal
        # buffer limits, accidental clears, truncated copy-paste). Persist
        # everything, including full tracebacks, so a crash is diagnosable
        # after the fact.
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)
    return logger
