import logging
from pathlib import Path
from typing import Optional


def get_logger(name: Optional[str] = None, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_file: Optional path to a log file.
        level: Logging level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name if name else __name__)
    logger.setLevel(level)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(ch_formatter)
            logger.addHandler(fh)

    return logger

