"""Project logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "imputation_reco") -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    _LOGGERS[name] = logger
    return logger


def configure_logger(
    name: str = "imputation_reco",
    *,
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    logger = get_logger(name)
    numeric_level = getattr(logging, str(level).upper(), level)
    logger.setLevel(numeric_level)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_path.resolve())
        has_file_handler = any(
            isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == resolved
            for handler in logger.handlers
        )
        if not has_file_handler:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    handlers = list(logger.handlers)
    for handler in handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    name = logger.name
    if name in _LOGGERS:
        del _LOGGERS[name]


def log_kv(logger: logging.Logger, message: str, **kwargs: Any) -> None:
    if kwargs:
        payload = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        logger.info("%s | %s", message, payload)
    else:
        logger.info(message)
