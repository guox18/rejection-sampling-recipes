"""Logging helpers for pipeline and stages."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Tuple

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(processName)s/%(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOGGING_CONFIGURED = False


def _pick_log_dir(user_dir: Optional[str] = None) -> Path:
    """Pick a writable log directory with fallbacks."""
    candidates = []

    if user_dir:
        candidates.append(Path(user_dir))

    env_dir = os.getenv("LOG_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    # repo-local logs directory
    candidates.append(Path.cwd() / "logs")
    # final fallback
    candidates.append(Path("/tmp/rejection-sampling-recipes-logs"))

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            continue

    # As a last resort, current working directory
    return Path.cwd()


def setup_logging(
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = "pipeline.log",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    role: str = "driver",
) -> Tuple[logging.Logger, Path]:
    """
    Configure root logging with console + rotating file handlers.

    Args:
        log_dir: preferred directory; env LOG_DIR overrides if set.
        log_filename: base filename. Workers will automatically append PID.
        console_level: logging level for stdout.
        file_level: logging level for file handler.
        role: 'driver' or 'worker' (affects filename).

    Returns:
        (root_logger, log_path)
    """
    global _LOGGING_CONFIGURED
    root = logging.getLogger()

    # Environment overrides
    console_level = os.getenv("LOG_CONSOLE_LEVEL", console_level)
    file_level = os.getenv("LOG_FILE_LEVEL", file_level)

    if _LOGGING_CONFIGURED and root.handlers:
        # Already configured in this process.
        log_path = getattr(root, "_log_file_path", None)
        return root, Path(log_path) if log_path else Path()

    log_dir_path = _pick_log_dir(log_dir)

    filename = log_filename or "pipeline.log"
    if role == "worker":
        # isolate worker logs to avoid handler contention across processes
        filename = f"pipeline_worker_{os.getpid()}.log"

    log_path = log_dir_path / filename

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # File handler with rotation
    max_bytes = int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10 MB
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)

    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Keep a reference for quick retrieval
    root._log_file_path = str(log_path)
    _LOGGING_CONFIGURED = True

    root.debug(
        "Logging configured",
        extra={"log_path": str(log_path), "role": role, "pid": os.getpid()},
    )
    return root, log_path
