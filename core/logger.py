# transcribe/utilities/logger_setup.py
from __future__ import annotations

import io
import logging
from datetime import datetime
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from core.config import settings


@dataclass(frozen=True)
class LogConfig:
    log_dir: str
    base_filename: str = "transcribe"
    run_id: str = ""  # optional prefix, e.g. "2026-01-02_12-00-00___"
    level: int = logging.INFO
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 10
    console: bool = True
    capture_prints: bool = True
    capture_warnings: bool = True
    separate_error_log: bool = True
    logger_name: str = "kredo_transcribe"


class _StreamToLogger(io.TextIOBase):
    """
    File-like stream that redirects writes (print, third-party stdout/stderr)
    into a logger at a chosen level.
    """
    def __init__(self, logger: logging.Logger, level: int):
        super().__init__()
        self.logger = logger
        self.level = level
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0

        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self.logger.log(self.level, self._buf.rstrip())
        self._buf = ""


def setup_logger() -> logging.Logger:
    """
    Creates a robust logger:
    - INFO/DEBUG/etc
    - logs to a directory (created if missing)
    - rotates file when size exceeds max_bytes
    - optional separate error log
    - optional capture of print() + warnings + uncaught exceptions
    """

    cfg = LogConfig(
        log_dir=settings.TRANSCRIBE_LOGS_DIR,
        base_filename=settings.TRANSCRIBE_LOGS_PREF,
        level=logging.DEBUG,
        logger_name=settings.TRANSCRIBE_LOGGER_NAME
    )
    
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(cfg.logger_name)

    # Prevent duplicate handlers if setup_logger() is called multiple times.
    if getattr(logger, "_kredo_configured", False):
        return logger

    logger.setLevel(cfg.level)
    logger.propagate = False

    run_prefix = cfg.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{cfg.base_filename}_{run_prefix}".strip()

    # Main rotating log file (all levels)
    main_log_path = log_dir / f"{base}.log"
    file_handler = RotatingFileHandler(
        filename=str(main_log_path),
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(cfg.level)

    # Optional separate ERROR log
    err_handler: Optional[RotatingFileHandler] = None
    if cfg.separate_error_log:
        err_log_path = log_dir / f"{base}.error.log"
        err_handler = RotatingFileHandler(
            filename=str(err_log_path),
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        err_handler.setLevel(logging.ERROR)

    fmt = logging.Formatter(
        # fmt="%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s",
        fmt="%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(fmt)
    if err_handler:
        err_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    if err_handler:
        logger.addHandler(err_handler)

    # Console
    if cfg.console:
        console_handler = logging.StreamHandler(stream=sys.__stdout__)
        console_handler.setLevel(cfg.level)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    # Capture warnings -> logging
    if cfg.capture_warnings:
        logging.captureWarnings(True)

    # Redirect print()/stdout/stderr to logger (keeps your existing prints from utilities)
    # Many of your utility modules use print() heavily. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}
    if cfg.capture_prints:
        sys.stdout = _StreamToLogger(logger, logging.INFO)
        sys.stderr = _StreamToLogger(logger, logging.ERROR)

    # Catch uncaught exceptions
    def _excepthook(exc_type, exc, tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook

    setattr(logger, "_kredo_configured", True)
    logger.info("Logger initialized. log_dir=%s main=%s", str(log_dir), str(main_log_path))
    return logger


def _shutdown_logger(logger: logging.Logger) -> None:
    """
    Flush and close handlers + restore stdout/stderr.
    """
    try:
        for h in list(logger.handlers):
            try:
                h.flush()
                h.close()
            finally:
                logger.removeHandler(h)
    finally:
        # Restore std streams if we redirected them
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        setattr(logger, "_kredo_configured", False)

def shutdown_logger() -> None:
    global _LOGGER
    if _LOGGER is not None:
        _shutdown_logger(_LOGGER)
        _LOGGER = None


_LOGGER: logging.Logger | None = None



def get_logger(module_name: str | None = None) -> logging.Logger:
    """
    Returns configured logger. Safe to call from any module.
    If module_name is provided, returns a child logger:
      kredo_transcribe.<module_name>
    """
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = setup_logger()  # idempotent because of _kredo_configured guard

    if module_name:
        # Child logger inherits handlers via propagation to parent name.
        return logging.getLogger(f"{_LOGGER.name}.{module_name}")

    return _LOGGER


log = get_logger()
