"""Application-wide logging setup.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from config.settings import APPLICATION_CONFIG, PATHS

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = PATHS["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    level_name = APPLICATION_CONFIG.get("logging", {}).get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("pdm")
    root.setLevel(level)
    root.propagate = False

    if not root.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        file_handler = RotatingFileHandler(
            log_dir / "application.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the 'pdm' root logger."""
    _configure_root_logger()
    return logging.getLogger(f"pdm.{name}")
