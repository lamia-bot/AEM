from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import platform
import sys
import threading
from pathlib import Path

APP_NAME = "MEA Analyzer"
APP_VERSION = "8.22"
BUILD_LABEL = "Build 22"


def log_root() -> Path:
    """Return a user-writable log directory without touching the analysis cache."""
    system = platform.system()
    if system == "Windows":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        root = base / APP_NAME / "Logs"
    elif system == "Darwin":
        root = Path.home() / "Library" / "Logs" / APP_NAME
    else:
        root = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "MEA_Analyzer" / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def current_log_path() -> Path:
    return log_root() / "MEA_Analyzer.log"


def configure_logging() -> Path:
    """Configure a small rotating local log and catch otherwise-unhandled crashes."""
    log_path = current_log_path()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicate handlers when app.py is re-run from an interactive shell.
    if not any(getattr(h, "baseFilename", None) == str(log_path) for h in root_logger.handlers):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=1_500_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root_logger.addHandler(handler)

    logging.getLogger(__name__).info(
        "%s %s starting | version=%s | python=%s | frozen=%s | platform=%s",
        APP_NAME,
        BUILD_LABEL,
        APP_VERSION,
        platform.python_version(),
        bool(getattr(sys, "frozen", False)),
        platform.platform(),
    )

    def _show_crash_message(message: str) -> None:
        # The packaged Windows build has no console. If Qt is alive, give the user
        # a useful location to send back rather than failing silently.
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            if QApplication.instance() is not None:
                QMessageBox.critical(
                    None,
                    "MEA Analyzer error",
                    f"MEA Analyzer encountered an unexpected error.\n\n"
                    f"A diagnostic log was saved to:\n{log_path}\n\n{message}",
                )
        except Exception:
            pass

    def _exception_hook(exc_type, exc_value, exc_traceback):
        if exc_type is KeyboardInterrupt:
            return sys.__excepthook__(exc_type, exc_value, exc_traceback)
        logging.getLogger("MEA Analyzer.crash").critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        _show_crash_message(str(exc_value))

    sys.excepthook = _exception_hook

    if hasattr(threading, "excepthook"):
        def _thread_exception_hook(args):
            logging.getLogger("MEA Analyzer.thread").critical(
                "Unhandled thread exception in %s",
                getattr(args.thread, "name", "unknown"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            _show_crash_message(str(args.exc_value))

        threading.excepthook = _thread_exception_hook

    return log_path
