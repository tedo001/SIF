"""System logging - the audit trail the Settings tab renders.

Everything the console does that matters operationally (model resolution, OCR
backends, training runs, batch imports, failures) goes through the standard
``logging`` module. This module wires that up three ways at once:

* a **rotating file** under ``logs/`` so a run can be inspected after the fact;
* a **ring buffer** the GUI polls, so the Settings tab can show live output
  without the log growing unbounded in memory; and
* the usual **stderr** stream for headless runs.

Handlers are installed once and are safe to call again - changing the level at
runtime (from the Settings tab) reuses the same handlers.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Deque, List, Optional

__all__ = ["LogEntry", "RingBufferHandler", "configure_logging", "get_ring_buffer",
           "set_level", "LOG_LEVELS", "log_file_path", "active_log_file"]

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
DEFAULT_LEVEL = "INFO"
DEFAULT_DIRECTORY = "logs"
FILE_NAME = "sif_console.log"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3
BUFFER_SIZE = 2000

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_state_lock = threading.Lock()
_ring: Optional["RingBufferHandler"] = None
_file_handler: Optional[logging.Handler] = None
_stream_handler: Optional[logging.Handler] = None


@dataclass(frozen=True)
class LogEntry:
    """One captured log record, flattened for display."""

    timestamp: str
    level: str
    logger: str
    message: str

    def as_row(self) -> List[str]:
        return [self.timestamp, self.level, self.logger, self.message]


class RingBufferHandler(logging.Handler):
    """Keeps the most recent records in memory for the Settings tab.

    A bounded ``deque`` means a long-running session cannot exhaust memory, and
    the lock makes it safe for worker threads to log while the GUI thread reads.
    """

    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        super().__init__()
        self._entries: Deque[LogEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._listener: Optional[Callable[[LogEntry], None]] = None

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102 - logging API
        try:
            entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created).strftime(_DATE_FORMAT),
                level=record.levelname,
                logger=record.name,
                message=record.getMessage(),
            )
        except Exception:  # pragma: no cover - never let logging break the app
            return
        with self._lock:
            self._entries.append(entry)
        listener = self._listener
        if listener is not None:
            try:
                listener(entry)
            except Exception:  # pragma: no cover - a bad listener must not cascade
                pass

    def entries(self, level: str = "DEBUG", limit: Optional[int] = None) -> List[LogEntry]:
        """Return buffered entries at or above ``level``, oldest first."""
        threshold = logging.getLevelName(level)
        threshold = threshold if isinstance(threshold, int) else logging.DEBUG
        with self._lock:
            items = list(self._entries)
        selected = [entry for entry in items
                    if logging.getLevelName(entry.level) >= threshold]
        return selected[-limit:] if limit else selected

    def clear(self) -> None:
        """Drop everything currently buffered."""
        with self._lock:
            self._entries.clear()

    def set_listener(self, listener: Optional[Callable[[LogEntry], None]]) -> None:
        """Install a callback invoked for every new record (used by the GUI)."""
        self._listener = listener


def log_file_path(directory: str = DEFAULT_DIRECTORY) -> str:
    """Absolute path of the rotating log file."""
    return os.path.abspath(os.path.join(directory, FILE_NAME))


def active_log_file() -> str:
    """Path of the log file currently being written, or '' when there is none.

    The handlers are installed once per process, so the first call to
    :func:`configure_logging` fixes the directory; this reports what that call
    actually settled on rather than what a later caller asked for.
    """
    handler = _file_handler
    return getattr(handler, "baseFilename", "") if handler is not None else ""


def configure_logging(level: str = DEFAULT_LEVEL, directory: str = DEFAULT_DIRECTORY,
                      to_stderr: bool = True) -> RingBufferHandler:
    """Install the handlers once and return the ring buffer.

    Safe to call repeatedly: later calls only adjust the level. The ``directory``
    of the first call wins for the whole process - see :func:`active_log_file`.
    """
    global _ring, _file_handler, _stream_handler

    with _state_lock:
        root = logging.getLogger()
        if _ring is None:
            formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

            _ring = RingBufferHandler()
            _ring.setFormatter(formatter)
            root.addHandler(_ring)

            try:
                os.makedirs(directory, exist_ok=True)
                handler = logging.handlers.RotatingFileHandler(
                    log_file_path(directory), maxBytes=MAX_BYTES,
                    backupCount=BACKUP_COUNT, encoding="utf-8")
                handler.setFormatter(formatter)
                root.addHandler(handler)
                _file_handler = handler
            except OSError as exc:  # pragma: no cover - read-only install directory
                root.warning("File logging disabled (%s)", exc)

            # Track the console handler explicitly: RotatingFileHandler is itself
            # a StreamHandler subclass, so sniffing by type would see the log file
            # and silently leave the console with no output at all.
            if to_stderr and _stream_handler is None:
                stream = logging.StreamHandler()
                stream.setFormatter(formatter)
                root.addHandler(stream)
                _stream_handler = stream

        set_level(level)
        return _ring


def set_level(level: str) -> None:
    """Change the root logging level at runtime."""
    resolved = getattr(logging, str(level).upper(), logging.INFO)
    logging.getLogger().setLevel(resolved)


def get_ring_buffer() -> Optional[RingBufferHandler]:
    """The installed ring buffer, or ``None`` before :func:`configure_logging`."""
    return _ring
