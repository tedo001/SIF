"""The audit trail - what the console did, and who asked for it.

This is not the debug log. :mod:`sif.logging_setup` writes diagnostics: it is
verbose, it carries third-party chatter, and it rotates away after a few
megabytes, which is exactly right for finding out why something failed this
morning and exactly wrong for answering "who cleared that report, and when?"
six months later.

So the audit trail is separate, and deliberately dull:

* **Append-only, one JSON object per line.** A line is written and never
  rewritten, so a corrupt or truncated tail costs at most the last entry rather
  than the file. Nothing here rotates on its own.
* **Two kinds of entry.** ``system`` records what the software did on this
  machine - started, loaded a model, fetched OCR models, checked for an update.
  ``functionality`` records what the console was *asked* to do and what came
  back - reports analysed, documents read, a model trained, a decision recorded,
  something exported. An auditor reads the second kind; support reads the first.
* **Every entry names its actor.** The operating-system user, plus the reviewer
  name when the operator has given one. A trail that cannot say who did
  something is not a trail.
* **Never fatal.** An unwritable location degrades to in-memory only and says so
  once; recording an event must not be able to interrupt the work being audited.

The file lives beside the settings, so it follows the operator's profile rather
than the installation directory: ``%APPDATA%\\SIF Insight Console\\audit.jsonl``
on Windows, ``~/.config/SIF Insight Console/audit.jsonl`` elsewhere.
"""

from __future__ import annotations

import csv
import getpass
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from . import prefs

__all__ = ["AuditEntry", "AuditLog", "SYSTEM", "FUNCTIONALITY", "CATEGORIES",
           "audit_file_path"]

LOGGER = logging.getLogger(__name__)

#: What the software did by itself.
SYSTEM = "system"
#: What an operator asked for, and what came back.
FUNCTIONALITY = "functionality"
CATEGORIES = (SYSTEM, FUNCTIONALITY)

FILE_NAME = "audit.jsonl"
#: Read back at most this many entries for the interface; the file keeps them all.
DEFAULT_LIMIT = 500


def audit_file_path() -> str:
    """Where the trail is written for this operator."""
    return os.path.join(prefs.config_directory(), FILE_NAME)


def _actor() -> str:
    """The operating-system user, as far as it can be determined."""
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no controlling terminal, odd container
        return os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"


@dataclass
class AuditEntry:
    """One line of the trail."""

    at: str
    category: str
    action: str
    actor: str = ""
    reviewer: str = ""
    version: str = ""
    detail: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"at": self.at, "category": self.category, "action": self.action,
                "actor": self.actor, "reviewer": self.reviewer,
                "version": self.version, "detail": dict(self.detail)}

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "AuditEntry":
        detail = payload.get("detail")
        return cls(
            at=str(payload.get("at", "")),
            category=str(payload.get("category", SYSTEM)),
            action=str(payload.get("action", "")),
            actor=str(payload.get("actor", "")),
            reviewer=str(payload.get("reviewer", "")),
            version=str(payload.get("version", "")),
            detail=dict(detail) if isinstance(detail, dict) else {},
        )

    @property
    def summary(self) -> str:
        """The detail as one readable line, for the table."""
        return "  ·  ".join(f"{key}: {value}" for key, value in self.detail.items())


class AuditLog:
    """Append-only record of everything worth answering for later."""

    def __init__(self, path: str = "", version: str = "") -> None:
        self.path = path or audit_file_path()
        self.version = version
        self.reviewer = ""
        self._lock = threading.Lock()
        self._memory: List[AuditEntry] = []
        #: False once a write has failed, so the interface can say the trail is
        #: only in memory rather than letting anyone believe it reached disk.
        self.writable = True

    # -- recording ---------------------------------------------------------

    def record(self, category: str, action: str, **detail: object) -> AuditEntry:
        """Append one entry. Never raises - auditing must not break the work."""
        entry = AuditEntry(
            at=datetime.now().isoformat(timespec="seconds"),
            category=category if category in CATEGORIES else SYSTEM,
            action=action,
            actor=_actor(),
            reviewer=self.reviewer,
            version=self.version,
            detail={key: value for key, value in detail.items() if value is not None},
        )
        with self._lock:
            self._memory.append(entry)
            self._append(entry)
        return entry

    def system(self, action: str, **detail: object) -> AuditEntry:
        """Record something the software did on this machine."""
        return self.record(SYSTEM, action, **detail)

    def functionality(self, action: str, **detail: object) -> AuditEntry:
        """Record something an operator asked for, and what came back."""
        return self.record(FUNCTIONALITY, action, **detail)

    def _append(self, entry: AuditEntry) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())     # a crash must not lose the last entry
            self.writable = True
        except Exception as exc:  # noqa: BLE001 - a read-only profile is not fatal
            if self.writable:                 # complain once, not once per event
                LOGGER.warning("Audit trail is memory-only (%s): %s", self.path, exc)
            self.writable = False

    # -- reading -----------------------------------------------------------

    def entries(self, category: str = "", limit: int = DEFAULT_LIMIT) -> List[AuditEntry]:
        """The most recent entries, newest first, optionally by category."""
        rows = self._read()
        if category in CATEGORIES:
            rows = [entry for entry in rows if entry.category == category]
        rows.reverse()
        return rows[:limit] if limit else rows

    def rows(self, category: str = "", limit: int = DEFAULT_LIMIT) -> List[Dict[str, object]]:
        """The same, as table payloads.

        ``when`` is the same instant as ``at``, with the ISO ``T`` replaced by a
        space so it reads like the log beside it. ``at`` stays machine-readable.
        """
        return [{**entry.to_dict(), "summary": entry.summary,
                 "when": entry.at.replace("T", " ")}
                for entry in self.entries(category, limit)]

    def _read(self) -> List[AuditEntry]:
        """Everything on disk, plus anything that never reached it."""
        found: List[AuditEntry] = []
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        found.append(AuditEntry.from_dict(json.loads(line)))
                    except Exception:  # noqa: BLE001 - one bad line, not the file
                        continue
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not read the audit trail (%s)", exc)
        if not self.writable:
            found.extend(self._memory)
        return found

    def counts(self) -> Dict[str, int]:
        """How much of each kind the trail holds."""
        rows = self._read()
        return {
            "total": len(rows),
            SYSTEM: sum(1 for entry in rows if entry.category == SYSTEM),
            FUNCTIONALITY: sum(1 for entry in rows if entry.category == FUNCTIONALITY),
        }

    def export_csv(self, path: str, category: str = "") -> str:
        """Write the trail out for an auditor; returns the path."""
        columns = ["at", "category", "action", "actor", "reviewer", "version", "summary"]
        entries = self.entries(category, limit=0)
        entries.reverse()                      # oldest first reads as a history
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for entry in entries:
                payload = entry.to_dict()
                payload["summary"] = entry.summary
                writer.writerow({key: payload.get(key, "") for key in columns})
        LOGGER.info("Exported %d audit entr(ies) to %s", len(entries), path)
        return path


def summarise(entries: Sequence[AuditEntry]) -> Dict[str, int]:
    """Count entries by action - what this machine actually spends its time on."""
    tally: Dict[str, int] = {}
    for entry in entries:
        tally[entry.action] = tally.get(entry.action, 0) + 1
    return dict(sorted(tally.items(), key=lambda item: item[1], reverse=True))


_ACTIVE: Optional[AuditLog] = None


def active(version: str = "") -> AuditLog:
    """The process-wide trail, created on first use."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = AuditLog(version=version)
    return _ACTIVE
