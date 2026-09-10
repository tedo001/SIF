"""Stage 6b - human review queue.

The architecture routes results two ways: to automated pattern detection, and to
a person. This module decides which reports a person must look at, and says why,
so the queue is a work list rather than a dump of everything the model touched.

Four triggers, in priority order:

1. **Disagreement** - the lexical rules and the semantic model reached different
   conclusions. One of them is wrong and only a human can say which. This
   trigger needs both opinions to exist: when the semantic layer is inactive
   (offline fallback, or a ranking too flat to be informative) there is nothing
   to disagree with, and the report is judged on the other three triggers.
2. **Critical risk** - anything scoring in the top band gets verified before it
   drives an intervention.
3. **Thin evidence** - the classification rests on very little extracted text.
4. **Unclassified with energy** - high energy present but no rule matched, which
   usually means vocabulary the system has never seen.
5. **Energy with no barrier found** - a high-energy source and a Life-Saving Rule
   were both recognised, but no failed barrier was. P(SIF) is energy x barrier, so
   such a report scores zero and would otherwise pass through the system unseen.
   Either the barrier genuinely held, which a person can confirm in seconds, or it
   failed in words the vocabulary does not yet carry - and that is exactly the
   report worth having.
6. **SIF potential, any other reason** - a catch-all. Every trigger above is a
   specific reason a report needs a person; this one is the guarantee behind all
   of them: no report the engine confirms as SIF-potential is ever closed
   without one. A confirmed finding outside the Critical risk band (High or
   Medium, with clean extraction and no disagreement) used to fall through every
   check above and reach nobody - exactly the silent failure this module's own
   contract rules out. If the report's own wording downplays the finding
   ("nothing serious", "no big deal"), the reason says so: the engine reads the
   extracted facts, not the tone, and a reviewer skimming for alarming language
   is precisely who misses that.

When a trained XGBoost model is attached (see :mod:`sif.mlops`), its verdict is a
third opinion: where it contradicts the pipeline, the report is queued as a
**model disagreement**. Those are the highest-value rows to label, because each
one either corrects the rules or corrects the model.

The other half of this module is the outcome. :class:`DecisionLog` records what
the expert decided, persists it, and reads it back two ways: as the set of
reports that no longer need reviewing, and as :meth:`DecisionLog.labels_for` -
the real labels that replace the pipeline's own verdicts when the model is
retrained. That is the loop closing. Nothing here closes a report on its own;
the engine queues and explains, a person decides.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Container, Dict, List, Optional, Sequence

from . import prefs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pipeline import PipelineResult

__all__ = ["ReviewItem", "ReviewQueue", "ReviewDecision", "DecisionLog",
           "fingerprint", "DECISIONS", "DECISION_LABELS", "DECISION_SHORT"]

LOGGER = logging.getLogger(__name__)

PRIORITY_ORDER = {"Disagreement": 0, "Model disagreement": 1, "LLM disagreement": 2,
                  "Critical risk": 3, "Thin evidence": 4, "Unclassified exposure": 5,
                  "Energy, no barrier": 6, "SIF potential": 7}


@dataclass
class ReviewItem:
    """One report queued for a human, with the reason and its rank."""

    index: int
    reference: str
    trigger: str
    reason: str
    risk_score: float
    sif_potential: bool
    summary: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index, "reference": self.reference, "trigger": self.trigger,
            "reason": self.reason, "risk_score": self.risk_score,
            "sif_potential": self.sif_potential, "summary": self.summary,
        }


class ReviewQueue:
    """Selects and ranks the reports that need human verification."""

    CONFIDENCE_FLOOR = 0.35
    CRITICAL_BAND = "Critical"

    def build(self, results: Sequence["PipelineResult"],
              skip: Optional[Container[str]] = None) -> List[ReviewItem]:
        """Return the review queue for ``results``, highest risk first.

        ``skip`` is a set of :func:`fingerprint` values to leave out - the reports
        an expert has already decided. Omit it and the queue is everything that
        meets a trigger, which is what a caller with no decision log wants.
        """
        items: List[ReviewItem] = []
        for position, result in enumerate(results, start=1):
            trigger, reason = self.classify(result)
            if trigger is None:
                continue
            if skip is not None and fingerprint(result) in skip:
                continue
            items.append(ReviewItem(
                index=position,
                reference=result.reference or f"#{position}",
                trigger=trigger,
                reason=reason,
                risk_score=result.risk_score,
                sif_potential=result.sif_potential,
                summary=result.raw_text[:160],
            ))
        items.sort(key=lambda item: (PRIORITY_ORDER.get(item.trigger, 9), -item.risk_score))
        return items

    @staticmethod
    def classify(result: "PipelineResult"):
        """Return ``(trigger, reason)`` for one result, or ``(None, "")``.

        Called per report by the pipeline so every row carries its own review
        flag, and again by :meth:`build` when the queue is assembled.
        """
        if result.semantic_active and result.lexical_flag != result.semantic_flag:
            agree, disagree = (("lexical rules", "semantic model")
                               if result.lexical_flag else ("semantic model", "lexical rules"))
            return "Disagreement", (
                f"{agree} flagged SIF potential, {disagree} did not "
                f"(P(SIF) {result.p_sif:.2f})")
        if result.ml_active and result.ml_flag != result.sif_potential:
            direction = "above" if result.ml_flag else "below"
            return "Model disagreement", (
                f"the trained model puts P(SIF) at {result.ml_probability:.2f}, "
                f"{direction} the pipeline verdict - check which is right and label it")
        if result.llm_active and result.llm_flag != result.sif_potential:
            side = "flags" if result.llm_flag else "clears"
            return "LLM disagreement", (
                f"the local model {side} this report against the pipeline verdict"
                + (f" - {result.llm_rationale}" if result.llm_rationale else ""))
        if result.risk_band == ReviewQueue.CRITICAL_BAND:
            return "Critical risk", (
                f"risk {result.risk_score:.0f}/100 - verify before it drives an intervention"
                + ReviewQueue._minimizing_note(result))
        if result.confidence < ReviewQueue.CONFIDENCE_FLOOR:
            return "Thin evidence", (
                f"extraction confidence {result.confidence:.2f} - narrative may be too "
                "sparse to classify")
        if result.high_energy and result.iogp_rule.startswith("Unclassified"):
            return "Unclassified exposure", (
                "high-energy source present but no Life-Saving Rule matched - likely "
                "unseen vocabulary")
        if result.high_energy and not result.barrier_failed and not result.sif_potential:
            return "Energy, no barrier", (
                f"{result.energy_source or 'a high-energy source'} was recognised under "
                f"{result.iogp_rule}, but no failed barrier was - confirm the barrier "
                "held, or name the one that did not")
        if result.sif_potential:
            # The catch-all: every confirmed finding reaches a person, even one
            # that is High or Medium band with clean extraction and no
            # disagreement - the combination none of the checks above catch.
            return "SIF potential", (
                f"engine confirms fatal potential at risk {result.risk_score:.0f}/100 "
                f"(band {result.risk_band}) under {result.iogp_rule} - verify before "
                "it is filed away" + ReviewQueue._minimizing_note(result))
        return None, ""

    @staticmethod
    def _minimizing_note(result: "PipelineResult") -> str:
        """Appended to a reason when the report's own wording undersells it.

        The engine scores energy x barrier, not tone, so a report can read
        "nothing serious, no injury" while describing a live cable with no
        LOTO applied. Naming that explicitly is the point: a reviewer skimming
        for alarming language is exactly who would otherwise miss it.
        """
        if not getattr(result, "minimizing_language", False):
            return ""
        return (" - note: the report's own wording downplays this "
                "('minor', 'nothing serious' or similar); the finding rests on "
                "the extracted facts, not the tone")


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

#: What a reviewer can say. ``unclear`` is deliberately not a label: a report the
#: expert cannot call is not training data, it is a request for more information.
DECISIONS: Dict[str, Optional[int]] = {"confirmed": 1, "rejected": 0, "unclear": None}

DECISION_LABELS = {
    "confirmed": "SIF potential confirmed",
    "rejected": "Not SIF potential",
    "unclear": "Unclear - more information needed",
}

#: The same three outcomes in a queue-column's worth of characters.
DECISION_SHORT = {"confirmed": "confirmed", "rejected": "not SIF", "unclear": "unclear"}

#: Where decisions live, under the per-user configuration directory.
DECISION_FILE = "review_decisions.json"
SCHEMA = 1


def fingerprint(result: "PipelineResult") -> str:
    """A stable identity for one report, independent of its position.

    The review queue is rebuilt on every analysis, and its ordering changes as
    the corpus grows, so a decision cannot be keyed on a row number. It is keyed
    on the report itself: the reference plus its narrative, whitespace-normalised
    so that re-ingesting the same document through a different path (CSV today,
    OCR tomorrow) still resolves to the same report.
    """
    reference = (getattr(result, "reference", "") or "").strip().lower()
    text = " ".join((getattr(result, "raw_text", "") or "").lower().split())
    return hashlib.sha256(f"{reference}\x00{text}".encode("utf-8")).hexdigest()[:16]


@dataclass
class ReviewDecision:
    """One expert verdict on one report.

    Everything needed to defend the decision later is on the record: who decided,
    when, what the engine had said, and why the report was queued in the first
    place. ``label`` is what training consumes; it is ``None`` for ``unclear``.
    """

    fingerprint: str
    reference: str
    decision: str
    reviewer: str = ""
    note: str = ""
    trigger: str = ""
    engine_verdict: bool = False
    risk_score: float = 0.0
    decided_at: str = ""

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"unknown decision {self.decision!r}; "
                             f"expected one of {sorted(DECISIONS)}")
        if not self.decided_at:
            self.decided_at = datetime.now().isoformat(timespec="seconds")

    @property
    def label(self) -> Optional[int]:
        """1, 0, or None when the reviewer could not call it."""
        return DECISIONS[self.decision]

    @property
    def overturns_engine(self) -> bool:
        """True when the expert disagreed with what the engine concluded."""
        return self.label is not None and bool(self.label) != self.engine_verdict

    def to_dict(self) -> Dict[str, object]:
        return {
            "fingerprint": self.fingerprint, "reference": self.reference,
            "decision": self.decision, "label": self.label, "reviewer": self.reviewer,
            "note": self.note, "trigger": self.trigger,
            "engine_verdict": self.engine_verdict, "risk_score": self.risk_score,
            "decided_at": self.decided_at, "overturns_engine": self.overturns_engine,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ReviewDecision":
        return cls(
            fingerprint=str(payload.get("fingerprint", "")),
            reference=str(payload.get("reference", "")),
            decision=str(payload.get("decision", "unclear")),
            reviewer=str(payload.get("reviewer", "")),
            note=str(payload.get("note", "")),
            trigger=str(payload.get("trigger", "")),
            engine_verdict=bool(payload.get("engine_verdict", False)),
            risk_score=float(payload.get("risk_score", 0.0) or 0.0),
            decided_at=str(payload.get("decided_at", "")),
        )


class DecisionLog:
    """The expert's decisions: appended, persisted, and read back as labels.

    Append-only. Changing your mind about a report appends a second entry that
    supersedes the first, and the trail stays readable - which is the point, since
    the interesting question six months later is not only what was decided but
    what was decided *first*. :meth:`undo` is the one exception, for the misclick.

    Every write reaches disk immediately: an expert who worked a queue for an
    hour must not lose it to a crash.
    """

    def __init__(self, path: str = "") -> None:
        self.path = path or os.path.join(prefs.config_directory(), DECISION_FILE)
        self.entries: List[ReviewDecision] = []
        #: False once a write has failed, so the interface can say the decisions
        #: are only in memory rather than letting an expert believe otherwise.
        self.saved = True

    # -- persistence -------------------------------------------------------

    def load(self) -> "DecisionLog":
        """Read the file; anything unreadable reads as empty and is logged."""
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return self
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not stop start-up
            LOGGER.warning("Ignoring unreadable review decisions (%s)", exc)
            return self
        raw = payload.get("decisions", []) if isinstance(payload, dict) else payload
        for item in raw if isinstance(raw, list) else []:
            try:
                self.entries.append(ReviewDecision.from_dict(item))
            except Exception as exc:  # noqa: BLE001 - skip the bad row, keep the rest
                LOGGER.warning("Skipping malformed review decision (%s)", exc)
        LOGGER.info("Loaded %d review decision(s) from %s", len(self.entries), self.path)
        return self

    def save(self) -> bool:
        """Write the log atomically; returns False when the location is read-only."""
        payload = {"schema": SCHEMA, "saved_at": datetime.now().isoformat(timespec="seconds"),
                   "decisions": [entry.to_dict() for entry in self.entries]}
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            temporary = f"{self.path}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            os.replace(temporary, self.path)      # atomic: never a half-written log
            self.saved = True
            return True
        except Exception as exc:  # noqa: BLE001 - a read-only home is not fatal
            LOGGER.warning("Could not save review decisions (%s)", exc)
            self.saved = False
            return False

    # -- recording ---------------------------------------------------------

    def record(self, result: "PipelineResult", decision: str, reviewer: str = "",
               note: str = "") -> ReviewDecision:
        """Record one decision against ``result`` and persist immediately."""
        entry = ReviewDecision(
            fingerprint=fingerprint(result),
            reference=getattr(result, "reference", "") or "",
            decision=decision,
            reviewer=reviewer.strip(),
            note=note.strip(),
            trigger=getattr(result, "review_trigger", "") or "",
            engine_verdict=bool(getattr(result, "sif_potential", False)),
            risk_score=float(getattr(result, "risk_score", 0.0) or 0.0),
        )
        self.entries.append(entry)
        self.save()
        LOGGER.info("Review decision %s on %s by %s%s", entry.decision,
                    entry.reference or entry.fingerprint, entry.reviewer or "unnamed",
                    " (overturns the engine)" if entry.overturns_engine else "")
        return entry

    def undo(self) -> Optional[ReviewDecision]:
        """Withdraw the most recent decision - the misclick, not a change of mind."""
        if not self.entries:
            return None
        entry = self.entries.pop()
        self.save()
        LOGGER.info("Withdrew the %s decision on %s", entry.decision,
                    entry.reference or entry.fingerprint)
        return entry

    # -- reading -----------------------------------------------------------

    def current(self) -> Dict[str, ReviewDecision]:
        """The standing decision per report: the newest entry wins."""
        latest: Dict[str, ReviewDecision] = {}
        for entry in self.entries:
            latest[entry.fingerprint] = entry
        return latest

    def for_result(self, result: "PipelineResult") -> Optional[ReviewDecision]:
        """The standing decision on one report, if it has been reviewed."""
        return self.current().get(fingerprint(result))

    def decided(self) -> set:
        """Fingerprints that carry a standing decision of any kind."""
        return set(self.current())

    def labelled(self) -> set:
        """Fingerprints a reviewer actually called, so training can use them."""
        return {key for key, entry in self.current().items() if entry.label is not None}

    def labels_for(self, results: Sequence["PipelineResult"]):
        """Return ``(results, labels)`` for the reports an expert has called.

        This is what turns the review queue into training data: reports the
        expert marked *unclear*, and reports never reviewed, are left out rather
        than guessed at.
        """
        standing = self.current()
        chosen: List["PipelineResult"] = []
        labels: List[int] = []
        for result in results:
            entry = standing.get(fingerprint(result))
            if entry is not None and entry.label is not None:
                chosen.append(result)
                labels.append(entry.label)
        return chosen, labels

    def counts(self) -> Dict[str, int]:
        """Headline counts for the workflow map and the dashboard."""
        standing = list(self.current().values())
        return {
            "decided": len(standing),
            "confirmed": sum(1 for entry in standing if entry.decision == "confirmed"),
            "rejected": sum(1 for entry in standing if entry.decision == "rejected"),
            "unclear": sum(1 for entry in standing if entry.decision == "unclear"),
            "labels": sum(1 for entry in standing if entry.label is not None),
            "overturned": sum(1 for entry in standing if entry.overturns_engine),
            "revisions": len(self.entries) - len(standing),
        }

    def rows(self) -> List[Dict[str, object]]:
        """Every entry as a table payload, newest first - the audit trail."""
        return [entry.to_dict() for entry in reversed(self.entries)]

    def export_csv(self, path: str) -> str:
        """Write the full trail as CSV for an auditor; returns the path."""
        columns = ["decided_at", "reference", "decision", "label", "reviewer",
                   "trigger", "engine_verdict", "overturns_engine", "risk_score",
                   "note", "fingerprint"]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for entry in self.entries:
                payload = entry.to_dict()
                writer.writerow({key: payload.get(key, "") for key in columns})
        LOGGER.info("Exported %d review decision(s) to %s", len(self.entries), path)
        return path
