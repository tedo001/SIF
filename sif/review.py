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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pipeline import PipelineResult

__all__ = ["ReviewItem", "ReviewQueue"]

PRIORITY_ORDER = {"Disagreement": 0, "Critical risk": 1, "Thin evidence": 2,
                  "Unclassified exposure": 3}


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

    def build(self, results: Sequence["PipelineResult"]) -> List[ReviewItem]:
        """Return the review queue for ``results``, highest risk first."""
        items: List[ReviewItem] = []
        for position, result in enumerate(results, start=1):
            trigger, reason = self.classify(result)
            if trigger is None:
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
        if result.risk_band == ReviewQueue.CRITICAL_BAND:
            return "Critical risk", (
                f"risk {result.risk_score:.0f}/100 - verify before it drives an intervention")
        if result.confidence < ReviewQueue.CONFIDENCE_FLOOR:
            return "Thin evidence", (
                f"extraction confidence {result.confidence:.2f} - narrative may be too "
                "sparse to classify")
        if result.high_energy and result.iogp_rule.startswith("Unclassified"):
            return "Unclassified exposure", (
                "high-energy source present but no Life-Saving Rule matched - likely "
                "unseen vocabulary")
        return None, ""
