"""Stage 4 - evidence engine.

A flag nobody can audit is worse than no flag: the safety officer who reads the
dashboard has to see *why* a report was raised before acting on it. This stage
collects what each earlier stage matched - lexical cues, the nearest semantic
prototypes and their similarity, which path drove the decision - into one
structure, and renders the one-line explanation shown in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .heads import Entities, RuleVerdict, SIFVerdict
from .lexical import SIFAssessment

__all__ = ["Evidence", "EvidenceEngine"]


@dataclass
class Evidence:
    """Everything that justifies one result."""

    lexical_cues: List[str] = field(default_factory=list)
    semantic_matches: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    rule_alternatives: List[Tuple[str, float]] = field(default_factory=list)
    field_sources: Dict[str, str] = field(default_factory=dict)
    decision_path: str = ""
    explanation: str = ""
    encoder: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "lexical_cues": list(self.lexical_cues),
            "semantic_matches": {key: list(value) for key, value in self.semantic_matches.items()},
            "rule_alternatives": [list(item) for item in self.rule_alternatives],
            "field_sources": dict(self.field_sources),
            "decision_path": self.decision_path,
            "explanation": self.explanation,
            "encoder": self.encoder,
        }


class EvidenceEngine:
    """Assembles the audit trail for one report."""

    def assemble(self, lexical: SIFAssessment, sif: SIFVerdict, rule: RuleVerdict,
                 entities: Entities, encoder_label: str) -> Evidence:
        semantic: Dict[str, Tuple[str, float]] = dict(entities.semantic_matches)
        if sif.semantic_energy:
            semantic["energy"] = sif.semantic_energy.as_tuple()
        if sif.semantic_barrier:
            semantic["barrier"] = sif.semantic_barrier.as_tuple()

        return Evidence(
            lexical_cues=self._dedupe([cue.strip().rstrip(",") for cue in lexical.evidence]),
            semantic_matches=semantic,
            rule_alternatives=[(label, round(score, 3)) for label, score in rule.alternatives],
            field_sources=dict(entities.sources),
            decision_path=sif.decision_path,
            explanation=self._explain(sif, rule, entities),
            encoder=encoder_label,
        )

    @staticmethod
    def _dedupe(cues):
        """Drop cue fragments wholly contained in a longer cue from the same list."""
        unique = []
        for cue in sorted(dict.fromkeys(cues), key=len, reverse=True):
            if not any(cue in kept for kept in unique):
                unique.append(cue)
        return sorted(unique)

    @staticmethod
    def _explain(sif: SIFVerdict, rule: RuleVerdict, entities: Entities) -> str:
        """One sentence a safety officer can act on without opening the record."""
        if not sif.sif_potential:
            if sif.high_energy:
                return (f"High energy present ({sif.energy_label}) but no barrier failure "
                        "was reported - controlled work.")
            if sif.barrier_failed:
                return (f"Barrier lapse ({sif.barrier_label}) with no high-energy source - "
                        "housekeeping or quality issue, not a fatality precursor.")
            return "Neither a high-energy source nor a barrier failure was identified."
        return (f"{sif.energy_label} was uncontrolled because {sif.barrier_label.lower()} "
                f"during {entities.activity.lower()} - {rule.rule} exposure.")
