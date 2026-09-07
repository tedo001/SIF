"""Stage 6a - pattern detection and risk hotspots.

One report is an incident; the same barrier failing at the same place three
times is a system problem. This stage aggregates results across the corpus and
surfaces where the exposure concentrates, which is the intelligence the
dashboard exists to show.

Four aggregations, ranked by **SIF-precursor density** - the share of a group's
reports that carry fatal potential - rather than by raw count, because a site
with 8 precursors in 400 reports is a different problem from one with 6 in 8:

* **Location hotspots** - where the precursors cluster;
* **Activity hotspots** - which task carries the fatal potential, wherever it runs;
* **Rule-at-location patterns** - the same exposure repeating at one site;
* **Repeat barrier failures** - the control that keeps breaking, wherever it is.

Ranking a rate over small samples needs care: 2 of 2 is 100% and means almost
nothing. The order therefore uses the **Wilson score lower bound** of the
proportion, which discounts a rate in proportion to how little evidence supports
it, and both the raw density and that bound are shown so the ranking can be
audited rather than trusted.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pipeline import PipelineResult

__all__ = ["Hotspot", "PatternDetector"]

UNKNOWN_LOCATIONS = {"Location not stated"}
UNKNOWN_ACTIVITIES = {"Unspecified activity"}

#: z for a 95% one-sided Wilson interval.
WILSON_Z = 1.96


def wilson_lower_bound(successes: int, total: int, z: float = WILSON_Z) -> float:
    """Lower bound of the 95% Wilson interval for a proportion, as a percentage.

    Used to rank precursor density without letting a two-report group outrank a
    well-evidenced one: 2/2 scores about 34%, while 20/30 scores about 49%.
    """
    if total <= 0:
        return 0.0
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return round(max(0.0, (centre - margin) / denominator) * 100.0, 1)


@dataclass
class Hotspot:
    """An aggregation of reports that share a location, rule or barrier."""

    kind: str
    label: str
    reports: int
    sif_reports: int
    mean_risk: float
    max_risk: float
    top_rule: str = ""
    top_barrier: str = ""
    indices: List[int] = field(default_factory=list)

    @property
    def sif_rate(self) -> float:
        """SIF-precursor density: share of the group's reports that are flagged."""
        return (self.sif_reports / self.reports * 100.0) if self.reports else 0.0

    @property
    def priority(self) -> float:
        """Evidence-adjusted density used for ranking (Wilson lower bound)."""
        return wilson_lower_bound(self.sif_reports, self.reports)

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind, "label": self.label, "reports": self.reports,
            "sif_reports": self.sif_reports, "sif_rate": round(self.sif_rate, 1),
            "priority": self.priority, "mean_risk": self.mean_risk,
            "max_risk": self.max_risk, "top_rule": self.top_rule,
            "top_barrier": self.top_barrier, "indices": list(self.indices),
        }


class PatternDetector:
    """Aggregates pipeline results into ranked hotspots."""

    #: A group must reach this many reports before it counts as a pattern.
    MIN_REPORTS = 2

    def detect(self, results: Sequence["PipelineResult"]) -> List[Hotspot]:
        """Return every hotspot found across ``results``, densest exposure first."""
        hotspots = (self._group(results, "Location", lambda r: r.location,
                                skip=UNKNOWN_LOCATIONS)
                    + self._group(results, "Activity", lambda r: r.activity,
                                  skip=UNKNOWN_ACTIVITIES)
                    + self._group(results, "Rule at location",
                                  lambda r: f"{r.iogp_rule} @ {r.location}",
                                  skip=None, require_sif=True)
                    + self._group(results, "Repeat barrier failure",
                                  lambda r: r.barrier_failure.split(";")[0].strip(),
                                  skip={"No barrier failure identified"}))
        # Density first (evidence-adjusted), then volume, then severity.
        hotspots.sort(key=lambda spot: (-spot.priority, -spot.sif_reports,
                                        -spot.mean_risk, spot.label))
        return hotspots

    def _group(self, results, kind, key, skip=None, require_sif=False) -> List[Hotspot]:
        buckets: Dict[str, List] = defaultdict(list)
        positions: Dict[str, List[int]] = defaultdict(list)
        for position, result in enumerate(results, start=1):
            if require_sif and not result.sif_potential:
                continue
            label = key(result)
            if not label or (skip and label in skip):
                continue
            if "Location not stated" in label:
                continue
            buckets[label].append(result)
            positions[label].append(position)

        hotspots: List[Hotspot] = []
        for label, group in buckets.items():
            if len(group) < self.MIN_REPORTS:
                continue
            risks = [item.risk_score for item in group]
            hotspots.append(Hotspot(
                kind=kind,
                label=label,
                reports=len(group),
                sif_reports=sum(1 for item in group if item.sif_potential),
                mean_risk=round(sum(risks) / len(risks), 1),
                max_risk=round(max(risks), 1),
                top_rule=self._mode(item.iogp_rule for item in group),
                top_barrier=self._mode(item.barrier_failure.split(";")[0].strip()
                                       for item in group),
                indices=positions[label],
            ))
        return hotspots

    @staticmethod
    def _mode(values) -> str:
        counts: Dict[str, int] = defaultdict(int)
        for value in values:
            counts[value] += 1
        return max(counts, key=counts.get) if counts else ""
