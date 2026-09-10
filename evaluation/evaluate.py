"""Score the NLP engine against a hand-labelled set.

"The engine works well" is not a claim anyone can check. This turns it into
numbers that can be checked, and that move when the knowledge base changes:

    python evaluation/evaluate.py                 # the offline rule engine
    python evaluation/evaluate.py --backend auto  # with the semantic layer
    python evaluation/evaluate.py --errors        # list every miss, with its text

The labels in ``labelled_reports.csv`` were written from the safety logic - high
energy AND a failed barrier - not from what the engine happens to output, and the
set deliberately includes two kinds of hard case:

* **Negative controls** (``*-N*``): the same incident with the barrier *holding*.
  "LOTO was applied and verified" must not flag. These are what stop a recall
  chase from quietly turning the engine into a keyword alarm.
* **Implicit failures**: a barrier that failed without the report using a negative
  phrase - "clipped to the handrail instead of the anchor point", "car-sealed open
  with no tag". These are where a rule engine loses recall, and where the review
  queue earns its place.

**Recall is the headline.** A missed precursor is an incident nobody looked at; a
false positive costs a reviewer two minutes. The two errors are not equal and the
report does not average them away.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sif.pipeline import SIFPipeline  # noqa: E402  - after the path fix

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labelled_reports.csv")


@dataclass
class Case:
    """One labelled report."""

    identifier: str
    expected_sif: bool
    expected_rule: str
    report: str
    note: str


@dataclass
class Outcome:
    """What the engine made of one case."""

    case: Case
    predicted_sif: bool
    predicted_rule: str
    risk: float
    energy: bool
    barrier: bool
    queued: str

    @property
    def correct(self) -> bool:
        return self.predicted_sif == self.case.expected_sif

    @property
    def kind(self) -> str:
        if self.predicted_sif and self.case.expected_sif:
            return "true positive"
        if self.predicted_sif:
            return "FALSE POSITIVE"
        if self.case.expected_sif:
            return "MISS"
        return "true negative"


def load(path: str = DATASET) -> List[Case]:
    """Read the labelled set."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return [Case(identifier=row["id"], expected_sif=row["sif"].strip() == "1",
                     expected_rule=row["rule"], report=row["report"], note=row["note"])
                for row in csv.DictReader(handle)]


def run(cases: Sequence[Case], backend: str = "hashing") -> List[Outcome]:
    """Analyse every case with one encoder backend."""
    pipeline = SIFPipeline(backend=backend)
    outcomes: List[Outcome] = []
    for case in cases:
        result = pipeline.analyze(case.report, reference=case.identifier)
        outcomes.append(Outcome(
            case=case, predicted_sif=result.sif_potential, predicted_rule=result.iogp_rule,
            risk=result.risk_score, energy=result.high_energy,
            barrier=result.barrier_failed, queued=result.review_trigger or ""))
    return outcomes


def score(outcomes: Sequence[Outcome]) -> Dict[str, float]:
    """Precision, recall and the counts behind them."""
    true_positive = sum(1 for item in outcomes if item.kind == "true positive")
    false_positive = sum(1 for item in outcomes if item.kind == "FALSE POSITIVE")
    missed = sum(1 for item in outcomes if item.kind == "MISS")
    true_negative = sum(1 for item in outcomes if item.kind == "true negative")

    precision = true_positive / (true_positive + false_positive) if true_positive else 0.0
    recall = true_positive / (true_positive + missed) if true_positive else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    positives = [item for item in outcomes if item.case.expected_sif]
    rules_right = sum(1 for item in positives
                      if item.predicted_rule == item.case.expected_rule)
    return {
        "cases": len(outcomes),
        "true_positive": true_positive, "false_positive": false_positive,
        "missed": missed, "true_negative": true_negative,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "accuracy": round((true_positive + true_negative) / len(outcomes), 3),
        "barrier_recall": round(
            sum(1 for item in positives if item.barrier) / len(positives), 3)
        if positives else 0.0,
        "energy_recall": round(
            sum(1 for item in positives if item.energy) / len(positives), 3)
        if positives else 0.0,
        "rule_accuracy": round(rules_right / len(positives), 3) if positives else 0.0,
        "reviewed": sum(1 for item in outcomes if item.queued),
        "missed_and_unqueued": sum(1 for item in outcomes
                                   if item.kind == "MISS" and not item.queued),
        # Independent of the label: any report the engine itself calls
        # SIF-potential must reach a person. This is the review queue's own
        # contract, checked against what it actually did - not against whether
        # the call happened to be right.
        "confirmed_unqueued": sum(1 for item in outcomes
                                  if item.predicted_sif and not item.queued),
    }


def report(outcomes: Sequence[Outcome], show_errors: bool = False) -> Dict[str, float]:
    """Print the scorecard; returns the numbers for a caller that wants them."""
    numbers = score(outcomes)
    print(f"  cases                 {numbers['cases']}")
    print(f"  recall                {numbers['recall']:.3f}   "
          f"<- the one that matters: a miss is an incident nobody looked at")
    print(f"  precision             {numbers['precision']:.3f}   "
          f"(a false positive costs a reviewer two minutes)")
    print(f"  f1                    {numbers['f1']:.3f}")
    print(f"  accuracy              {numbers['accuracy']:.3f}")
    print(f"  rule accuracy         {numbers['rule_accuracy']:.3f}  "
          f"(on true positives - where the rule drives an intervention)")
    print(f"  energy recall         {numbers['energy_recall']:.3f}  (on true positives)")
    print(f"  barrier recall        {numbers['barrier_recall']:.3f}  (on true positives)")
    print(f"  confusion             {numbers['true_positive']} TP · "
          f"{numbers['false_positive']} FP · {numbers['missed']} miss · "
          f"{numbers['true_negative']} TN")
    print(f"  queued for a human    {numbers['reviewed']}")
    print(f"  missed AND unqueued   {numbers['missed_and_unqueued']}   "
          f"<- the only truly silent failures")
    print(f"  confirmed, unqueued   {numbers['confirmed_unqueued']}   "
          f"<- must always be 0: a flag the engine raised and nobody saw")

    if show_errors:
        for item in outcomes:
            if item.correct:
                continue
            print(f"\n  {item.kind}  {item.case.identifier}  "
                  f"(expected {item.case.expected_rule}, got {item.predicted_rule})")
            print(f"    energy={item.energy} barrier={item.barrier} "
                  f"risk={item.risk:.1f} queued={item.queued or 'NO'}")
            print(f"    note: {item.case.note}")
            print(f"    {item.case.report[:150]}")
    return numbers


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="hashing",
                        help="encoder backend: hashing (offline), transformer, or auto")
    parser.add_argument("--errors", action="store_true",
                        help="print every misclassified report with its text")
    parser.add_argument("--dataset", default=DATASET)
    arguments = parser.parse_args(list(argv) or None)

    cases = load(arguments.dataset)
    print(f"\nSENTRA engine evaluation - {len(cases)} labelled reports, "
          f"backend={arguments.backend}\n")
    numbers = report(run(cases, arguments.backend), arguments.errors)
    print()
    # Non-zero when a positive was both missed and never queued: that is the
    # failure mode this product exists to prevent.
    return 1 if numbers["missed_and_unqueued"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
