"""The SIF analysis pipeline - orchestration of every stage.

    report text
        -> NLPPreprocessor          (stage 1: clean, expand, segment)
        -> SemanticEncoder          (stage 2: transformer sentence embeddings)
        -> SIFClassifier            (stage 3a: fatality-precursor probability)
           RuleClassifier           (stage 3b: IOGP Life-Saving Rule)
           EntityExtractor          (stage 3c: activity / location / barrier)
        -> EvidenceEngine           (stage 4: audit trail and explanation)
        -> RiskScorer               (stage 5: 0-100 ranked risk)
        -> PatternDetector          (stage 6a: hotspots across the corpus)
           ReviewQueue              (stage 6b: what a human must verify)
        -> HSE intelligence         (what the dashboard renders)

The lexical engine from :mod:`sif.lexical` runs alongside the encoder at every
stage; it is the deterministic backbone, and the semantic layer extends it
rather than replacing it.

Nothing here touches Qt: the pipeline is a plain Python object the GUI drives
from a worker thread, and the same object is what a batch job or an API would
use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .encoders import SemanticEncoder, load_encoder
from .evidence import EvidenceEngine
from .heads import EntityExtractor, RuleClassifier, SIFClassifier, SemanticIndex
from .lexical import LexicalEngine, SEED_REPORTS
from .patterns import Hotspot, PatternDetector
from .preprocessing import NLPPreprocessor
from .review import ReviewItem, ReviewQueue
from .scoring import RiskScorer

__all__ = ["PipelineResult", "Intelligence", "SIFPipeline"]


@dataclass
class PipelineResult:
    """The structured output for one field report.

    The five fields required by Problem Statement 26165 - ``sif_potential``,
    ``iogp_rule``, ``activity``, ``location`` and ``barrier_failure`` - keep
    their original names and meaning; everything else is the semantic layer's
    added detail.
    """

    sif_potential: bool
    iogp_rule: str
    activity: str
    location: str
    barrier_failure: str
    energy_source: str
    p_sif: float = 0.0
    risk_score: float = 0.0
    risk_band: str = "Low"
    severity_hint: str = "Low"
    confidence: float = 0.0
    rule_confidence: float = 0.0
    high_energy: bool = False
    barrier_failed: bool = False
    lexical_flag: bool = False
    semantic_flag: bool = False
    semantic_active: bool = False
    ml_probability: Optional[float] = None
    ml_flag: bool = False
    ml_active: bool = False
    needs_review: bool = False
    review_trigger: str = ""
    review_reason: str = ""
    explanation: str = ""
    reference: str = ""
    encoder: str = ""
    elapsed_ms: float = 0.0
    evidence: Dict[str, object] = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Return a plain, JSON-serialisable dictionary."""
        payload = dict(self.__dict__)
        payload["evidence"] = dict(self.evidence)
        return payload


@dataclass
class Intelligence:
    """Corpus-level output: what the dashboard header and panels render."""

    kpis: Dict[str, object] = field(default_factory=dict)
    hotspots: List[Hotspot] = field(default_factory=list)
    review_queue: List[ReviewItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "kpis": dict(self.kpis),
            "hotspots": [spot.to_dict() for spot in self.hotspots],
            "review_queue": [item.to_dict() for item in self.review_queue],
        }


class SIFPipeline:
    """End-to-end analysis pipeline.

    Construction is cheap and does no I/O; the encoder is resolved on first use
    (or by an explicit :meth:`warm_up`), so a GUI can build the pipeline on the
    main thread and pay the model-loading cost on a worker.

    Example
    -------
    >>> pipeline = SIFPipeline(backend="hashing")      # offline, deterministic
    >>> result = pipeline.analyze("No harness worn on the scaffold at 6 m.")
    >>> result.sif_potential, result.risk_band
    (True, 'Critical')
    """

    def __init__(self, encoder: Optional[SemanticEncoder] = None, backend: str = "auto",
                 model_name: Optional[str] = None, model_provider: object = None) -> None:
        self._encoder = encoder
        self._backend = backend
        self._model_name = model_name
        self._index: Optional[SemanticIndex] = None
        # Duck-typed: anything with .predict(result) -> Optional[float]. Kept
        # untyped so the learned layer stays optional and this module never has
        # to import xgboost.
        self._model_provider = model_provider

        self.preprocessor = NLPPreprocessor()
        self.lexical = LexicalEngine()
        self.sif_head = SIFClassifier()
        self.rule_head = RuleClassifier()
        self.entity_head = EntityExtractor()
        self.evidence_engine = EvidenceEngine()
        self.scorer = RiskScorer()
        self.detector = PatternDetector()
        self.reviewer = ReviewQueue()

    # -- lifecycle ---------------------------------------------------------

    @property
    def encoder(self) -> SemanticEncoder:
        """The resolved encoder, loading it on first access."""
        if self._encoder is None:
            self._encoder = load_encoder(self._backend, self._model_name)
        return self._encoder

    @property
    def index(self) -> SemanticIndex:
        """The prototype index, built on first access."""
        if self._index is None:
            self._index = SemanticIndex(self.encoder)
        return self._index

    def warm_up(self) -> str:
        """Resolve the encoder, load its weights and embed the prototypes.

        Returns the encoder label for the status bar. Call it from a worker
        thread before the first report so the user never waits for a model
        download mid-analysis - and so a missing model degrades to the offline
        fallback before any report is scored, not halfway through the batch.
        """
        self.index.build()
        return self.encoder.info.label()

    def attach_model(self, provider: object) -> None:
        """Attach a trained model (see :class:`sif.mlops.MLOpsService`).

        The provider only has to expose ``predict(result) -> float | None``; the
        pipeline treats it as a third opinion alongside the rules and the
        encoder, never as an override.
        """
        self._model_provider = provider

    @property
    def has_model(self) -> bool:
        """True when a learned model is attached."""
        return self._model_provider is not None

    # -- analysis ----------------------------------------------------------

    def analyze(self, text: str, reference: str = "") -> PipelineResult:
        """Run every stage over one report."""
        started = time.perf_counter()
        document = self.preprocessor.process(text)
        lexical = self.lexical.assess(text)

        if document.is_empty:
            return PipelineResult(
                sif_potential=False, iogp_rule=lexical.iogp_rule, activity=lexical.activity,
                location=lexical.location, barrier_failure=lexical.barrier_failure,
                energy_source=lexical.energy_source, reference=reference,
                encoder=self.encoder.info.label(),
                explanation="Empty report - nothing to analyse.",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        vectors = self.encoder.encode(document.sentences)
        sif = self.sif_head.classify(document, lexical, vectors, self.index)
        rule = self.rule_head.classify(document, self.lexical.rule_scores(text),
                                       vectors, self.index)
        entities = self.entity_head.extract(document, lexical, vectors, self.index,
                                            sif.barrier_label)
        evidence = self.evidence_engine.assemble(lexical, sif, rule, entities,
                                                 self.encoder.info.label())
        risk = self.scorer.score(sif, lexical.confidence)

        result = PipelineResult(
            sif_potential=sif.sif_potential,
            iogp_rule=rule.rule,
            activity=entities.activity,
            location=entities.location,
            barrier_failure=entities.barrier,
            energy_source=sif.energy_label,
            p_sif=sif.probability,
            risk_score=risk.value,
            risk_band=risk.band,
            severity_hint=self._severity(sif, lexical.severity_hint),
            confidence=lexical.confidence,
            rule_confidence=rule.confidence,
            high_energy=sif.high_energy,
            barrier_failed=sif.barrier_failed,
            lexical_flag=sif.lexical_flag,
            semantic_flag=sif.semantic_flag,
            semantic_active=sif.semantic_active,
            explanation=evidence.explanation,
            reference=reference,
            encoder=self.encoder.info.label(),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            evidence={**evidence.to_dict(), "risk": risk.to_dict()},
            raw_text=document.raw,
        )
        self._apply_model(result)
        trigger, reason = self.reviewer.classify(result)
        result.needs_review = trigger is not None
        result.review_trigger = trigger or ""
        result.review_reason = reason
        return result

    def analyze_many(self, texts: Sequence[str],
                     references: Optional[Sequence[str]] = None) -> List[PipelineResult]:
        """Analyse a sequence of reports, skipping blank entries."""
        results: List[PipelineResult] = []
        for position, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                continue
            reference = references[position] if references and position < len(references) else ""
            results.append(self.analyze(text, reference))
        return results

    def _apply_model(self, result: PipelineResult) -> None:
        """Score the result with the learned model, when one is attached."""
        if self._model_provider is None:
            return
        probability = self._model_provider.predict(result)
        if probability is None:
            return
        result.ml_probability = round(float(probability), 3)
        result.ml_flag = result.ml_probability >= 0.5
        result.ml_active = True
        result.evidence["model"] = {
            "probability": result.ml_probability,
            "agrees": result.ml_flag == result.sif_potential,
        }

    # -- corpus level ------------------------------------------------------

    def aggregate(self, results: Sequence[PipelineResult]) -> Intelligence:
        """Turn a set of results into dashboard-level HSE intelligence."""
        total = len(results)
        flagged = [item for item in results if item.sif_potential]
        risks = [item.risk_score for item in results]
        rule_counts: Dict[str, int] = {}
        for item in flagged:
            rule_counts[item.iogp_rule] = rule_counts.get(item.iogp_rule, 0) + 1

        kpis = {
            "total": total,
            "sif_potential": len(flagged),
            "sif_rate": round(len(flagged) / total * 100.0, 1) if total else 0.0,
            "mean_risk": round(float(np.mean(risks)), 1) if risks else 0.0,
            "max_risk": round(max(risks), 1) if risks else 0.0,
            "critical": sum(1 for item in results if item.risk_band == "Critical"),
            "top_rule": max(rule_counts, key=rule_counts.get) if rule_counts else "-",
            "needs_review": sum(1 for item in results if item.needs_review),
            "encoder": results[0].encoder if results else "",
        }
        return Intelligence(kpis=kpis,
                            hotspots=self.detector.detect(results),
                            review_queue=self.reviewer.build(results))

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _severity(sif, lexical_hint: str) -> str:
        """Escalate the lexical severity hint when the model flags the report."""
        if sif.sif_potential:
            return "High"
        return lexical_hint


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    pipeline = SIFPipeline()
    print("encoder:", pipeline.warm_up())
    outcomes = pipeline.analyze_many(SEED_REPORTS)
    for number, outcome in enumerate(outcomes, start=1):
        print(f"[{number}] SIF={outcome.sif_potential} risk={outcome.risk_score:5.1f} "
              f"({outcome.risk_band:8}) {outcome.iogp_rule}")
        print(f"    {outcome.explanation}")
    intelligence = pipeline.aggregate(outcomes)
    print("KPIs:", intelligence.kpis)
    for spot in intelligence.hotspots[:3]:
        print(f"  hotspot [{spot.kind}] {spot.label}: {spot.reports} reports, "
              f"{spot.sif_reports} SIF, mean risk {spot.mean_risk}")
    for item in intelligence.review_queue[:3]:
        print(f"  review [{item.trigger}] {item.reference}: {item.reason}")
