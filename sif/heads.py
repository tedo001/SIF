"""Stage 3 - the three heads that read the encoded report.

All three consume the same inputs (the preprocessed :class:`~sif.preprocessing.Document`,
its sentence embeddings, and the deterministic lexical assessment) and each fuses
two independent signals:

* **lexical** - regular-expression evidence from :mod:`sif.lexical`, which is
  precise, auditable and never hallucinates, but only fires on phrasing it has
  seen; and
* **semantic** - cosine similarity between the report's sentences and the label
  prototypes in :mod:`sif.prototypes`, which generalises to paraphrases the
  patterns miss.

Fusion is deliberately conservative: the lexical decision is never overturned,
only *extended*. A report the patterns already flag stays flagged; the semantic
score can add a flag the patterns missed, and it supplies the continuous
probability that the risk score is built from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoders import SemanticEncoder
from .lexical import (
    NO_BARRIER_FAILURE,
    NO_ENERGY,
    UNCLASSIFIED_RULE,
    UNKNOWN_ACTIVITY,
    UNKNOWN_LOCATION,
    SIFAssessment,
)
from .preprocessing import Document
from .prototypes import (
    ACTIVITY_PROTOTYPES,
    BARRIER_PROTOTYPES,
    ENERGY_PROTOTYPES,
    LOCATION_PROTOTYPES,
    RULE_PROTOTYPES,
)

__all__ = ["SemanticIndex", "LabelMatch", "SIFVerdict", "RuleVerdict", "Entities",
           "SIFClassifier", "RuleClassifier", "EntityExtractor"]


@dataclass(frozen=True)
class LabelMatch:
    """One label and how strongly the text matched it."""

    label: str
    score: float
    sentence: str = ""

    def as_tuple(self) -> Tuple[str, float]:
        return self.label, round(self.score, 3)


class SemanticIndex:
    """Embeds every prototype set once and answers nearest-label queries.

    Building the index costs one encoder pass over ~70 short sentences. It is
    done lazily on first use - which always happens on a worker thread - and
    then reused for the life of the process.
    """

    SETS = {
        "rule": RULE_PROTOTYPES,
        "energy": ENERGY_PROTOTYPES,
        "barrier": BARRIER_PROTOTYPES,
        "activity": ACTIVITY_PROTOTYPES,
        "location": LOCATION_PROTOTYPES,
    }

    def __init__(self, encoder: SemanticEncoder) -> None:
        self.encoder = encoder
        self._matrices: Dict[str, np.ndarray] = {}
        self._labels: Dict[str, List[str]] = {}
        self._built = False

    def build(self) -> None:
        """Embed all prototype sentences (idempotent)."""
        if self._built:
            return
        for name, mapping in self.SETS.items():
            sentences: List[str] = []
            labels: List[str] = []
            for label, prototypes in mapping.items():
                for sentence in prototypes:
                    sentences.append(sentence)
                    labels.append(label)
            self._matrices[name] = self.encoder.encode(sentences)
            self._labels[name] = labels
        self._built = True

    def rank(self, name: str, vectors: np.ndarray,
             sentences: Optional[Sequence[str]] = None) -> List[LabelMatch]:
        """Rank every label in set ``name`` against the report's sentences.

        A label's score is the highest similarity between any of its prototypes
        and any sentence of the report - the report only has to say the thing
        once, in one sentence, to match.
        """
        self.build()
        if vectors.size == 0:
            return []
        similarity = self.encoder.similarity(vectors, self._matrices[name])
        best: Dict[str, Tuple[float, int]] = {}
        for column, label in enumerate(self._labels[name]):
            row = int(np.argmax(similarity[:, column]))
            score = float(similarity[row, column])
            if label not in best or score > best[label][0]:
                best[label] = (score, row)
        ranked = [
            LabelMatch(label, score,
                       sentences[row] if sentences and row < len(sentences) else "")
            for label, (score, row) in best.items()
        ]
        ranked.sort(key=lambda match: match.score, reverse=True)
        return ranked


#: A ranking is informative only if the best label stands clear of the field.
#: An encoder that scores every prototype alike (an untrained, mis-loaded or
#: otherwise degenerate model) produces a near-zero margin, and its "nearest"
#: label is an artefact of noise - so the decision falls back to the lexical
#: rules instead of trusting it.
MIN_MARGIN = 0.05


def _margin(ranked: Sequence[LabelMatch]) -> float:
    """Spread between the top label and the median label of a ranking."""
    if len(ranked) < 3:
        return 0.0
    scores = np.array([match.score for match in ranked], dtype=np.float32)
    return float(scores.max() - np.median(scores))


def _calibrate(similarity: float, semantic_backend: bool) -> float:
    """Map a raw cosine similarity onto a 0-1 confidence.

    The two backends live on different scales - a transformer puts unrelated
    sentences around 0.1 and a good match above 0.5, while the hashing fallback
    is compressed near zero - so each gets its own band, and the fallback is
    capped so it can never carry a decision on its own.
    """
    # The fallback is capped below the presence floor: lexical overlap can rank
    # and enrich, but it can never on its own assert that a factor is present.
    low, high, cap = (0.22, 0.55, 1.0) if semantic_backend else (0.18, 0.45, 0.45)
    if similarity <= low:
        return 0.0
    return float(min(cap, (similarity - low) / (high - low)))


@dataclass
class SIFVerdict:
    """Output of the SIF head."""

    sif_potential: bool
    probability: float
    energy_label: str
    energy_score: float
    barrier_label: str
    barrier_score: float
    high_energy: bool
    barrier_failed: bool
    lexical_flag: bool
    semantic_flag: bool
    #: True only when a real language model produced a discriminative ranking.
    #: Without it there is no second opinion to agree or disagree with.
    semantic_active: bool = False
    semantic_energy: Optional[LabelMatch] = None
    semantic_barrier: Optional[LabelMatch] = None
    decision_path: str = ""


class SIFClassifier:
    """Head 1 - does this report describe a fatality precursor?

    The model is unchanged from the lexical engine, expressed continuously:
    ``P(SIF) = energy_score x barrier_score``. The product *is* the "high energy
    AND failed barrier" rule - either factor at zero drives the probability to
    zero - but it now degrades smoothly instead of flipping on a keyword.
    """

    #: Probability at which the semantic path alone raises a flag.
    THRESHOLD = 0.5
    #: Factor score at which a component counts as present.
    PRESENCE = 0.5
    #: Both factors must clear this before the semantic path may raise a flag the
    #: lexical rules did not. A single strong factor paired with a marginal one is
    #: how a false positive gets in, so marginal is not enough on its own.
    SEMANTIC_FACTOR_FLOOR = 0.6

    def classify(self, document: Document, lexical: SIFAssessment,
                 vectors: np.ndarray, index: SemanticIndex) -> SIFVerdict:
        semantic = index.encoder.info.semantic
        energy_ranked = index.rank("energy", vectors, document.sentences)
        barrier_ranked = index.rank("barrier", vectors, document.sentences)
        top_energy = energy_ranked[0] if energy_ranked else None
        top_barrier = barrier_ranked[0] if barrier_ranked else None

        # Semantic factors count towards the verdict only under a real language
        # model. The hashing fallback still reports its nearest neighbours as
        # evidence, but they carry no weight in the decision or the score, so
        # offline mode behaves exactly like the deterministic rules.
        energy_informative = _margin(energy_ranked) >= MIN_MARGIN
        barrier_informative = _margin(barrier_ranked) >= MIN_MARGIN
        sem_energy = (_calibrate(top_energy.score, semantic)
                      if (top_energy and semantic and energy_informative) else 0.0)
        sem_barrier = (_calibrate(top_barrier.score, semantic)
                       if (top_barrier and semantic and barrier_informative) else 0.0)
        energy_score = max(1.0 if lexical.high_energy else 0.0, sem_energy)
        barrier_score = max(1.0 if lexical.barrier_failed else 0.0, sem_barrier)
        probability = round(energy_score * barrier_score, 3)

        lexical_flag = bool(lexical.sif_potential)
        # A non-semantic backend never raises a flag of its own: offline mode is
        # exactly as precise as the lexical rules, never noisier.
        semantic_flag = (
            semantic
            and probability >= self.THRESHOLD
            and energy_score >= self.SEMANTIC_FACTOR_FLOOR
            and barrier_score >= self.SEMANTIC_FACTOR_FLOOR
        )
        # Only name a semantically inferred factor once it clears the presence
        # floor - a weak nearest neighbour is noise, not a finding.
        energy_label = (lexical.energy_source if lexical.high_energy
                        else (top_energy.label if top_energy and sem_energy >= self.PRESENCE
                              else NO_ENERGY))
        barrier_label = (lexical.barrier_failure if lexical.barrier_failed
                         else (top_barrier.label if top_barrier and sem_barrier >= self.PRESENCE
                               else NO_BARRIER_FAILURE))

        if not semantic:
            path = "lexical rules only - semantic layer inactive (offline encoder)"
        elif not (energy_informative and barrier_informative):
            path = "lexical rules only - semantic ranking was not discriminative"
        elif lexical_flag and semantic_flag:
            path = "lexical and semantic agree"
        elif lexical_flag:
            path = "lexical rule only - semantic score below threshold"
        elif semantic_flag:
            path = "semantic only - phrasing not covered by the patterns"
        else:
            path = "neither path met both conditions"

        return SIFVerdict(
            sif_potential=lexical_flag or semantic_flag,
            probability=probability,
            energy_label=energy_label,
            energy_score=round(energy_score, 3),
            barrier_label=barrier_label,
            barrier_score=round(barrier_score, 3),
            high_energy=energy_score >= self.PRESENCE,
            barrier_failed=barrier_score >= self.PRESENCE,
            lexical_flag=lexical_flag,
            semantic_flag=semantic_flag,
            semantic_active=bool(semantic and energy_informative and barrier_informative),
            semantic_energy=top_energy,
            semantic_barrier=top_barrier,
            decision_path=path,
        )


@dataclass
class RuleVerdict:
    """Output of the rule head."""

    rule: str
    confidence: float
    lexical_score: float
    semantic_score: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


class RuleClassifier:
    """Head 2 - which IOGP Life-Saving Rule does the report belong to?"""

    #: Lexical score treated as full confidence (three primary cues).
    LEXICAL_SATURATION = 9.0
    #: Fused score below which the report stays unclassified.
    FLOOR = 0.28

    def classify(self, document: Document, lexical_scores: Dict[str, int],
                 vectors: np.ndarray, index: SemanticIndex) -> RuleVerdict:
        ranked = index.rank("rule", vectors, document.sentences)
        semantic = index.encoder.info.semantic and _margin(ranked) >= MIN_MARGIN
        semantic_scores = {match.label: _calibrate(match.score, semantic) for match in ranked}
        lex_weight, sem_weight = (0.55, 0.45) if semantic else (0.8, 0.2)

        fused: Dict[str, float] = {}
        for label in set(lexical_scores) | set(semantic_scores):
            lex = min(lexical_scores.get(label, 0) / self.LEXICAL_SATURATION, 1.0)
            sem = semantic_scores.get(label, 0.0)
            fused[label] = round(lex_weight * lex + sem_weight * sem, 3)

        if not fused:
            return RuleVerdict(UNCLASSIFIED_RULE, 0.0, 0.0, 0.0, [])

        order = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        label, score = order[0]
        lexical_component = round(
            min(lexical_scores.get(label, 0) / self.LEXICAL_SATURATION, 1.0), 3)
        if score < self.FLOOR:
            return RuleVerdict(UNCLASSIFIED_RULE, round(score, 3), lexical_component,
                               round(semantic_scores.get(label, 0.0), 3), order[:3])
        return RuleVerdict(
            rule=label,
            confidence=round(score, 3),
            lexical_score=lexical_component,
            semantic_score=round(semantic_scores.get(label, 0.0), 3),
            alternatives=order[1:4],
        )


@dataclass
class Entities:
    """Output of the extraction head: the three narrative fields."""

    activity: str
    location: str
    barrier: str
    sources: Dict[str, str] = field(default_factory=dict)
    semantic_matches: Dict[str, Tuple[str, float]] = field(default_factory=dict)


class EntityExtractor:
    """Head 3 - activity, location and failed barrier.

    Patterns run first because an exact gazetteer hit ("GGS-4" -> Group
    Gathering Station) beats any similarity score. The semantic index is
    consulted only where the lexical layer returned its fallback, and only above
    a similarity floor - below it the honest answer is "not stated", not a
    plausible guess.
    """

    FLOOR = 0.30

    def extract(self, document: Document, lexical: SIFAssessment,
                vectors: np.ndarray, index: SemanticIndex,
                barrier_label: str) -> Entities:
        semantic = index.encoder.info.semantic
        sources: Dict[str, str] = {}
        matches: Dict[str, Tuple[str, float]] = {}

        def resolve(field_name: str, lexical_value: str, fallback_value: str,
                    set_name: str) -> str:
            if lexical_value != fallback_value:
                sources[field_name] = "lexical"
                return lexical_value
            ranked = index.rank(set_name, vectors, document.sentences)
            if ranked:
                matches[field_name] = ranked[0].as_tuple()
                if _calibrate(ranked[0].score, semantic) >= self.FLOOR:
                    sources[field_name] = "semantic"
                    return ranked[0].label
            sources[field_name] = "fallback"
            return fallback_value

        activity = resolve("activity", lexical.activity, UNKNOWN_ACTIVITY, "activity")
        location = resolve("location", lexical.location, UNKNOWN_LOCATION, "location")
        if lexical.barrier_failed:
            sources["barrier"] = "lexical"
            barrier = lexical.barrier_failure
        elif barrier_label != NO_BARRIER_FAILURE:
            sources["barrier"] = "semantic"
            barrier = barrier_label
        else:
            sources["barrier"] = "fallback"
            barrier = NO_BARRIER_FAILURE

        return Entities(activity=activity, location=location, barrier=barrier,
                        sources=sources, semantic_matches=matches)
