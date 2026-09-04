"""Stage 1 - NLP preprocessor.

Field reports arrive as unedited shift-log prose: mixed case, inconsistent
punctuation, and dense with upstream abbreviations ("11 kV feeder at GGS-4, no
LOTO, PTW expired"). Two consumers need different things from that text:

* the **lexical layer** needs the surface form preserved, so its patterns keep
  matching what a safety officer actually typed;
* the **semantic encoder** does better on expanded, sentence-split prose, since
  a transformer trained on general English has never seen "LOTO" or "OCS".

:class:`NLPPreprocessor` therefore produces one :class:`Document` carrying both
views plus the sentence segmentation that every downstream head works over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern

__all__ = ["Document", "NLPPreprocessor", "ABBREVIATIONS"]

#: Upstream oil & gas shorthand expanded for the semantic view of the text.
ABBREVIATIONS: Dict[str, str] = {
    "loto": "lock out tag out energy isolation",
    "ptw": "permit to work",
    "jsa": "job safety analysis",
    "tbt": "toolbox talk",
    "ppe": "personal protective equipment",
    "scba": "self contained breathing apparatus",
    "sba": "supplied air breathing apparatus",
    "h2s": "hydrogen sulphide toxic gas",
    "lel": "lower explosive limit",
    "psv": "pressure safety relief valve",
    "esd": "emergency shutdown system",
    "bop": "blow out preventer",
    "swl": "safe working load",
    "mcc": "motor control centre",
    "ggs": "group gathering station",
    "ocs": "oil collecting station",
    "eps": "early production system",
    "etp": "effluent treatment plant",
    "row": "pipeline right of way",
    "ivms": "in vehicle monitoring system",
    "lpg": "liquefied petroleum gas",
    "hipo": "high potential incident",
    "ua": "unsafe act",
    "uc": "unsafe condition",
    "kv": "kilovolt high voltage",
}

#: Abbreviations that must not end a sentence during segmentation.
_NON_TERMINAL = ("no", "nos", "mr", "dr", "approx", "etc", "vs", "fig", "eq")

_SENTENCE_SPLIT: Pattern[str] = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(])")
_TOKEN: Pattern[str] = re.compile(r"[a-z0-9][a-z0-9\-/]*")
_WHITESPACE: Pattern[str] = re.compile(r"\s+")
_CONTROL: Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class Document:
    """One preprocessed field report.

    Attributes
    ----------
    raw:
        Text exactly as supplied, trimmed only of surrounding whitespace.
    normalised:
        Lower-cased, whitespace-collapsed surface form - what the lexical
        patterns run against.
    semantic:
        Abbreviation-expanded prose handed to the encoder.
    sentences:
        Sentence segmentation of ``semantic``; never empty for non-blank input.
    tokens:
        Lower-cased word tokens of ``normalised``.
    """

    raw: str = ""
    normalised: str = ""
    semantic: str = ""
    sentences: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when the report carried no usable text."""
        return not self.normalised


class NLPPreprocessor:
    """Cleans, expands and segments a raw report into a :class:`Document`."""

    #: Reports longer than this are truncated; field narratives never approach it.
    MAX_CHARS = 20_000

    def __init__(self, abbreviations: Dict[str, str] | None = None) -> None:
        self._abbreviations = dict(abbreviations or ABBREVIATIONS)
        # One alternation over all keys, longest first so "loto" wins over "lo".
        keys = sorted(self._abbreviations, key=len, reverse=True)
        self._abbrev_re = re.compile(
            r"\b(" + "|".join(re.escape(key) for key in keys) + r")\b", re.IGNORECASE
        )

    def process(self, text: str | None) -> Document:
        """Return the preprocessed view of ``text`` (blank input is tolerated)."""
        if not isinstance(text, str) or not text.strip():
            return Document()

        raw = _CONTROL.sub(" ", text).strip()[: self.MAX_CHARS]
        normalised = _WHITESPACE.sub(" ", raw).lower()
        normalised = normalised.replace("l.o.t.o", "loto").replace("p.t.w", "ptw")
        semantic = self._expand(_WHITESPACE.sub(" ", raw))
        return Document(
            raw=raw,
            normalised=normalised,
            semantic=semantic,
            sentences=self._split_sentences(semantic),
            tokens=_TOKEN.findall(normalised),
        )

    # -- internals ---------------------------------------------------------

    def _expand(self, text: str) -> str:
        """Replace known shorthand with its full form, keeping the original word."""

        def replace(match: "re.Match[str]") -> str:
            term = match.group(0)
            expansion = self._abbreviations[term.lower()]
            # Keep the acronym as well - it carries the reporter's own phrasing.
            return f"{term} ({expansion})"

        return self._abbrev_re.sub(replace, text)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Segment on terminal punctuation, guarding common abbreviations."""
        if not text:
            return []
        parts = _SENTENCE_SPLIT.split(text)
        merged: List[str] = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            previous = merged[-1].rstrip(". ").split()[-1].lower() if merged else ""
            if previous in _NON_TERMINAL:
                merged[-1] = f"{merged[-1]} {stripped}"
            else:
                merged.append(stripped)
        return merged or [text.strip()]
