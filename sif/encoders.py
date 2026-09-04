"""Stage 2 - semantic encoder.

The pipeline is written against a small interface, :class:`SemanticEncoder`, so
the transformer is a component rather than a hard dependency of every stage.

Two implementations ship:

``TransformerEncoder``
    A sentence-transformers bi-encoder (default ``all-MiniLM-L6-v2``, 384-d).
    Weights are fetched from the model hub on first use and cached by
    ``sentence-transformers`` thereafter, or loaded from a local directory when
    the machine has no internet access.

``HashingEncoder``
    A dependency-light, fully deterministic fallback: signed feature hashing
    over word unigrams, bigrams and character 4-grams. It is not a language
    model - it captures lexical overlap, not meaning - but it keeps the exact
    same interface, so the application runs (and the test suite passes) on a
    machine with no model, no network and no torch.

:func:`load_encoder` resolves which one to use and always returns a working
encoder; a failed transformer load degrades to hashing with the reason recorded
in :attr:`EncoderInfo.detail` rather than raising into the GUI.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "EncoderInfo",
    "SemanticEncoder",
    "TransformerEncoder",
    "HashingEncoder",
    "load_encoder",
    "DEFAULT_MODEL",
]

LOGGER = logging.getLogger(__name__)

#: Small, fast, widely used sentence embedding model (384 dimensions, ~90 MB).
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Environment overrides, so an air-gapped site can point at a local copy.
ENV_MODEL = "SIF_ENCODER_MODEL"
ENV_BACKEND = "SIF_ENCODER"  # "transformer" | "hashing" | "auto"


@dataclass(frozen=True)
class EncoderInfo:
    """What the running encoder is, for the status bar and the audit trail."""

    backend: str
    name: str
    dimension: int
    semantic: bool
    detail: str = ""

    def label(self) -> str:
        """One-line description, e.g. ``transformer: all-MiniLM-L6-v2 (384-d)``."""
        return f"{self.backend}: {self.name} ({self.dimension}-d)"


class SemanticEncoder:
    """Interface every encoder implements.

    Implementations return L2-normalised row vectors, so a dot product is the
    cosine similarity and the heads can compare scores across backends.
    """

    info: EncoderInfo

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix between two batches of unit vectors."""
        if left.size == 0 or right.size == 0:
            return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
        return np.clip(left @ right.T, -1.0, 1.0)


class TransformerEncoder(SemanticEncoder):
    """sentence-transformers bi-encoder.

    The model is loaded lazily on the first :meth:`encode` call - which the
    application makes on a worker thread - so importing this module never blocks
    the GUI or pulls torch into a process that will not use it.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None,
                 batch_size: int = 32) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()
        self.info = EncoderInfo(
            backend="transformer", name=model_name.rsplit("/", 1)[-1],
            dimension=0, semantic=True, detail="not loaded yet",
        )

    def load(self) -> None:
        """Load the weights. Raises if the model cannot be obtained."""
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer  # local import

            model = SentenceTransformer(self._model_name, device=self._device)
            # The accessor was renamed in sentence-transformers 6; support both.
            getter = getattr(model, "get_sentence_embedding_dimension", None) or \
                getattr(model, "get_embedding_dimension")
            dimension = int(getter())
            self._model = model
            self.info = EncoderInfo(
                backend="transformer", name=self._model_name.rsplit("/", 1)[-1],
                dimension=dimension, semantic=True,
                detail=f"device={model.device}",
            )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into L2-normalised rows."""
        items = [text if isinstance(text, str) and text.strip() else " " for text in texts]
        if not items:
            return np.zeros((0, max(self.info.dimension, 1)), dtype=np.float32)
        if self._model is None:
            self.load()
        vectors = self._model.encode(
            items, batch_size=self._batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEncoder(SemanticEncoder):
    """Deterministic offline fallback based on signed feature hashing.

    Word unigrams and bigrams carry most of the weight; character 4-grams give
    partial credit for morphology and typos ("barricaded" / "barricading").
    Vectors are L2-normalised and non-negative, so similarities land in [0, 1]
    and stay comparable with the transformer's - but they measure lexical
    overlap, not meaning: paraphrases with no shared wording score near zero,
    which is exactly why the lexical rule layer remains the decision backbone.
    """

    def __init__(self, dimension: int = 512, detail: str = "") -> None:
        self._dimension = dimension
        self.info = EncoderInfo(
            backend="hashing", name="lexical-hash", dimension=dimension,
            semantic=False, detail=detail or "deterministic offline fallback",
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text or ""):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "big") % self._dimension
                # Unsigned accumulation keeps similarities in [0, 1], so a
                # collision can only ever overstate overlap, never invert it.
                vectors[row, index] += weight
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)

    @staticmethod
    def _features(text: str) -> List[tuple]:
        """Yield (feature, weight) pairs for one text."""
        lowered = " ".join(text.lower().split())
        words = [word for word in lowered.replace("/", " ").split() if word]
        features: List[tuple] = [(f"w:{word}", 1.0) for word in words]
        features += [(f"b:{first}_{second}", 0.8)
                     for first, second in zip(words, words[1:])]
        padded = f" {lowered} "
        features += [(f"c:{padded[i:i + 4]}", 0.25)
                     for i in range(max(len(padded) - 3, 0))]
        return features


def load_encoder(backend: str = "auto", model_name: Optional[str] = None) -> SemanticEncoder:
    """Resolve and return a *working* encoder.

    Resolution is eager: the transformer's weights are loaded here, not on the
    first ``encode`` call, so a machine with no model degrades to the fallback
    at a predictable moment instead of raising in the middle of a batch. Call
    this off the GUI thread - the pipeline does.

    Parameters
    ----------
    backend:
        ``"transformer"``, ``"hashing"`` or ``"auto"``. ``auto`` prefers the
        transformer and silently degrades to hashing when it is unavailable, and
        is the only value the ``SIF_ENCODER`` environment variable can redirect -
        an explicit backend is never overridden by the environment.
    model_name:
        Model id or local directory; defaults to :data:`DEFAULT_MODEL` or the
        ``SIF_ENCODER_MODEL`` environment variable.
    """
    # An explicit argument always wins; the environment only sets the default,
    # so a deployment can pin the backend without silently overriding callers
    # that ask for a specific one.
    backend = (backend or "auto").strip().lower()
    if backend == "auto":
        backend = (os.environ.get(ENV_BACKEND) or "auto").strip().lower()
    model_name = model_name or os.environ.get(ENV_MODEL) or DEFAULT_MODEL

    if backend == "hashing":
        return HashingEncoder(detail="selected explicitly")

    encoder = TransformerEncoder(model_name)
    try:
        encoder.load()
        return encoder
    except Exception as exc:  # noqa: BLE001 - any load failure must degrade, not crash
        reason = f"{type(exc).__name__}: {exc}".split("\n", 1)[0][:180]
        if backend == "transformer":
            raise
        LOGGER.warning("Transformer encoder unavailable (%s); using hashing fallback", reason)
        return HashingEncoder(detail=f"transformer unavailable - {reason}")
