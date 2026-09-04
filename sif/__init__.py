"""SIF Insight Console - semantic analysis of UA/UC and near-miss reports.

Oil India Limited, Problem Statement 26165.

The package implements the pipeline described in :mod:`sif.pipeline`: a
transformer sentence encoder feeding three heads (SIF classification, IOGP rule
classification, entity extraction), an evidence engine, a risk score, and
corpus-level pattern detection with a human-review queue.

Typical use::

    from sif import SIFPipeline

    pipeline = SIFPipeline()
    result = pipeline.analyze("No harness worn on the scaffold at 6 m.")
    intelligence = pipeline.aggregate([result])
"""

from .encoders import DEFAULT_MODEL, HashingEncoder, TransformerEncoder, load_encoder
from .lexical import SEED_REPORTS, LexicalEngine, SIFEngine
from .patterns import Hotspot, PatternDetector
from .pipeline import Intelligence, PipelineResult, SIFPipeline
from .review import ReviewItem, ReviewQueue
from .scoring import RiskScore, RiskScorer

__version__ = "2.0.0"

__all__ = [
    "SIFPipeline", "PipelineResult", "Intelligence",
    "LexicalEngine", "SIFEngine", "SEED_REPORTS",
    "TransformerEncoder", "HashingEncoder", "load_encoder", "DEFAULT_MODEL",
    "PatternDetector", "Hotspot", "ReviewQueue", "ReviewItem",
    "RiskScorer", "RiskScore", "__version__",
]
