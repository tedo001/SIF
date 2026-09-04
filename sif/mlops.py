"""MLOps layer - supervised XGBoost model with MLflow experiment tracking.

Where the rest of the pipeline is deterministic and hand-authored, this layer is
the part that *learns*. It sits alongside the rule and semantic paths as a third
opinion:

1. :func:`featurise` turns a :class:`~sif.pipeline.PipelineResult` into a fixed,
   interpretable feature vector - the two SIF factors, the energy and barrier
   families as multi-hot flags, the severity weights, the rule one-hot and a few
   text statistics. Nothing opaque: every column has a name, and feature
   importance therefore reads as safety language, not as ``f37``.
2. :class:`SIFModel` trains an ``XGBClassifier`` over those vectors with
   stratified cross-validation, and persists as a JSON booster plus metadata.
3. :class:`MLflowTracker` logs each run - parameters, metrics, feature
   importances, the model artifact - to a local MLflow store, so training is
   reproducible and comparable rather than a one-off notebook.
4. :class:`MLOpsService` is the façade the Settings tab drives.

On labels
---------
Most sites start with no labelled corpus, so training defaults to the pipeline's
own verdicts as weak labels. Be clear-eyed about what that is: **distillation**.
The model learns to reproduce the rules, which is useful for speed and for
finding where the rules are internally inconsistent - but it adds no knowledge
until real reviewed outcomes replace those labels. Supply a ``label`` column (or
pass ``labels=``) as soon as the review queue has produced verified decisions,
which is exactly what :mod:`sif.review` is for.

Every component degrades gracefully: without ``xgboost`` the console runs with
the rule and semantic paths only, and without ``mlflow`` training still works
and simply is not tracked.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lexical import CRITICAL_BARRIERS, HIGH_ENERGY_SOURCES, IOGP_RULES
from .pipeline import PipelineResult
from .scoring import BARRIER_CRITICALITY, DEFAULT_WEIGHT, ENERGY_SEVERITY

__all__ = ["FEATURE_NAMES", "featurise", "featurise_many", "TrainingReport",
           "SIFModel", "MLflowTracker", "MLOpsService", "DEFAULT_PARAMS",
           "adapt_params"]

LOGGER = logging.getLogger(__name__)

ENERGY_LABELS = tuple(signature.label for signature in HIGH_ENERGY_SOURCES)
BARRIER_LABELS = tuple(signature.label for signature in CRITICAL_BARRIERS)
RULE_LABELS = tuple(rule.name for rule in IOGP_RULES)

#: Column names, in vector order. Kept explicit so importances stay readable.
FEATURE_NAMES: Tuple[str, ...] = (
    "p_sif", "rule_confidence", "extraction_confidence",
    "high_energy", "barrier_failed", "semantic_active", "lexical_flag", "semantic_flag",
    "energy_severity", "barrier_criticality",
    "n_chars", "n_words", "n_sentences",
    *(f"energy::{label}" for label in ENERGY_LABELS),
    *(f"barrier::{label}" for label in BARRIER_LABELS),
    *(f"rule::{label}" for label in RULE_LABELS),
)

#: Corpora below this size get relaxed regularisation - see :func:`adapt_params`.
SMALL_CORPUS = 200

#: Defaults for a production-sized, tabular, class-imbalanced problem.
DEFAULT_PARAMS: Dict[str, object] = {
    "n_estimators": 220,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1.0,
    "reg_lambda": 1.5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
}

MODEL_DIRECTORY = "models"
MODEL_FILE = "sif_xgboost.json"
METADATA_FILE = "sif_xgboost.meta.json"
#: MLflow 3 put the filesystem store into maintenance mode, so the default is
#: the SQLite backend it recommends - still a single local file, no server to run.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "sif-insight-console"


def adapt_params(params: Dict[str, object], samples: int, positives: int,
                 warnings: List[str]) -> Dict[str, object]:
    """Relax regularisation when the corpus is too small to satisfy it.

    ``min_child_weight`` is a floor on the summed hessian in a leaf; for binary
    logistic that is about 0.25 per sample at the first split, so the default of
    1.0 silently forbids any split that isolates fewer than ~4 reports. On a
    pilot corpus with a handful of one class, XGBoost then returns split-less
    trees and a constant probability - a model that looks trained and predicts
    nothing. Scaling the floor to the minority class keeps small runs honest.
    """
    adapted = dict(params)
    minority = min(positives, samples - positives)
    if samples < SMALL_CORPUS or minority < 8:
        adapted["min_child_weight"] = 0.1
        adapted["n_estimators"] = min(int(adapted.get("n_estimators", 220)), 150)
        adapted["learning_rate"] = 0.15
        adapted["max_depth"] = 3
        warnings.append(
            f"Small corpus ({samples} reports, {minority} in the minority class): "
            "regularisation relaxed so the trees can split. Metrics from this few "
            "samples are indicative only.")
    return adapted


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _weight(table: Dict[str, float], label: str) -> float:
    weights = [weight for key, weight in table.items() if key in label]
    return max(weights) if weights else DEFAULT_WEIGHT


def featurise(result: PipelineResult) -> np.ndarray:
    """Turn one pipeline result into the model's feature vector."""
    text = result.raw_text or ""
    energy_label = result.energy_source or ""
    barrier_label = result.barrier_failure or ""

    base = [
        float(result.p_sif),
        float(result.rule_confidence),
        float(result.confidence),
        float(result.high_energy),
        float(result.barrier_failed),
        float(result.semantic_active),
        float(result.lexical_flag),
        float(result.semantic_flag),
        _weight(ENERGY_SEVERITY, energy_label) if result.high_energy else 0.0,
        _weight(BARRIER_CRITICALITY, barrier_label) if result.barrier_failed else 0.0,
        float(len(text)),
        float(len(text.split())),
        float(text.count(".") + text.count(";") + 1),
    ]
    energies = [float(label in energy_label) for label in ENERGY_LABELS]
    barriers = [float(label in barrier_label) for label in BARRIER_LABELS]
    rules = [float(result.iogp_rule == label) for label in RULE_LABELS]
    return np.asarray(base + energies + barriers + rules, dtype=np.float32)


def featurise_many(results: Sequence[PipelineResult]) -> np.ndarray:
    """Stack feature vectors for a corpus (shape ``(n, len(FEATURE_NAMES))``)."""
    if not results:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([featurise(result) for result in results])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class TrainingReport:
    """Outcome of one training run."""

    samples: int
    positives: int
    metrics: Dict[str, float] = field(default_factory=dict)
    importances: List[Tuple[str, float]] = field(default_factory=list)
    params: Dict[str, object] = field(default_factory=dict)
    model_path: str = ""
    run_id: str = ""
    label_source: str = "weak (pipeline verdicts)"
    warnings: List[str] = field(default_factory=list)
    trained_at: str = ""

    def summary(self) -> str:
        """One line for the status bar."""
        scores = " ".join(f"{key}={value:.3f}" for key, value in sorted(self.metrics.items()))
        return f"{self.samples} samples ({self.positives} positive) | {scores}"

    def to_dict(self) -> Dict[str, object]:
        payload = dict(self.__dict__)
        payload["importances"] = [list(item) for item in self.importances]
        return payload


class SIFModel:
    """XGBoost classifier over the interpretable feature vector."""

    def __init__(self, params: Optional[Dict[str, object]] = None) -> None:
        self.params: Dict[str, object] = {**DEFAULT_PARAMS, **(params or {})}
        self._booster = None
        self.metadata: Dict[str, object] = {}

    # -- availability ------------------------------------------------------

    @staticmethod
    def installed() -> bool:
        """True when ``xgboost`` can be imported."""
        try:
            import xgboost  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    @property
    def is_trained(self) -> bool:
        return self._booster is not None

    # -- training ----------------------------------------------------------

    def train(self, results: Sequence[PipelineResult],
              labels: Optional[Sequence[int]] = None,
              label_source: str = "weak (pipeline verdicts)") -> TrainingReport:
        """Fit the model and return its cross-validated report."""
        if not self.installed():
            raise RuntimeError("xgboost is not installed (pip install xgboost)")
        if len(results) < 4:
            raise ValueError("At least 4 analysed reports are needed to train")

        import xgboost as xgb

        features = featurise_many(results)
        targets = np.asarray(
            [int(item.sif_potential) for item in results] if labels is None
            else [int(value) for value in labels], dtype=int)
        if len(targets) != len(results):
            raise ValueError("labels and results must be the same length")

        warnings: List[str] = []
        classes = set(targets.tolist())
        if len(classes) < 2:
            raise ValueError(
                "Training needs both SIF-potential and non-SIF reports; this corpus "
                f"has only class {classes.pop()}")

        positives = int(targets.sum())
        scale = float(len(targets) - positives) / max(positives, 1)
        params = adapt_params({**self.params, "scale_pos_weight": round(scale, 3)},
                              len(targets), positives, warnings)

        metrics = self._cross_validate(features, targets, params, warnings)
        model = xgb.XGBClassifier(**params)
        model.fit(features, targets)
        self._booster = model
        importances = sorted(
            ((name, float(value)) for name, value in
             zip(FEATURE_NAMES, model.feature_importances_) if value > 0),
            key=lambda item: item[1], reverse=True)

        self.metadata = {
            "features": list(FEATURE_NAMES),
            "params": params,
            "samples": int(len(targets)),
            "positives": positives,
            "metrics": metrics,
            "label_source": label_source,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        LOGGER.info("Trained XGBoost model on %d samples (%d positive): %s",
                    len(targets), positives,
                    " ".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items())))
        return TrainingReport(
            samples=int(len(targets)), positives=positives, metrics=metrics,
            importances=importances[:15], params=params, label_source=label_source,
            warnings=warnings, trained_at=str(self.metadata["trained_at"]))

    def _cross_validate(self, features: np.ndarray, targets: np.ndarray,
                        params: Dict[str, object], warnings: List[str]) -> Dict[str, float]:
        """Stratified k-fold metrics, degrading sensibly on tiny corpora."""
        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold

        minority = int(min(np.bincount(targets)))
        folds = min(5, minority)
        if folds < 2:
            warnings.append(
                "Too few reports in one class for cross-validation - metrics are "
                "in-sample and will be optimistic.")
            model = xgb.XGBClassifier(**params).fit(features, targets)
            predicted = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]
            return self._score(targets, predicted, probabilities)

        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)
        predicted = np.zeros_like(targets)
        probabilities = np.zeros(len(targets), dtype=float)
        for train_index, test_index in splitter.split(features, targets):
            model = xgb.XGBClassifier(**params)
            model.fit(features[train_index], targets[train_index])
            predicted[test_index] = model.predict(features[test_index])
            probabilities[test_index] = model.predict_proba(features[test_index])[:, 1]
        if folds < 5:
            warnings.append(f"Only {folds}-fold cross-validation was possible for this corpus.")
        return self._score(targets, predicted, probabilities)

    @staticmethod
    def _score(targets, predicted, probabilities) -> Dict[str, float]:
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, roc_auc_score)

        metrics = {
            "accuracy": float(accuracy_score(targets, predicted)),
            "precision": float(precision_score(targets, predicted, zero_division=0)),
            "recall": float(recall_score(targets, predicted, zero_division=0)),
            "f1": float(f1_score(targets, predicted, zero_division=0)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
        except ValueError:  # single class in a fold
            metrics["roc_auc"] = float("nan")
        return {key: round(value, 4) for key, value in metrics.items()}

    # -- inference ---------------------------------------------------------

    def predict(self, result: PipelineResult) -> float:
        """Probability that ``result`` is SIF-potential according to the model."""
        if not self.is_trained:
            raise RuntimeError("The model has not been trained or loaded")
        vector = featurise(result).reshape(1, -1)
        return float(self._booster.predict_proba(vector)[0, 1])

    # -- persistence -------------------------------------------------------

    def save(self, directory: str = MODEL_DIRECTORY) -> str:
        """Write the booster and its metadata; returns the model path."""
        if not self.is_trained:
            raise RuntimeError("Nothing to save - train the model first")
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, MODEL_FILE)
        self._booster.save_model(model_path)
        with open(os.path.join(directory, METADATA_FILE), "w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2)
        LOGGER.info("Saved model to %s", model_path)
        return model_path

    def load(self, directory: str = MODEL_DIRECTORY) -> bool:
        """Load a previously saved model; returns False when none exists."""
        model_path = os.path.join(directory, MODEL_FILE)
        if not os.path.isfile(model_path) or not self.installed():
            return False
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(model_path)
        self._booster = model
        metadata_path = os.path.join(directory, METADATA_FILE)
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        LOGGER.info("Loaded model from %s", model_path)
        return True


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------


class MLflowTracker:
    """Logs training runs to MLflow, or degrades to a no-op when it is absent."""

    def __init__(self, tracking_uri: str = DEFAULT_TRACKING_URI,
                 experiment: str = DEFAULT_EXPERIMENT) -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment

    @staticmethod
    def installed() -> bool:
        try:
            import mlflow  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def status(self) -> str:
        """One line for the Settings tab."""
        if not self.installed():
            return "MLflow not installed - training runs will not be tracked (pip install mlflow)"
        return f"MLflow tracking to {self.tracking_uri} (experiment '{self.experiment}')"

    def _connect(self):
        """Point MLflow at the configured store and return the module."""
        import mlflow

        if self.tracking_uri.startswith("file:"):
            # Opt back in to the legacy store when a site explicitly asks for it.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(self.tracking_uri)
        return mlflow

    def log_training(self, report: TrainingReport, model_path: str = "") -> str:
        """Log one run; returns the MLflow run id, or '' when unavailable."""
        if not self.installed():
            LOGGER.info("MLflow unavailable - skipping run logging")
            return ""
        mlflow = self._connect()
        mlflow.set_experiment(self.experiment)
        with mlflow.start_run() as run:
            mlflow.log_params({key: value for key, value in report.params.items()})
            mlflow.log_param("label_source", report.label_source)
            mlflow.log_param("samples", report.samples)
            mlflow.log_param("positives", report.positives)
            mlflow.log_metrics({key: value for key, value in report.metrics.items()
                                if value == value})  # skip NaN
            if report.importances:
                mlflow.log_dict(
                    {name: score for name, score in report.importances},
                    "feature_importances.json")
            if model_path and os.path.isfile(model_path):
                mlflow.log_artifact(model_path, artifact_path="model")
            run_id = run.info.run_id
        LOGGER.info("Logged MLflow run %s to experiment '%s'", run_id, self.experiment)
        return run_id

    def recent_runs(self, limit: int = 10) -> List[Dict[str, object]]:
        """Return recent runs of the experiment, newest first."""
        if not self.installed():
            return []
        mlflow = self._connect()
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment)
        if experiment is None:
            return []
        runs = client.search_runs([experiment.experiment_id],
                                  order_by=["attributes.start_time DESC"], max_results=limit)
        return [
            {
                "run_id": run.info.run_id[:8],
                "started": datetime.fromtimestamp(run.info.start_time / 1000)
                .strftime("%Y-%m-%d %H:%M:%S"),
                "status": run.info.status,
                "samples": run.data.params.get("samples", ""),
                "labels": run.data.params.get("label_source", ""),
                "f1": f'{float(run.data.metrics.get("f1", float("nan"))):.3f}'
                      if "f1" in run.data.metrics else "",
                "roc_auc": f'{float(run.data.metrics.get("roc_auc", float("nan"))):.3f}'
                           if "roc_auc" in run.data.metrics else "",
            }
            for run in runs
        ]


# ---------------------------------------------------------------------------
# Façade
# ---------------------------------------------------------------------------


class MLOpsService:
    """What the Settings tab drives: train, track, persist, predict."""

    def __init__(self, model_directory: str = MODEL_DIRECTORY,
                 tracking_uri: str = DEFAULT_TRACKING_URI,
                 experiment: str = DEFAULT_EXPERIMENT) -> None:
        self.model_directory = model_directory
        self.model = SIFModel()
        self.tracker = MLflowTracker(tracking_uri, experiment)
        self.last_report: Optional[TrainingReport] = None

    def load_existing(self) -> bool:
        """Load a saved model at start-up, if there is one."""
        try:
            return self.model.load(self.model_directory)
        except Exception as exc:  # noqa: BLE001 - a corrupt model must not stop the app
            LOGGER.warning("Could not load saved model: %s", exc)
            return False

    def train(self, results: Sequence[PipelineResult],
              labels: Optional[Sequence[int]] = None,
              label_source: Optional[str] = None) -> TrainingReport:
        """Train, persist and log one run over the analysed corpus."""
        source = label_source or ("supplied labels" if labels is not None
                                  else "weak (pipeline verdicts)")
        report = self.model.train(results, labels=labels, label_source=source)
        report.model_path = self.model.save(self.model_directory)
        report.run_id = self.tracker.log_training(report, report.model_path)
        self.last_report = report
        return report

    def predict(self, result: PipelineResult) -> Optional[float]:
        """Model probability for one result, or ``None`` when no model is loaded."""
        if not self.model.is_trained:
            return None
        try:
            return round(self.model.predict(result), 3)
        except Exception as exc:  # noqa: BLE001 - inference must never break analysis
            LOGGER.warning("Model inference failed: %s", exc)
            return None

    def status(self) -> Dict[str, str]:
        """Capability and state summary for the Settings tab."""
        if not SIFModel.installed():
            model_state = "xgboost not installed (pip install xgboost)"
        elif self.model.is_trained:
            metrics = self.model.metadata.get("metrics", {})
            trained = self.model.metadata.get("trained_at", "unknown time")
            scores = " ".join(f"{key}={value}" for key, value in sorted(metrics.items()))
            model_state = f"trained {trained} | {scores}"
        else:
            model_state = "no model trained yet"
        return {"model": model_state, "tracking": self.tracker.status(),
                "directory": os.path.abspath(self.model_directory)}
