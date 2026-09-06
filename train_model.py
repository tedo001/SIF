"""Command-line trainer for the SIF risk model.

The Settings tab trains on whatever is on screen, which is right for a demo but
wrong for a real corpus: production training runs from a file, is repeatable,
and records where its labels came from.

    # Distil the rules (no labels yet) - useful as a baseline only
    python train_model.py data/reports.csv

    # Train on reviewed labels, the run that actually adds knowledge
    python train_model.py data/reports.csv --label-column sif_label

    # Pin the encoder and the tracking store
    python train_model.py data/reports.csv --encoder hashing \\
        --tracking-uri sqlite:///mlflow.db --experiment oil-india-2026

Exit status is 0 on success and 1 on any refusal (missing file, unusable labels,
too few reports), so it can be wired into a scheduler or CI.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import List, Optional, Sequence, Tuple

from sif.lexical import CSV_TEXT_COLUMNS
from sif.logging_setup import configure_logging
from sif.mlops import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MLOpsService
from sif.pipeline import SIFPipeline

__all__ = ["read_corpus", "main", "LABEL_COLUMNS", "TRUE_VALUES", "FALSE_VALUES"]

LOGGER = logging.getLogger("sif.train")

#: Column names accepted as the reviewed outcome, in priority order.
LABEL_COLUMNS: Tuple[str, ...] = ("sif_label", "label", "reviewed_sif", "is_sif",
                                  "sif_potential")
TRUE_VALUES = {"1", "y", "yes", "true", "sif", "sif-potential", "positive"}
FALSE_VALUES = {"0", "n", "no", "false", "not sif", "non-sif", "negative", ""}


def _parse_label(raw: str, row_number: int) -> Optional[int]:
    """Turn a spreadsheet cell into 1, 0 or ``None`` (unlabelled)."""
    value = (raw or "").strip().lower()
    if value in TRUE_VALUES:
        return 1
    if value in FALSE_VALUES:
        return None if value == "" else 0
    LOGGER.warning("Row %d: unrecognised label %r - treated as unlabelled", row_number, raw)
    return None


def read_corpus(path: str, text_column: Optional[str] = None,
                label_column: Optional[str] = None):
    """Read narratives, references and labels from a CSV export.

    Returns ``(texts, references, labels)`` where ``labels`` is ``None`` when no
    label column was requested, and otherwise a list with ``None`` for rows the
    reviewer has not decided yet.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    texts: List[str] = []
    references: List[str] = []
    labels: List[Optional[int]] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")

        lookup = {(name or "").strip().lower(): name for name in reader.fieldnames}
        text_key = (lookup.get((text_column or "").strip().lower()) if text_column
                    else next((lookup[key] for key in CSV_TEXT_COLUMNS if key in lookup), None))
        if text_key is None:
            raise ValueError(
                f"No narrative column found in {path}. Expected one of "
                f"{', '.join(CSV_TEXT_COLUMNS)}, or pass --text-column.")

        label_key = None
        if label_column:
            label_key = lookup.get(label_column.strip().lower())
            if label_key is None:
                raise ValueError(f"Label column '{label_column}' is not in {path}")
        elif any(key in lookup for key in LABEL_COLUMNS):
            label_key = next(lookup[key] for key in LABEL_COLUMNS if key in lookup)
            LOGGER.info("Using '%s' as the label column (found automatically)", label_key)

        reference_key = next((lookup[key] for key in ("report_id", "id", "ref", "reference")
                              if key in lookup), None)

        for row_number, row in enumerate(reader, start=2):
            narrative = str(row.get(text_key) or "").strip()
            if not narrative:
                continue
            texts.append(narrative)
            references.append(str(row.get(reference_key, "")) if reference_key else "")
            if label_key:
                labels.append(_parse_label(str(row.get(label_key, "")), row_number))

    return texts, references, (labels if label_key else None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the SIF risk model from a CSV of field reports.")
    parser.add_argument("csv_path", help="CSV export of UA/UC and near-miss reports")
    parser.add_argument("--text-column", help="Column holding the narrative "
                                              "(default: auto-detect)")
    parser.add_argument("--label-column", help="Column holding the reviewed outcome; "
                                               "without it the pipeline's own verdicts are "
                                               "distilled")
    parser.add_argument("--encoder", default="auto",
                        choices=("auto", "transformer", "hashing"),
                        help="Semantic encoder to analyse with (default: auto)")
    parser.add_argument("--model-dir", default="models", help="Where to save the booster")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI,
                        help="MLflow tracking store")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse and report the corpus, but do not train")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the trainer; returns a process exit code."""
    args = _build_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        texts, references, labels = read_corpus(args.csv_path, args.text_column,
                                                args.label_column)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    if not texts:
        LOGGER.error("No usable narratives in %s", args.csv_path)
        return 1

    pipeline = SIFPipeline(backend=args.encoder)
    LOGGER.info("Encoder: %s", pipeline.warm_up())
    LOGGER.info("Analysing %d report(s)...", len(texts))
    results = pipeline.analyze_many(texts, references)

    if labels is not None:
        # Keep only rows a reviewer has actually decided.
        pairs = [(result, label) for result, label in zip(results, labels) if label is not None]
        if len(pairs) < len(results):
            LOGGER.info("%d of %d rows are labelled; the rest are ignored for training",
                        len(pairs), len(results))
        results = [result for result, _ in pairs]
        labels = [label for _, label in pairs]
        label_source = f"reviewed labels ({args.label_column or 'auto-detected column'})"
    else:
        label_source = "weak (pipeline verdicts)"

    positives = (sum(labels) if labels is not None
                 else sum(1 for item in results if item.sif_potential))
    LOGGER.info("Corpus: %d report(s), %d positive, labels = %s",
                len(results), positives, label_source)

    if args.dry_run:
        LOGGER.info("Dry run - stopping before training")
        return 0

    service = MLOpsService(model_directory=args.model_dir, tracking_uri=args.tracking_uri,
                           experiment=args.experiment)
    try:
        report = service.train(results, labels=labels, label_source=label_source)
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("Training refused: %s", exc)
        return 1

    for warning in report.warnings:
        LOGGER.warning("%s", warning)
    LOGGER.info("Trained: %s", report.summary())
    LOGGER.info("Model saved to %s", report.model_path)
    LOGGER.info("MLflow run: %s", report.run_id or "not tracked")
    LOGGER.info("Top features: %s",
                ", ".join(f"{name}={value:.3f}" for name, value in report.importances[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
