"""Entry point for the SIF Insight Console desktop application.

Oil India Limited - Problem Statement 26165.

Run with::

    python app.py

Module map
----------
``sif/``
    The analysis stack, independent of Qt:
    ``ocr`` (read PDFs, scans and photographs) ->
    ``preprocessing`` (clean, expand abbreviations, segment) ->
    ``encoders`` (transformer sentence embeddings, offline fallback) ->
    ``heads`` (SIF classifier, IOGP rule classifier, entity extraction) ->
    ``evidence`` -> ``scoring`` -> ``patterns`` / ``review``, orchestrated by
    ``pipeline.SIFPipeline``. ``lexical`` holds the deterministic rule layer every
    stage is anchored to, ``mlops`` adds the XGBoost model with MLflow tracking,
    and ``logging_setup`` provides the audit trail shown in Settings.
``ui/``
    Presentation only: ``theme`` (palette and style sheet), ``charts`` (painted
    bar and donut widgets), ``components`` (KPI tiles, panels, tables, sidebar)
    and ``views`` (the pages).
``main.py``
    The controller: owns the pipeline, MLOps service and document extractor, runs
    them on ``QThread`` workers, and pushes results into the views.
``app.py``
    This launcher: environment checks, application bootstrap, event loop.

Encoder selection
-----------------
The console picks its encoder at run time and can be steered without code
changes::

    SIF_ENCODER=transformer|hashing|auto   # default: auto
    SIF_ENCODER_MODEL=<hub id or local dir>  # default: all-MiniLM-L6-v2

``auto`` uses the transformer when it can be loaded and otherwise falls back to
the offline lexical engine, so the application always starts.

Optional components (XGBoost, MLflow, PaddleOCR) are all detected at run time;
the Settings tab reports which are available and why, and the console works
without any of them.
"""

from __future__ import annotations

import sys


def _require_pyqt6() -> None:
    """Fail fast with an actionable message if PyQt6 is not installed."""
    try:
        import PyQt6  # noqa: F401  (import is the check)
    except ImportError:
        sys.stderr.write(
            "PyQt6 is required to run the SIF Insight Console.\n"
            "Install it with:\n\n    pip install -r requirements.txt\n\n"
            "or:\n\n    pip install PyQt6\n"
        )
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    """Launch the desktop application and return the Qt exit code."""
    _require_pyqt6()

    # Imported after the guard so the error message above stays readable.
    from main import MainWindow, create_application

    app = create_application(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
