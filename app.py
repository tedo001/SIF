"""Entry point for the SIF Insight Console desktop application.

Oil India Limited - Problem Statement 26165.

Run with::

    python app.py

Module map
----------
``sif/``
    The analysis pipeline, independent of Qt:
    ``preprocessing`` (clean, expand abbreviations, segment) ->
    ``encoders`` (transformer sentence embeddings, offline fallback) ->
    ``heads`` (SIF classifier, IOGP rule classifier, entity extraction) ->
    ``evidence`` -> ``scoring`` -> ``patterns`` / ``review``, orchestrated by
    ``pipeline.SIFPipeline``. ``lexical`` holds the deterministic rule layer that
    every stage is anchored to.
``main.py``
    PyQt6 presentation layer: control panel, KPI dashboard, incident matrix,
    hotspot and review panels, and the :class:`~main.AnalysisWorker` ``QThread``
    that keeps the UI responsive while the model loads and runs.
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
