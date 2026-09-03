"""Entry point for the SIF Insight Console desktop application.

Oil India Limited - Problem Statement 26165.

Run with::

    python app.py

Module map
----------
``sif_engine.py``
    Dependency-free heuristic parser (:class:`~sif_engine.SIFEngine`) that turns
    a raw UA/UC or near-miss narrative into structured safety insight and flags
    SIF potential.
``main.py``
    PyQt6 presentation layer: control panel, KPI dashboard, results matrix and
    the :class:`~main.AnalysisWorker` ``QThread`` that keeps the UI responsive.
``app.py``
    This launcher: environment checks, application bootstrap, event loop.
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
