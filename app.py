"""SENTRA - deep navy build. Oil India Limited, Problem Statement 26165.

Run with::

    python app.py

The same console as ``app2.py``, in the alternative design: a near-black navy
ground, teal as the interactive colour, hairline borders and a navigation item
that tints rather than fills. Every capability is the one implementation - the
workflow map, multilingual OCR, translation, the dashboard, hotspots, the human
review bench, analytics, training and the audit trail - so a fix to any of them
lands in both builds at once and neither can drift into being a stale copy of
the other.

Why a skin rather than a second application: duplicating the controller and its
pages to restyle them would mean two copies of every rule about when a report
reaches a person, and the copies would disagree within a week. The re-skin is
delivered in two halves instead, because a Qt style sheet alone cannot reach a
widget that paints itself:

1. :func:`ui.theme.apply_palette` repoints the shared colours *before* the
   window is built, which catches the badges, KPI values, chart series and the
   brand mark.
2. :data:`ui.gov_theme.STYLESHEET` is set on the window afterwards, and carries
   the structure - corner radii, the tinted selection, the flat scroll bars.

``app2.py`` is untouched and still launches the console's own look.

Module map
----------
``sif/``
    The analysis stack, independent of Qt: ``ocr`` -> ``preprocessing`` ->
    ``encoders`` -> ``heads`` -> ``evidence`` -> ``scoring`` -> ``patterns`` /
    ``review``, orchestrated by ``pipeline.SIFPipeline``. ``lexical`` holds the
    deterministic rule layer, ``llm`` the optional local model, ``narrative``
    the generated briefs and bulletins, ``mlops`` the XGBoost model and
    ``audit`` the append-only trail.
``ui/``
    Shared presentation: ``theme`` (the console's own look), ``gov_theme``
    (this one), ``charts``, ``components``.
``ui2/``
    The pages: navigation, workflow map, review bench, dashboards, settings.
``main2.py``
    The controller both builds run on.

Encoder selection
-----------------
    SIF_ENCODER=transformer|hashing|auto     # default: auto
    SIF_ENCODER_MODEL=<hub id or local dir>  # default: all-MiniLM-L6-v2

``auto`` uses the transformer when it loads and falls back to the offline
lexical engine otherwise, so the application always starts. XGBoost, MLflow,
PaddleOCR and Ollama are all detected at run time and the console runs without
any of them.
"""

from __future__ import annotations

import sys

#: Shown in the title bar, so an operator running both builds side by side can
#: tell which window is which.
WINDOW_TITLE = "SENTRA - deep navy"


def _require_pyqt6() -> None:
    """Fail fast with an actionable message if PyQt6 is not installed."""
    try:
        import PyQt6  # noqa: F401  (import is the check)
    except ImportError:
        sys.stderr.write(
            "PyQt6 is required to run SENTRA.\n"
            "Install it with:\n\n    pip install -r requirements.txt\n\n"
            "or:\n\n    pip install PyQt6\n"
        )
        raise SystemExit(1)


def build_window():
    """Construct the console wearing the deep-navy skin.

    The palette is applied first and the window built second: that order is the
    whole mechanism, because every widget that styles itself reads the palette
    in its constructor.
    """
    from ui import gov_theme
    from ui.theme import apply_palette

    apply_palette(gov_theme.PALETTE)

    from main2 import MainWindow

    window = MainWindow()
    window.setStyleSheet(gov_theme.STYLESHEET)
    window.setWindowTitle(WINDOW_TITLE)
    return window


def main(argv: list[str] | None = None) -> int:
    """Launch the desktop application and return the Qt exit code."""
    _require_pyqt6()

    from main2 import create_application

    application = create_application(argv if argv is not None else sys.argv)
    window = build_window()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
