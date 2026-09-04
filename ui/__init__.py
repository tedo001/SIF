"""Presentation layer for the SIF Insight Console (PyQt6).

``theme`` holds the palette and style sheet, ``charts`` the painted chart
widgets, ``components`` the reusable pieces (KPI tiles, panels, tables,
navigation) and ``views`` the pages themselves. Nothing in this package touches
the pipeline: :mod:`main` wires views to workers.
"""

from .theme import BAND_COLORS, C, STYLESHEET

__all__ = ["C", "STYLESHEET", "BAND_COLORS"]
