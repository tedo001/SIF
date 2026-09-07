"""Presentation layer for the console's second build (``app2.py``).

Two differences from :mod:`ui`, which ``app.py`` keeps using unchanged:

* **No pictographic icons.** Every label is words or a typographic mark, so the
  interface renders identically on a plant workstation with no emoji font and
  reads as instrumentation rather than chat.
* **A workflow map.** The first page is the functional map of the whole system -
  ingest, OCR, analyse, dashboard, review, train - showing which stages are ready
  and letting the operator run each one in order.

Shared, emoji-free widgets (panels, tables, KPI tiles, charts) are imported from
:mod:`ui` rather than duplicated.
"""

from .components import HeaderBar, Sidebar
from .workflow import STAGES, WorkflowMap

__all__ = ["HeaderBar", "Sidebar", "WorkflowMap", "STAGES"]
