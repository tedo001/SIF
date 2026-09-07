"""The functional map: every capability, in the order an operator uses it.

The console has grown several independent capabilities - OCR, translation, the
rule and semantic analysers, the learned model, hotspots, the review queue. On
its own each one is a page; what was missing was the *map*: what feeds what, what
is ready right now, and what to press next.

:class:`WorkflowMap` renders that as eight linked stages. Each stage shows a live
status line and runs its own action, so the map is the control surface, not a
picture of one.
"""

from __future__ import annotations

from typing import Dict, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.theme import C

__all__ = ["STAGES", "StageCard", "WorkflowMap"]

#: ``(key, title, what it does, button label)`` in operator order.
STAGES: Tuple[Tuple[str, str, str, str], ...] = (
    ("ingest", "1. Ingest",
     "Paste text, import a CSV export, or add PDFs, scans and photographs.",
     "Add documents"),
    ("ocr", "2. Read the document",
     "PDF text layer first; PaddleOCR for scans, in English, Hindi, Marathi, "
     "Tamil, Telugu, Kannada and Urdu.",
     "Check OCR"),
    ("translate", "3. Translate",
     "A local Ollama model renders a non-English report into English so the "
     "analysers can read it. Skipped for English.",
     "Check Ollama"),
    ("analyse", "4. Analyse",
     "Rules, sentence encoder and, when trained, the model: SIF potential, IOGP "
     "rule, activity, location, failed barrier.",
     "Load seed and analyse"),
    ("dashboard", "5. Dashboard",
     "Headline metrics and the three exposure charts for the whole corpus.",
     "Open dashboard"),
    ("hotspots", "6. Hotspots",
     "Sites and activities ranked by SIF-precursor density, not by volume.",
     "Open hotspots"),
    ("review", "7. Human review",
     "Disagreements, critical risk and thin evidence queued for an expert. "
     "Nothing is closed automatically.",
     "Open review queue"),
    ("train", "8. Learn",
     "Reviewed decisions become labels; XGBoost trains and the run is logged to "
     "MLflow.",
     "Train model"),
)


class StageCard(QFrame):
    """One stage: title, what it does, live status, and its action."""

    activated = pyqtSignal(str)

    def __init__(self, key: str, title: str, description: str, action: str) -> None:
        super().__init__()
        self.key = key
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(186)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {C.TEXT}; font-size: 13.5px; font-weight: 700;")

        body = QLabel(description)
        body.setWordWrap(True)
        body.setStyleSheet(f"color: {C.TEXT_DIM}; font-size: 11px;")
        body.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.status = QLabel("not run yet")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {C.TEXT_FAINT}; font-size: 11px; font-weight: 600;")

        self.button = QPushButton(action)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.clicked.connect(lambda: self.activated.emit(self.key))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        layout.addWidget(title_label)
        layout.addWidget(body, stretch=1)
        layout.addWidget(self.status)
        layout.addWidget(self.button)

    def set_status(self, text: str, tone: str = C.TEXT_FAINT) -> None:
        """Update the stage's live status line."""
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {tone}; font-size: 11px; font-weight: 600;")


class WorkflowMap(QWidget):
    """Eight linked stage cards - the console's capabilities, wired in order."""

    stage_activated = pyqtSignal(str)

    COLUMNS = 4

    def __init__(self) -> None:
        super().__init__()
        self.cards: Dict[str, StageCard] = {}

        heading = QLabel("How the console works, end to end")
        heading.setObjectName("SectionTitle")
        caption = QLabel(
            "Each box is a capability and its own control. Statuses are live: they "
            "show what is ready on this machine right now.")
        caption.setObjectName("Faint")
        caption.setWordWrap(True)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(12)
        for index, (key, title, description, action) in enumerate(STAGES):
            row, column = divmod(index, self.COLUMNS)
            card = StageCard(key, title, description, action)
            card.activated.connect(self.stage_activated.emit)
            self.cards[key] = card
            grid.addWidget(card, row * 2, column * 2)
            if column < self.COLUMNS - 1 and index + 1 < len(STAGES):
                grid.addWidget(self._arrow("→"), row * 2, column * 2 + 1)
        # The wrap from stage 4 to stage 5 continues down the left edge.
        grid.addWidget(self._arrow("↓"), 1, 0)
        for column in range(self.COLUMNS):
            grid.setColumnStretch(column * 2, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        layout.addWidget(heading)
        layout.addWidget(caption)
        layout.addLayout(grid, stretch=1)
        layout.addWidget(self._legend())

    @staticmethod
    def _arrow(glyph: str) -> QLabel:
        label = QLabel(glyph)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"color: {C.ACCENT}; font-size: 16px; font-weight: 700;")
        label.setFixedWidth(22)
        return label

    @staticmethod
    def _legend() -> QWidget:
        widget = QFrame()
        widget.setObjectName("Tile")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(18)
        for text, tone in (
            ("Ready - the capability is available on this machine", C.OK),
            ("Waiting - needs input or a previous stage", C.WARN),
            ("Unavailable - dependency or model missing; the console still runs", C.DANGER),
        ):
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {tone}; font-size: 11px;")
            label = QLabel(text)
            label.setStyleSheet(f"color: {C.TEXT_DIM}; font-size: 11px;")
            layout.addWidget(dot)
            layout.addWidget(label)
        layout.addStretch(1)
        return widget

    def set_status(self, key: str, text: str, tone: str = C.TEXT_FAINT) -> None:
        """Update one stage's status line."""
        card = self.cards.get(key)
        if card is not None:
            card.set_status(text, tone)
