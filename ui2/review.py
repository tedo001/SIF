"""The human review bench.

Card 7 of the workflow map. Everything else in the console produces an opinion;
this is where a person turns opinions into a decision, and the decision into a
label the model can learn from.

The page is built around how the work is actually done. An HSE reviewer does not
browse: they work a queue, one report at a time, and the thing that decides
whether the queue gets worked is how many seconds each row costs. So:

* **The queue is on the left, the report is on the right.** Selecting a row shows
  the whole case - the narrative, the extracted fields, every opinion, the cues
  that produced it, and why this report was queued.
* **Three decisions, three keys.** Confirm, reject, or say it is unclear, on
  ``1``/``2``/``3``. After a decision the bench advances to the next report on
  its own, so a queue can be worked without touching the mouse.
* **Unclear is not a label.** It records that an expert looked and could not
  call it, which is a different fact from "not SIF" and must never be trained on.
* **Nothing is closed automatically** and nothing is hidden: the trail tab shows
  every decision ever recorded, including the ones that were changed.
* **Overturning the engine is first-class.** When the expert disagrees with the
  verdict, the row says so - those are the reports that teach the system most.

The view is passive. It renders what the controller gives it and emits signals;
it never touches the pipeline, the decision log or the disk.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.components import DataTable, FieldRow, Panel, Pill
from ui2.components import scrollable
from ui.theme import BAND_COLORS, C

__all__ = ["ReviewView", "QUEUE_COLUMNS", "TRAIL_COLUMNS", "DECISION_BUTTONS"]

# Widths are sized for the queue panel rather than for the text: the full reason
# is on every row as a tooltip and in the bench beside it, so the column carries
# as much as fits and no more.
QUEUE_COLUMNS: Sequence[Tuple[str, str, int]] = (
    ("Trigger", "trigger", 142),
    ("Ref", "reference", 76),
    ("Risk", "risk_score", 60),
    ("Engine", "sif_potential", 64),
    ("Status", "status", 92),
    ("Why a human is needed", "reason", 174),
)

TRAIL_COLUMNS: Sequence[Tuple[str, str, int]] = (
    ("Decided", "decided_short", 104),
    ("Ref", "reference", 78),
    ("Decision", "decision_label", 156),
    ("Outcome", "outcome", 116),
    ("Reviewer", "reviewer", 108),
    ("Note", "note", 260),
)

#: Button text, decision key, colour and shortcut - one row per decision.
DECISION_BUTTONS = (
    ("1  Confirm SIF potential", "confirmed", C.DANGER, "1"),
    ("2  Not SIF potential", "rejected", C.OK, "2"),
    ("3  Unclear - need more information", "unclear", C.WARN, "3"),
)


class ReviewView(QWidget):
    """Queue, case detail and decision controls on one page."""

    #: ``(decision, note)`` - the reviewer called the selected report.
    decision_made = pyqtSignal(str, str)
    undo_requested = pyqtSignal()
    export_requested = pyqtSignal()
    reviewer_changed = pyqtSignal(str)
    #: Row index within the currently displayed queue.
    row_selected = pyqtSignal(int)

    #: Below this the case detail scrolls; the decision bar never does.
    MIN_CASE_HEIGHT = 560

    def __init__(self) -> None:
        super().__init__()
        self._rows: List[Dict[str, object]] = []
        self._current = -1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        layout.addWidget(self._build_queue(), stretch=5)
        layout.addWidget(self._build_bench(), stretch=4)
        self._bind_shortcuts()
        self.set_case(None)

    # -- construction ------------------------------------------------------

    def _build_queue(self) -> QWidget:
        self.tabs = QTabWidget()

        queue_page = QWidget()
        queue_layout = QVBoxLayout(queue_page)
        queue_layout.setContentsMargins(0, 8, 0, 0)
        queue_layout.setSpacing(8)

        self.progress = QLabel("Nothing analysed yet.")
        self.progress.setObjectName("Muted")
        self.progress.setWordWrap(True)

        self.show_decided = QCheckBox("Show reports already decided")
        self.show_decided.setToolTip(
            "Decided reports leave the queue. Tick this to bring them back into "
            "view - to check a colleague's call, or to change your own.")

        self.table = DataTable(QUEUE_COLUMNS, on_select=self._on_row)
        queue_layout.addWidget(self.progress)
        queue_layout.addWidget(self.show_decided)
        queue_layout.addWidget(self.table, stretch=1)

        trail_page = QWidget()
        trail_layout = QVBoxLayout(trail_page)
        trail_layout.setContentsMargins(0, 8, 0, 0)
        trail_layout.setSpacing(8)
        trail_caption = QLabel(
            "Every decision ever recorded, newest first. Changed decisions keep "
            "their earlier entry: the trail is what an auditor reads.")
        trail_caption.setObjectName("Faint")
        trail_caption.setWordWrap(True)
        self.trail_table = DataTable(TRAIL_COLUMNS)
        export = QPushButton("Export the trail as CSV")
        export.clicked.connect(self.export_requested.emit)
        trail_layout.addWidget(trail_caption)
        trail_layout.addWidget(self.trail_table, stretch=1)
        trail_layout.addWidget(export)

        self.tabs.addTab(queue_page, "Queue")
        self.tabs.addTab(trail_page, "Decision trail")

        panel = Panel("Human review")
        caption = QLabel(
            "Reports a person must verify: rule, model or LLM disagreement, critical "
            "risk, thin evidence, or high energy with no rule match. The engine never "
            "closes one of these itself.")
        caption.setObjectName("Faint")
        caption.setWordWrap(True)
        panel.add(caption)
        panel.add(self.tabs, stretch=1)
        return panel

    def _build_bench(self) -> QWidget:
        """The case on the right: detail that scrolls, decisions that do not.

        The three decision buttons are the only controls on this page that must
        never move or be scrolled out of reach - a reviewer working a queue looks
        for them in the same place every time. So the case detail above them
        scrolls inside its own area, and the decision bar stays pinned to the
        bottom of the panel however short the window is.
        """
        panel = Panel("Report under review")

        self.trigger_pill = Pill("NO SELECTION", C.TEXT_DIM)
        self.verdict_pill = Pill("Engine: -", C.TEXT_DIM)
        self.risk_pill = Pill("Risk: -", C.TEXT_DIM)
        pills = QHBoxLayout()
        pills.setSpacing(8)
        pills.addWidget(self.trigger_pill)
        pills.addWidget(self.verdict_pill)
        pills.addStretch(1)
        pills.addWidget(self.risk_pill)

        self.reference = QLabel("-")
        self.reference.setObjectName("Muted")
        self.reason = QLabel("Select a report from the queue.")
        self.reason.setObjectName("Faint")
        self.reason.setWordWrap(True)

        self.narrative = QTextEdit()
        self.narrative.setReadOnly(True)
        self.narrative.setFixedHeight(92)

        self.fields = {
            "rule": FieldRow("", "IOGP rule", "-", C.WARN),
            "energy": FieldRow("", "Energy source", "-", C.DANGER),
            "barrier": FieldRow("", "Failed barrier", "-", C.DANGER),
            "activity": FieldRow("", "Activity", "-", C.BLUE),
            "location": FieldRow("", "Location", "-", C.ACCENT),
        }

        self.opinions = QLabel("-")
        self.opinions.setObjectName("Muted")
        self.opinions.setWordWrap(True)

        self.evidence = QTextEdit()
        self.evidence.setReadOnly(True)
        self.evidence.setPlaceholderText(
            "The cues and the decision path behind this report appear here.")

        case = QWidget()
        case_layout = QVBoxLayout(case)
        case_layout.setContentsMargins(0, 0, 0, 0)
        case_layout.setSpacing(8)
        case_layout.addLayout(pills)
        case_layout.addWidget(self.reference)
        case_layout.addWidget(self.reason)
        case_layout.addWidget(self.narrative)
        for row in self.fields.values():
            case_layout.addWidget(row)
        opinions_caption = QLabel("WHAT EACH ENGINE SAID")
        opinions_caption.setObjectName("Caption")
        case_layout.addWidget(opinions_caption)
        case_layout.addWidget(self.opinions)
        evidence_caption = QLabel("EVIDENCE AND DECISION PATH")
        evidence_caption.setObjectName("Caption")
        case_layout.addWidget(evidence_caption)
        case_layout.addWidget(self.evidence, stretch=1)

        self.case_scroll = scrollable(case, self.MIN_CASE_HEIGHT)
        panel.add(self.case_scroll, stretch=1)
        panel.add(self._build_decision_bar())
        return panel

    def _build_decision_bar(self) -> QWidget:
        bar = QWidget()
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(6)

        who = QHBoxLayout()
        who.setSpacing(8)
        who_label = QLabel("Reviewer")
        who_label.setObjectName("Caption")
        self.reviewer = QLineEdit()
        self.reviewer.setPlaceholderText("Your name - it goes on the record")
        self.reviewer.editingFinished.connect(
            lambda: self.reviewer_changed.emit(self.reviewer.text().strip()))
        who.addWidget(who_label)
        who.addWidget(self.reviewer, stretch=1)

        self.note = QLineEdit()
        self.note.setPlaceholderText(
            "Optional: why. A sentence here is what makes the decision reusable.")

        self.buttons: Dict[str, QPushButton] = {}
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        for text, decision, colour, key in DECISION_BUTTONS:
            button = QPushButton(text)
            button.setToolTip(f"Shortcut: {key}")
            button.setStyleSheet(
                f"QPushButton {{ color: {colour}; border: 1px solid {colour}; "
                f"border-radius: 6px; padding: 7px 10px; font-weight: 700; }}"
                f"QPushButton:hover {{ background: {C.PANEL_ALT}; }}"
                f"QPushButton:disabled {{ color: {C.TEXT_FAINT}; "
                f"border-color: {C.BORDER}; }}")
            button.clicked.connect(lambda _, value=decision: self._decide(value))
            self.buttons[decision] = button
            buttons.addWidget(button, stretch=1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.undo = QPushButton("Undo the last decision")
        self.undo.clicked.connect(self.undo_requested.emit)
        self.hint = QLabel("Keys 1, 2 and 3 decide; the bench advances by itself.")
        self.hint.setObjectName("Faint")
        footer.addWidget(self.hint, stretch=1)
        footer.addWidget(self.undo)

        layout.addLayout(who)
        layout.addWidget(self.note)
        layout.addLayout(buttons)
        layout.addLayout(footer)
        return bar

    def _bind_shortcuts(self) -> None:
        """1/2/3 decide - a queue is worked with the keyboard or it is not worked."""
        for _, decision, _, key in DECISION_BUTTONS:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda value=decision: self._decide(value))

    # -- controller interface ----------------------------------------------

    def set_reviewer(self, name: str) -> None:
        """Restore the remembered reviewer name."""
        self.reviewer.setText(name)

    def set_queue(self, rows: Sequence[Dict[str, object]], outstanding: int,
                  decided: int) -> None:
        """Render the queue and the progress line, keeping the selection."""
        previous = self._reference_at(self._current)
        self._rows = [dict(row) for row in rows]
        self.table.set_rows(self._rows)
        total = outstanding + decided
        if not total:
            self.progress.setText("Nothing analysed yet - run the analyser first.")
        elif not outstanding:
            self.progress.setText(
                f"Queue clear: all {decided} report(s) that needed a person have been "
                "decided. New reports join the queue as they are analysed.")
        else:
            self.progress.setText(
                f"{decided} of {total} decided  ·  {outstanding} still to review")
        self._restore(previous)

    def set_trail(self, rows: Sequence[Dict[str, object]]) -> None:
        """Render the audit trail tab."""
        self.trail_table.set_rows([self._trail_row(row) for row in rows])

    def set_case(self, result: Optional[Dict[str, object]],
                 decision: Optional[Dict[str, object]] = None) -> None:
        """Show one report, with any standing decision on it."""
        enabled = result is not None
        for button in self.buttons.values():
            button.setEnabled(enabled)
        if not enabled:
            self.trigger_pill.setText("NO SELECTION")
            self.trigger_pill.set_colour(C.TEXT_DIM)
            self.verdict_pill.setText("Engine: -")
            self.verdict_pill.set_colour(C.TEXT_DIM)
            self.risk_pill.setText("Risk: -")
            self.risk_pill.set_colour(C.TEXT_DIM)
            self.reference.setText("-")
            self.reason.setText("Select a report from the queue.")
            self.narrative.clear()
            self.evidence.clear()
            self.opinions.setText("-")
            for row in self.fields.values():
                row.set_value("-")
            return

        is_sif = bool(result.get("sif_potential"))
        band = str(result.get("risk_band", "Low"))
        self.trigger_pill.setText(str(result.get("review_trigger") or "QUEUED").upper())
        self.trigger_pill.set_colour(C.WARN)
        self.verdict_pill.setText(
            f"Engine: {'SIF potential' if is_sif else 'not SIF potential'}")
        self.verdict_pill.set_colour(C.DANGER if is_sif else C.OK)
        self.risk_pill.setText(f"Risk: {float(result.get('risk_score', 0.0)):.1f}")
        self.risk_pill.set_colour(BAND_COLORS.get(band, C.OK))
        self.reference.setText(str(result.get("reference") or "unreferenced report"))

        reason = str(result.get("review_reason") or "")
        if decision:
            self.reason.setText(
                f"Already decided: {decision.get('decision_label', '')} by "
                f"{decision.get('reviewer') or 'an unnamed reviewer'} on "
                f"{decision.get('decided_at', '')}. Deciding again supersedes it.\n{reason}")
        else:
            self.reason.setText(reason or "Queued for verification.")

        self.narrative.setPlainText(str(result.get("raw_text", "")))
        self.fields["rule"].set_value(str(result.get("iogp_rule", "-")))
        self.fields["energy"].set_value(str(result.get("energy_source", "-")))
        self.fields["barrier"].set_value(str(result.get("barrier_failure", "-")))
        self.fields["activity"].set_value(str(result.get("activity", "-")))
        self.fields["location"].set_value(str(result.get("location", "-")))
        self.opinions.setText(self._opinions(result))
        self.evidence.setHtml(self._evidence_html(result))
        self.note.setText(str(decision.get("note", "")) if decision else "")

    def advance(self) -> None:
        """Move to the next undecided report, or clear the bench when none is left.

        A decision rebuilds the queue, and the decided report usually leaves it -
        which slides the next report into the selected row on its own. Advancing
        again from there would skip it, so this returns when the current row is
        already undecided.
        """
        if 0 <= self._current < len(self._rows) and not self._rows[self._current].get("decided"):
            return
        for offset in range(1, len(self._rows) + 1):
            candidate = (self._current + offset) % len(self._rows) if self._rows else -1
            if candidate < 0:
                break
            if not self._rows[candidate].get("decided"):
                self.select(candidate)
                return
        self.select(-1)

    def select(self, row: int) -> None:
        """Select a queue row programmatically (``-1`` clears the selection)."""
        if 0 <= row < len(self._rows):
            self.table.selectRow(row)
        else:
            self.table.clearSelection()
            self._current = -1
            self.row_selected.emit(-1)

    def current_row(self) -> int:
        return self._current

    # -- internals ---------------------------------------------------------

    def _on_row(self, row: int) -> None:
        self._current = row
        self.row_selected.emit(row)

    def _decide(self, decision: str) -> None:
        """A decision was made on the selected report."""
        if self._current < 0 or self._current >= len(self._rows):
            return
        self.decision_made.emit(decision, self.note.text().strip())
        self.note.clear()

    def _reference_at(self, row: int) -> str:
        if 0 <= row < len(self._rows):
            return str(self._rows[row].get("reference", ""))
        return ""

    def _restore(self, reference: str) -> None:
        """Keep the reviewer on the same report across a queue rebuild."""
        if not self._rows:
            self._current = -1
            self.set_case(None)
            return
        for index, row in enumerate(self._rows):
            if reference and str(row.get("reference", "")) == reference:
                self.table.selectRow(index)
                return
        self.table.selectRow(0)

    @staticmethod
    def _trail_row(payload: Dict[str, object]) -> Dict[str, object]:
        """Turn a decision record into its table row."""
        from sif.review import DECISION_LABELS

        decision = str(payload.get("decision", ""))
        engine = bool(payload.get("engine_verdict"))
        outcome = ("overturned the engine" if payload.get("overturns_engine")
                   else "no label" if payload.get("label") is None else "agreed")
        stamp = str(payload.get("decided_at", ""))
        return {
            **payload,
            "decision_label": DECISION_LABELS.get(decision, decision),
            "engine_label": "SIF" if engine else "not SIF",
            "decided_short": stamp.replace("T", "  ")[5:16] if len(stamp) >= 16 else stamp,
            "outcome": outcome,
        }

    @staticmethod
    def _opinions(result: Dict[str, object]) -> str:
        """One line per engine, so the reviewer sees who disagreed with whom."""
        parts = [
            f"Rules: {'SIF' if result.get('lexical_flag') else 'not SIF'} "
            f"(confidence {float(result.get('rule_confidence', 0.0)):.2f})",
            f"Semantic: {'SIF' if result.get('semantic_flag') else 'not SIF'}"
            if result.get("semantic_active") else "Semantic: not active",
        ]
        if result.get("ml_active"):
            probability = float(result.get("ml_probability") or 0.0)
            parts.append(f"Model: P(SIF) {probability:.2f} "
                         f"({'SIF' if result.get('ml_flag') else 'not SIF'})")
        else:
            parts.append("Model: not trained")
        if result.get("llm_active"):
            parts.append(f"Local LLM: {'SIF' if result.get('llm_flag') else 'not SIF'}"
                         f" - {result.get('llm_rule', '')}")
        else:
            parts.append("Local LLM: not consulted")
        return "   ·   ".join(parts)

    @staticmethod
    def _evidence_html(result: Dict[str, object]) -> str:
        evidence = result.get("evidence", {}) or {}
        cues = "; ".join(evidence.get("lexical_cues", [])) or "none"
        risk = evidence.get("risk", {}) or {}
        llm = evidence.get("llm", {}) or {}
        html = [
            f"<b>{result.get('explanation', '')}</b>",
            f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Decision path</p>"
            f"{evidence.get('decision_path', '')}",
            f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Risk</p>"
            f"{risk.get('rationale', '')}",
            f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Lexical cues</p>{cues}",
        ]
        translated = str(result.get("translated_text", ""))
        if translated:
            html.append(f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>English "
                        f"rendering used for analysis</p>{translated}")
        if llm and llm.get("rationale"):
            html.append(f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Local LLM"
                        f"</p>{llm.get('rationale')}")
        return "".join(html)
