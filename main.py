"""PyQt6 desktop interface for the SIF Insight Console.

Oil India Limited - Problem Statement 26165
"Prediction and prevention of Serious Injury and Fatality (SIF) events from
UA/UC and near-miss reporting."

Layout
------
+----------------------------+------------------------------------------------+
| LEFT  - Control panel      | RIGHT - Analytics dashboard                    |
|  * free-text ingestion box |  * KPI header (processed, SIF count, % SIF)    |
|  * Process Text            |  * live results matrix (QTableWidget)          |
|  * Batch Import CSV        |                                                |
+----------------------------+------------------------------------------------+

Concurrency
-----------
Every parsing and file-reading operation runs inside :class:`AnalysisWorker`,
a ``QThread`` subclass.  Results are streamed back one row at a time through
``pyqtSignal`` connections, so the Qt event loop never blocks and the table
fills in progressively even for large CSV batches.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sif_engine import CSV_TEXT_COLUMNS, SEED_REPORTS, SIFEngine

__all__ = ["AnalysisWorker", "MetricCard", "DashboardHeader", "MainWindow"]

APP_NAME = "SIF Insight Console"
APP_SUBTITLE = "Oil India Limited | PS 26165 - UA/UC & Near-Miss Intelligence"

#: Dashboard table schema: (header label, assessment key, column width hint).
TABLE_COLUMNS: Sequence[tuple] = (
    ("#", "_index", 48),
    ("Timestamp", "_timestamp", 105),
    ("SIF", "sif_potential", 70),
    ("IOGP Rule", "iogp_rule", 190),
    ("Activity", "activity", 200),
    ("Location", "location", 190),
    ("Barrier Failure", "barrier_failure", 300),
    ("Energy Source", "energy_source", 220),
    ("Severity", "severity_hint", 90),
    ("Conf.", "confidence", 70),
    ("Source Narrative", "raw_text", 460),
)

# Palette -- a single place to retune the visual identity.
COLOR_BG = "#0f1720"
COLOR_PANEL = "#16212e"
COLOR_CARD = "#1d2a3a"
COLOR_ACCENT = "#00b8a9"
COLOR_TEXT = "#e6edf3"
COLOR_MUTED = "#8ba0b5"
COLOR_DANGER = "#ff5c5c"
COLOR_WARN = "#f4b942"
COLOR_OK = "#4cc38a"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif;
    font-size: 13px;
}}
QFrame#Panel {{
    background-color: {COLOR_PANEL};
    border: 1px solid #24354a;
    border-radius: 10px;
}}
QFrame#Card {{
    background-color: {COLOR_CARD};
    border: 1px solid #2b3d52;
    border-radius: 10px;
}}
QLabel#Title {{ font-size: 20px; font-weight: 700; color: {COLOR_TEXT}; }}
QLabel#Subtitle {{ font-size: 12px; color: {COLOR_MUTED}; }}
QLabel#SectionTitle {{ font-size: 14px; font-weight: 600; color: {COLOR_ACCENT}; }}
QLabel#CardValue {{ font-size: 26px; font-weight: 700; color: {COLOR_TEXT}; }}
QLabel#CardCaption {{ font-size: 11px; color: {COLOR_MUTED}; letter-spacing: 1px; }}
QTextEdit {{
    background-color: #101a25;
    border: 1px solid #2b3d52;
    border-radius: 8px;
    padding: 8px;
    selection-background-color: {COLOR_ACCENT};
}}
QPushButton {{
    background-color: {COLOR_CARD};
    border: 1px solid #33506b;
    border-radius: 8px;
    padding: 9px 14px;
    font-weight: 600;
}}
QPushButton:hover {{ background-color: #24374d; }}
QPushButton:pressed {{ background-color: #1a2836; }}
QPushButton:disabled {{ color: #5c6f82; border-color: #24354a; }}
QPushButton#Primary {{
    background-color: {COLOR_ACCENT};
    border: 1px solid {COLOR_ACCENT};
    color: #06231f;
}}
QPushButton#Primary:hover {{ background-color: #16cbbc; }}
QPushButton#Primary:disabled {{ background-color: #2c4a49; color: #7a9694; }}
QTableWidget {{
    background-color: #101a25;
    alternate-background-color: #13202d;
    gridline-color: #24354a;
    border: 1px solid #2b3d52;
    border-radius: 8px;
    selection-background-color: #1f4a55;
}}
QHeaderView::section {{
    background-color: {COLOR_CARD};
    color: {COLOR_MUTED};
    padding: 7px;
    border: none;
    border-right: 1px solid #24354a;
    font-weight: 600;
}}
QProgressBar {{
    background-color: #101a25;
    border: 1px solid #2b3d52;
    border-radius: 6px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{ background-color: {COLOR_ACCENT}; border-radius: 5px; }}
QStatusBar {{ color: {COLOR_MUTED}; border-top: 1px solid #24354a; }}
"""


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class AnalysisWorker(QThread):
    """Background thread that parses reports and streams rows to the GUI.

    The worker accepts either an in-memory list of narratives or a CSV path.
    File I/O and parsing both happen inside :meth:`run`, i.e. off the GUI
    thread, and each finished row is emitted immediately rather than being
    batched, so the dashboard updates in real time.

    Signals
    -------
    row_ready(dict)
        One completed assessment dictionary (see :class:`sif_engine.SIFAssessment`).
    progress(int, int)
        ``(completed, total)`` counters for the progress bar.
    failed(str)
        Human-readable error message; the run is aborted after this.
    completed(int)
        Number of rows successfully emitted once the run finishes.
    """

    row_ready = pyqtSignal(dict)
    progress = pyqtSignal(int, int)
    failed = pyqtSignal(str)
    completed = pyqtSignal(int)

    #: Delay between rows (ms) so batch streaming is visible during a demo.
    STREAM_DELAY_MS = 40

    def __init__(
        self,
        engine: SIFEngine,
        texts: Optional[Sequence[str]] = None,
        csv_path: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._texts: List[str] = list(texts or [])
        self._csv_path = csv_path

    # -- QThread entry point ----------------------------------------------

    def run(self) -> None:  # noqa: D102 - documented in the class docstring
        try:
            texts = self._texts
            if self._csv_path:
                texts = self._read_csv(self._csv_path)

            texts = [item.strip() for item in texts if isinstance(item, str) and item.strip()]
            total = len(texts)
            if total == 0:
                self.failed.emit("No usable report text was found in the input.")
                return

            emitted = 0
            for index, narrative in enumerate(texts, start=1):
                if self.isInterruptionRequested():
                    break
                result = self._engine.analyze(narrative)
                result["_timestamp"] = datetime.now().strftime("%H:%M:%S")
                self.row_ready.emit(result)
                emitted += 1
                self.progress.emit(index, total)
                if self.STREAM_DELAY_MS:
                    self.msleep(self.STREAM_DELAY_MS)

            self.completed.emit(emitted)
        except Exception as exc:  # pragma: no cover - defensive GUI guard
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _read_csv(path: str) -> List[str]:
        """Extract narratives from a CSV file.

        Recognises the common column names listed in
        :data:`sif_engine.CSV_TEXT_COLUMNS`.  If none is present, the longest
        text cell in each row is used, which keeps the importer usable with
        arbitrary contractor-supplied exports.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        narratives: List[str] = []
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel  # Fall back to a plain comma-separated read.

            reader = csv.DictReader(handle, dialect=dialect)
            if reader.fieldnames:
                lookup = {
                    (name or "").strip().lower(): name for name in reader.fieldnames
                }
                target = next((lookup[key] for key in CSV_TEXT_COLUMNS if key in lookup), None)
                for row in reader:
                    if target and row.get(target):
                        narratives.append(str(row[target]))
                    else:
                        values = [str(value) for value in row.values() if value]
                        if values:
                            narratives.append(max(values, key=len))
            else:  # Header-less file: treat every line as one narrative.
                handle.seek(0)
                narratives = [line.strip() for line in handle if line.strip()]

        return narratives


# ---------------------------------------------------------------------------
# Dashboard widgets
# ---------------------------------------------------------------------------


class MetricCard(QFrame):
    """Stylised KPI tile showing one headline metric."""

    def __init__(self, caption: str, value: str = "0", accent: str = COLOR_TEXT) -> None:
        super().__init__()
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)

        self._caption = QLabel(caption.upper())
        self._caption.setObjectName("CardCaption")

        self._value = QLabel(value)
        self._value.setObjectName("CardValue")
        self._value.setStyleSheet(f"color: {accent};")

        layout.addWidget(self._caption)
        layout.addWidget(self._value)

    def set_value(self, value: str) -> None:
        """Update the displayed metric value."""
        self._value.setText(value)


class DashboardHeader(QFrame):
    """Header strip holding the four headline SIF metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Panel")

        self.total_card = MetricCard("Total Processed", "0", COLOR_TEXT)
        self.sif_card = MetricCard("SIF-Potential Events", "0", COLOR_DANGER)
        self.rate_card = MetricCard("% SIF-Potential", "0.0%", COLOR_WARN)
        self.rule_card = MetricCard("Top IOGP Exposure", "-", COLOR_ACCENT)

        layout = QGridLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(self.total_card, 0, 0)
        layout.addWidget(self.sif_card, 0, 1)
        layout.addWidget(self.rate_card, 0, 2)
        layout.addWidget(self.rule_card, 0, 3)

    def update_metrics(self, total: int, sif_count: int, top_rule: str) -> None:
        """Refresh all KPI tiles from the current dashboard state."""
        rate = (sif_count / total * 100.0) if total else 0.0
        self.total_card.set_value(str(total))
        self.sif_card.set_value(str(sif_count))
        self.rate_card.set_value(f"{rate:.1f}%")
        self.rule_card.set_value(top_rule or "-")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Two-pane console: ingestion controls on the left, analytics on the right."""

    def __init__(self) -> None:
        super().__init__()
        self.engine = SIFEngine()
        self.worker: Optional[AnalysisWorker] = None
        self.rows: List[Dict[str, object]] = []

        self.setWindowTitle(APP_NAME)
        self.resize(1500, 860)
        self.setStyleSheet(STYLESHEET)

        self._build_menu()
        self._build_ui()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready - paste a report or load the seed incidents.")

    # -- construction ------------------------------------------------------

    def _build_menu(self) -> None:
        """Create the minimal application menu."""
        file_menu = self.menuBar().addMenu("&File")

        import_action = QAction("&Batch Import CSV...", self)
        import_action.setShortcut("Ctrl+O")
        import_action.triggered.connect(self.import_csv)
        file_menu.addAction(import_action)

        export_action = QAction("&Export Results CSV...", self)
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self.export_csv)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_ui(self) -> None:
        """Assemble the splitter, control panel and analytics dashboard."""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_dashboard())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 1070])

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.addWidget(splitter)
        self.setCentralWidget(container)

    def _build_control_panel(self) -> QWidget:
        """Left pane: title, text ingestion box and action buttons."""
        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setMinimumWidth(380)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel(APP_NAME)
        title.setObjectName("Title")
        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("Subtitle")
        subtitle.setWordWrap(True)

        section = QLabel("REPORT INGESTION")
        section.setObjectName("SectionTitle")

        hint = QLabel(
            "Paste one UA/UC or near-miss narrative per blank-line-separated "
            "block. The engine extracts the IOGP rule, activity, location and "
            "failed barrier, then flags SIF potential."
        )
        hint.setObjectName("Subtitle")
        hint.setWordWrap(True)

        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText(
            "e.g. Near miss at GGS-4: worker on an incomplete scaffold at 6 m "
            "with his lanyard not anchored while replacing a light fitting..."
        )
        self.input_box.setMinimumHeight(220)

        self.process_button = QPushButton("Process Text")
        self.process_button.setObjectName("Primary")
        self.process_button.clicked.connect(self.process_text)

        self.import_button = QPushButton("Batch Import CSV")
        self.import_button.clicked.connect(self.import_csv)

        self.seed_button = QPushButton("Load 5 Seed Incidents")
        self.seed_button.clicked.connect(self.load_seed_data)

        self.clear_button = QPushButton("Clear Dashboard")
        self.clear_button.clicked.connect(self.clear_dashboard)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(8)
        layout.addWidget(section)
        layout.addWidget(hint)
        layout.addWidget(self.input_box, stretch=1)
        layout.addWidget(self.process_button)
        layout.addWidget(self.import_button)
        layout.addWidget(self.seed_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.progress_bar)
        return panel

    def _build_dashboard(self) -> QWidget:
        """Right pane: KPI header above the streaming results matrix."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.header = DashboardHeader()

        matrix_frame = QFrame()
        matrix_frame.setObjectName("Panel")
        matrix_layout = QVBoxLayout(matrix_frame)
        matrix_layout.setContentsMargins(14, 14, 14, 14)
        matrix_layout.setSpacing(8)

        matrix_title = QLabel("PARSED INCIDENT MATRIX")
        matrix_title.setObjectName("SectionTitle")

        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([label for label, _, _ in TABLE_COLUMNS])
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        for column, (_, _, width) in enumerate(TABLE_COLUMNS):
            self.table.setColumnWidth(column, width)

        matrix_layout.addWidget(matrix_title)
        matrix_layout.addWidget(self.table)

        layout.addWidget(self.header)
        layout.addWidget(matrix_frame, stretch=1)
        return panel

    # -- actions -----------------------------------------------------------

    def process_text(self) -> None:
        """Parse the narratives currently in the ingestion box."""
        raw = self.input_box.toPlainText().strip()
        if not raw:
            QMessageBox.information(
                self, APP_NAME, "Paste at least one report narrative before processing."
            )
            return

        # Blank lines separate incidents; a single block is one incident.
        blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
        self._start_worker(AnalysisWorker(self.engine, texts=blocks, parent=self))

    def load_seed_data(self) -> None:
        """Populate the ingestion box with the five seed incidents and run them."""
        self.input_box.setPlainText("\n\n".join(SEED_REPORTS))
        self._start_worker(AnalysisWorker(self.engine, texts=list(SEED_REPORTS), parent=self))

    def import_csv(self) -> None:
        """Choose a CSV file and parse it on the worker thread."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Batch Import UA/UC Reports", os.getcwd(), "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        self._start_worker(AnalysisWorker(self.engine, csv_path=path, parent=self))

    def export_csv(self) -> None:
        """Write the current dashboard matrix to a CSV file."""
        if not self.rows:
            QMessageBox.information(self, APP_NAME, "There is nothing to export yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Parsed Results", os.path.join(os.getcwd(), "sif_results.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        headers = [label for label, _, _ in TABLE_COLUMNS]
        try:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                for index, row in enumerate(self.rows, start=1):
                    writer.writerow(
                        [self._cell_text(row, key, index) for _, key, _ in TABLE_COLUMNS]
                    )
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not write the file:\n{exc}")
            return
        self.statusBar().showMessage(f"Exported {len(self.rows)} rows to {path}", 8000)

    def clear_dashboard(self) -> None:
        """Reset the table, counters and KPI header."""
        self.rows.clear()
        self.table.setRowCount(0)
        self.header.update_metrics(0, 0, "-")
        self.statusBar().showMessage("Dashboard cleared.", 4000)

    # -- worker plumbing ---------------------------------------------------

    def _start_worker(self, worker: AnalysisWorker) -> None:
        """Wire a worker's signals to the GUI and start it."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self, APP_NAME, "An analysis run is already in progress. Please wait."
            )
            return

        self.worker = worker
        worker.row_ready.connect(self.on_row_ready)
        worker.progress.connect(self.on_progress)
        worker.failed.connect(self.on_failed)
        worker.completed.connect(self.on_completed)
        worker.finished.connect(self._release_worker)

        self._set_controls_enabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage("Analysing reports on the worker thread...")
        worker.start()

    def on_row_ready(self, result: Dict[str, object]) -> None:
        """Slot: append one streamed assessment to the matrix (GUI thread)."""
        self.rows.append(result)
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)

        for column, (_, key, _) in enumerate(TABLE_COLUMNS):
            item = QTableWidgetItem(self._cell_text(result, key, row_index + 1))
            item.setToolTip(str(result.get("raw_text", "")))
            if key == "sif_potential":
                is_sif = bool(result.get("sif_potential"))
                item.setForeground(QColor(COLOR_DANGER if is_sif else COLOR_OK))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                font = QFont()
                font.setBold(True)
                item.setFont(font)
            elif key == "severity_hint":
                item.setForeground(QColor(self._severity_color(str(result.get(key, "")))))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            elif key in {"_index", "_timestamp", "confidence"}:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_index, column, item)

        self.table.scrollToBottom()
        self._refresh_metrics()

    def on_progress(self, done: int, total: int) -> None:
        """Slot: advance the progress bar."""
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(done)
        self.statusBar().showMessage(f"Processing report {done} of {total}...")

    def on_failed(self, message: str) -> None:
        """Slot: surface a worker-side error without killing the session."""
        self.statusBar().showMessage("Analysis failed.", 6000)
        QMessageBox.warning(self, APP_NAME, message)

    def on_completed(self, count: int) -> None:
        """Slot: final bookkeeping once a run finishes."""
        self.statusBar().showMessage(
            f"Completed - {count} report(s) analysed, {self._sif_count()} flagged SIF-potential.",
            10000,
        )

    def _release_worker(self) -> None:
        """Slot: re-enable controls when the thread's event loop exits."""
        self.progress_bar.setVisible(False)
        self._set_controls_enabled(True)
        self.worker = None

    # -- small helpers -----------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        for button in (
            self.process_button,
            self.import_button,
            self.seed_button,
            self.clear_button,
        ):
            button.setEnabled(enabled)

    def _sif_count(self) -> int:
        return sum(1 for row in self.rows if row.get("sif_potential"))

    def _refresh_metrics(self) -> None:
        """Recompute the KPI header from the accumulated rows."""
        total = len(self.rows)
        sif_count = self._sif_count()

        tally: Dict[str, int] = {}
        for row in self.rows:
            if row.get("sif_potential"):
                rule = str(row.get("iogp_rule", ""))
                tally[rule] = tally.get(rule, 0) + 1
        top_rule = max(tally, key=tally.get) if tally else "-"
        self.header.update_metrics(total, sif_count, top_rule)

    @staticmethod
    def _severity_color(severity: str) -> str:
        return {"High": COLOR_DANGER, "Medium": COLOR_WARN}.get(severity, COLOR_OK)

    @staticmethod
    def _cell_text(result: Dict[str, object], key: str, index: int) -> str:
        """Render one assessment field as display text."""
        if key == "_index":
            return str(index)
        if key == "_timestamp":
            return str(result.get("_timestamp", ""))
        if key == "sif_potential":
            return "YES" if result.get("sif_potential") else "no"
        if key == "confidence":
            return f"{float(result.get('confidence', 0.0)):.2f}"
        if key == "raw_text":
            text = str(result.get("raw_text", ""))
            return text if len(text) <= 300 else text[:297] + "..."
        return str(result.get(key, ""))

    # -- Qt lifecycle ------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop any running worker cleanly before the window closes."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(2000)
        super().closeEvent(event)


def create_application(argv: Optional[List[str]] = None) -> QApplication:
    """Build a configured :class:`QApplication` instance."""
    app = QApplication(argv if argv is not None else [])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Oil India Limited")
    return app
