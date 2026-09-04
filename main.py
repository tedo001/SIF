"""PyQt6 desktop interface for the SIF Insight Console.

Oil India Limited - Problem Statement 26165.

The window is the last stage of the pipeline in :mod:`sif.pipeline`: it drives
the analysis from a worker thread and renders what comes back as HSE
intelligence.

Layout
------
+----------------------------+------------------------------------------------+
| LEFT  - Control panel      | RIGHT - Analytics dashboard                    |
|  * free-text ingestion box |  * KPI header (processed, SIF, % SIF, risk)    |
|  * encoder selector        |  * Incident matrix / Risk hotspots / Review    |
|  * Process Text            |  * evidence pane for the selected report       |
|  * Batch Import CSV        |                                                |
+----------------------------+------------------------------------------------+

Concurrency
-----------
Model loading, CSV reading and every pipeline stage run inside
:class:`AnalysisWorker`, a ``QThread``. Results are streamed back one row at a
time through ``pyqtSignal`` connections, so the Qt event loop never blocks -
including during the first-run model download, which reports progress through
the status bar instead of freezing the window.
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
    QComboBox,
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
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sif import SEED_REPORTS, SIFPipeline
from sif.lexical import CSV_TEXT_COLUMNS

__all__ = ["AnalysisWorker", "MetricCard", "DashboardHeader", "MainWindow"]

APP_NAME = "SIF Insight Console"
APP_SUBTITLE = "Oil India Limited | PS 26165 - UA/UC & Near-Miss Intelligence"

#: Incident matrix schema: (header label, result key, column width hint).
TABLE_COLUMNS: Sequence[tuple] = (
    ("#", "_index", 42),
    ("Ref", "reference", 74),
    ("SIF", "sif_potential", 54),
    ("P(SIF)", "p_sif", 56),
    ("Risk", "risk_score", 52),
    ("Band", "risk_band", 66),
    ("IOGP Rule", "iogp_rule", 168),
    ("Activity", "activity", 158),
    ("Location", "location", 158),
    ("Barrier Failure", "barrier_failure", 250),
    ("Energy Source", "energy_source", 180),
    ("Review", "review_trigger", 104),
    ("Source Narrative", "raw_text", 360),
)

HOTSPOT_COLUMNS: Sequence[tuple] = (
    ("Type", "kind", 150),
    ("Cluster", "label", 320),
    ("Reports", "reports", 76),
    ("SIF", "sif_reports", 60),
    ("SIF %", "sif_rate", 66),
    ("Mean risk", "mean_risk", 84),
    ("Peak risk", "max_risk", 84),
    ("Dominant rule", "top_rule", 190),
    ("Dominant barrier", "top_barrier", 260),
)

REVIEW_COLUMNS: Sequence[tuple] = (
    ("Trigger", "trigger", 150),
    ("Ref", "reference", 74),
    ("Risk", "risk_score", 60),
    ("SIF", "sif_potential", 54),
    ("Why a human is needed", "reason", 380),
    ("Report", "summary", 460),
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

#: Risk band -> colour, shared by the matrix and the review queue.
BAND_COLORS = {"Critical": COLOR_DANGER, "High": "#ff9152",
               "Medium": COLOR_WARN, "Low": COLOR_OK}

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
QComboBox {{
    background-color: {COLOR_CARD};
    border: 1px solid #33506b;
    border-radius: 8px;
    padding: 6px 10px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_CARD};
    selection-background-color: #1f4a55;
    border: 1px solid #33506b;
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
QTabWidget::pane {{ border: 1px solid #24354a; border-radius: 8px; top: -1px; }}
QTabBar::tab {{
    background: {COLOR_CARD};
    color: {COLOR_MUTED};
    padding: 7px 16px;
    border: 1px solid #24354a;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 3px;
}}
QTabBar::tab:selected {{ color: {COLOR_TEXT}; background: {COLOR_PANEL}; }}
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
    """Background thread that runs the pipeline and streams rows to the GUI.

    The worker accepts either an in-memory list of narratives or a CSV path.
    Encoder loading, file I/O and every pipeline stage happen inside :meth:`run`,
    i.e. off the GUI thread, and each finished row is emitted immediately rather
    than being batched, so the dashboard updates in real time.

    Signals
    -------
    row_ready(dict)
        One completed :class:`~sif.pipeline.PipelineResult` as a dictionary.
    progress(int, int)
        ``(completed, total)`` counters for the progress bar.
    status(str)
        Human-readable stage message ("loading encoder", "encoder ready: ...").
    failed(str)
        Error message; the run is aborted after this.
    completed(int)
        Number of rows successfully emitted once the run finishes.
    """

    row_ready = pyqtSignal(dict)
    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    failed = pyqtSignal(str)
    completed = pyqtSignal(int)

    #: Delay between rows (ms) so batch streaming is visible during a demo.
    STREAM_DELAY_MS = 30

    def __init__(
        self,
        pipeline: SIFPipeline,
        texts: Optional[Sequence[str]] = None,
        csv_path: Optional[str] = None,
        references: Optional[Sequence[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._texts: List[str] = list(texts or [])
        self._references: List[str] = list(references or [])
        self._csv_path = csv_path

    # -- QThread entry point ----------------------------------------------

    def run(self) -> None:  # noqa: D102 - documented in the class docstring
        try:
            texts, references = self._texts, self._references
            if self._csv_path:
                self.status.emit("Reading CSV...")
                texts, references = self._read_csv(self._csv_path)

            pairs = [(text.strip(), references[index] if index < len(references) else "")
                     for index, text in enumerate(texts)
                     if isinstance(text, str) and text.strip()]
            if not pairs:
                self.failed.emit("No usable report text was found in the input.")
                return

            self.status.emit("Loading semantic encoder (first run may download the model)...")
            self.status.emit(f"Encoder ready - {self._pipeline.warm_up()}")

            total = len(pairs)
            emitted = 0
            for index, (narrative, reference) in enumerate(pairs, start=1):
                if self.isInterruptionRequested():
                    break
                result = self._pipeline.analyze(narrative, reference)
                payload = result.to_dict()
                payload["_timestamp"] = datetime.now().strftime("%H:%M:%S")
                self.row_ready.emit(payload)
                emitted += 1
                self.progress.emit(index, total)
                if self.STREAM_DELAY_MS:
                    self.msleep(self.STREAM_DELAY_MS)

            self.completed.emit(emitted)
        except Exception as exc:  # pragma: no cover - defensive GUI guard
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _read_csv(path: str):
        """Extract narratives (and references, when present) from a CSV file.

        Recognises the common column names listed in
        :data:`sif.lexical.CSV_TEXT_COLUMNS`. If none is present, the longest
        text cell in each row is used, which keeps the importer usable with
        arbitrary contractor-supplied exports.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        narratives: List[str] = []
        references: List[str] = []
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel  # Fall back to a plain comma-separated read.

            reader = csv.DictReader(handle, dialect=dialect)
            if reader.fieldnames:
                lookup = {(name or "").strip().lower(): name for name in reader.fieldnames}
                target = next((lookup[key] for key in CSV_TEXT_COLUMNS if key in lookup), None)
                ref_key = next((lookup[key] for key in ("report_id", "id", "ref", "reference")
                                if key in lookup), None)
                for row in reader:
                    if target and row.get(target):
                        narratives.append(str(row[target]))
                    else:
                        values = [str(value) for value in row.values() if value]
                        if not values:
                            continue
                        narratives.append(max(values, key=len))
                    references.append(str(row.get(ref_key, "")) if ref_key else "")
            else:  # Header-less file: treat every line as one narrative.
                handle.seek(0)
                narratives = [line.strip() for line in handle if line.strip()]
                references = [""] * len(narratives)

        return narratives, references


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
    """Header strip holding the headline SIF metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Panel")

        self.total_card = MetricCard("Total Processed", "0", COLOR_TEXT)
        self.sif_card = MetricCard("SIF-Potential Events", "0", COLOR_DANGER)
        self.rate_card = MetricCard("% SIF-Potential", "0.0%", COLOR_WARN)
        self.risk_card = MetricCard("Mean Risk Score", "0.0", COLOR_ACCENT)
        self.review_card = MetricCard("Awaiting Human Review", "0", COLOR_TEXT)

        layout = QGridLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        for column, card in enumerate((self.total_card, self.sif_card, self.rate_card,
                                       self.risk_card, self.review_card)):
            layout.addWidget(card, 0, column)

    def update_metrics(self, kpis: Dict[str, object]) -> None:
        """Refresh all KPI tiles from the pipeline's aggregate."""
        self.total_card.set_value(str(kpis.get("total", 0)))
        self.sif_card.set_value(str(kpis.get("sif_potential", 0)))
        self.rate_card.set_value(f'{float(kpis.get("sif_rate", 0.0)):.1f}%')
        self.risk_card.set_value(f'{float(kpis.get("mean_risk", 0.0)):.1f}')
        self.review_card.set_value(str(kpis.get("needs_review", 0)))


def _make_table(columns: Sequence[tuple]) -> QTableWidget:
    """Build a read-only, row-selecting table with the given column schema."""
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels([label for label, _, _ in columns])
    table.setAlternatingRowColors(True)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    table.horizontalHeader().setStretchLastSection(True)
    for column, (_, _, width) in enumerate(columns):
        table.setColumnWidth(column, width)
    return table


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Two-pane console: ingestion controls on the left, analytics on the right."""

    #: Hotspots and the review queue are recomputed every N streamed rows.
    PANEL_REFRESH_EVERY = 5

    def __init__(self) -> None:
        super().__init__()
        self.pipeline = SIFPipeline()
        self.worker: Optional[AnalysisWorker] = None
        self.rows: List[Dict[str, object]] = []

        self.setWindowTitle(APP_NAME)
        self.resize(1560, 900)
        self.setStyleSheet(STYLESHEET)

        self._build_menu()
        self._build_ui()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Ready - paste a report or load the seed incidents. The encoder loads on "
            "the first run."
        )

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
        splitter.setSizes([430, 1130])

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.addWidget(splitter)
        self.setCentralWidget(container)

    def _build_control_panel(self) -> QWidget:
        """Left pane: title, encoder selector, ingestion box and action buttons."""
        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setMinimumWidth(390)

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
            "Paste one UA/UC or near-miss narrative per blank-line-separated block. "
            "The pipeline classifies SIF potential and the IOGP rule, extracts the "
            "activity, location and failed barrier, then scores and ranks the risk."
        )
        hint.setObjectName("Subtitle")
        hint.setWordWrap(True)

        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText(
            "e.g. Near miss at GGS-4: worker on an incomplete scaffold at 6 m with his "
            "lanyard not anchored while replacing a light fitting..."
        )
        self.input_box.setMinimumHeight(190)

        encoder_label = QLabel("SEMANTIC ENCODER")
        encoder_label.setObjectName("CardCaption")
        self.encoder_box = QComboBox()
        self.encoder_box.addItem("Auto - transformer, fall back offline", "auto")
        self.encoder_box.addItem("Transformer (all-MiniLM-L6-v2)", "transformer")
        self.encoder_box.addItem("Offline - lexical rules only", "hashing")
        self.encoder_box.currentIndexChanged.connect(self.on_encoder_changed)

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
        layout.addWidget(encoder_label)
        layout.addWidget(self.encoder_box)
        layout.addWidget(self.process_button)
        layout.addWidget(self.import_button)
        layout.addWidget(self.seed_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.progress_bar)
        return panel

    def _build_dashboard(self) -> QWidget:
        """Right pane: KPI header, the three analysis tabs and the evidence pane."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.header = DashboardHeader()

        self.table = _make_table(TABLE_COLUMNS)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        self.hotspot_table = _make_table(HOTSPOT_COLUMNS)
        self.review_table = _make_table(REVIEW_COLUMNS)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.table, "Parsed Incident Matrix")
        self.tabs.addTab(self.hotspot_table, "Risk Hotspots")
        self.tabs.addTab(self.review_table, "Human Review Queue")

        self.evidence_box = QTextEdit()
        self.evidence_box.setReadOnly(True)
        self.evidence_box.setFixedHeight(112)
        self.evidence_box.setPlaceholderText(
            "Select a row to see why the pipeline reached its verdict."
        )

        matrix_frame = QFrame()
        matrix_frame.setObjectName("Panel")
        matrix_layout = QVBoxLayout(matrix_frame)
        matrix_layout.setContentsMargins(14, 14, 14, 14)
        matrix_layout.setSpacing(8)

        evidence_title = QLabel("EVIDENCE FOR THE SELECTED REPORT")
        evidence_title.setObjectName("SectionTitle")

        matrix_layout.addWidget(self.tabs, stretch=1)
        matrix_layout.addWidget(evidence_title)
        matrix_layout.addWidget(self.evidence_box)

        layout.addWidget(self.header)
        layout.addWidget(matrix_frame, stretch=1)
        return panel

    # -- actions -----------------------------------------------------------

    def on_encoder_changed(self) -> None:
        """Rebuild the pipeline when the operator picks a different encoder."""
        backend = self.encoder_box.currentData()
        self.pipeline = SIFPipeline(backend=backend)
        self.statusBar().showMessage(
            f"Encoder set to '{backend}' - it loads on the next run.", 6000)

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
        self._start_worker(AnalysisWorker(self.pipeline, texts=blocks, parent=self))

    def load_seed_data(self) -> None:
        """Populate the ingestion box with the five seed incidents and run them."""
        self.input_box.setPlainText("\n\n".join(SEED_REPORTS))
        references = [f"SEED-{number:02d}" for number in range(1, len(SEED_REPORTS) + 1)]
        self._start_worker(AnalysisWorker(self.pipeline, texts=list(SEED_REPORTS),
                                          references=references, parent=self))

    def import_csv(self) -> None:
        """Choose a CSV file and run the pipeline over it on the worker thread."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Batch Import UA/UC Reports", os.getcwd(), "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        self._start_worker(AnalysisWorker(self.pipeline, csv_path=path, parent=self))

    def export_csv(self) -> None:
        """Write the current incident matrix to a CSV file."""
        if not self.rows:
            QMessageBox.information(self, APP_NAME, "There is nothing to export yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Parsed Results", os.path.join(os.getcwd(), "sif_results.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        headers = [label for label, _, _ in TABLE_COLUMNS] + ["Explanation"]
        try:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                for index, row in enumerate(self.rows, start=1):
                    writer.writerow(
                        [self._cell_text(row, key, index) for _, key, _ in TABLE_COLUMNS]
                        + [row.get("explanation", "")]
                    )
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not write the file:\n{exc}")
            return
        self.statusBar().showMessage(f"Exported {len(self.rows)} rows to {path}", 8000)

    def clear_dashboard(self) -> None:
        """Reset every table, counter and panel."""
        self.rows.clear()
        for table in (self.table, self.hotspot_table, self.review_table):
            table.setRowCount(0)
        self.header.update_metrics({})
        self.evidence_box.clear()
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
        worker.status.connect(self.on_status)
        worker.failed.connect(self.on_failed)
        worker.completed.connect(self.on_completed)
        worker.finished.connect(self._release_worker)

        self._set_controls_enabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        worker.start()

    def on_row_ready(self, result: Dict[str, object]) -> None:
        """Slot: append one streamed result to the matrix (GUI thread)."""
        self.rows.append(result)
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)

        for column, (_, key, _) in enumerate(TABLE_COLUMNS):
            item = QTableWidgetItem(self._cell_text(result, key, row_index + 1))
            item.setToolTip(str(result.get("explanation", "")))
            if key == "sif_potential":
                is_sif = bool(result.get("sif_potential"))
                item.setForeground(QColor(COLOR_DANGER if is_sif else COLOR_OK))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                font = QFont()
                font.setBold(True)
                item.setFont(font)
            elif key in {"risk_band", "risk_score"}:
                band = str(result.get("risk_band", "Low"))
                item.setForeground(QColor(BAND_COLORS.get(band, COLOR_OK)))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            elif key == "review_trigger":
                tone = COLOR_WARN if result.get("needs_review") else COLOR_MUTED
                item.setForeground(QColor(tone))
            elif key in {"_index", "reference", "p_sif"}:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_index, column, item)

        self.table.scrollToBottom()
        if len(self.rows) % self.PANEL_REFRESH_EVERY == 0:
            self._refresh_panels()
        else:
            self.header.update_metrics(self._aggregate().kpis)

    def on_progress(self, done: int, total: int) -> None:
        """Slot: advance the progress bar."""
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(done)
        self.statusBar().showMessage(f"Processing report {done} of {total}...")

    def on_status(self, message: str) -> None:
        """Slot: surface a stage message from the worker."""
        self.statusBar().showMessage(message)

    def on_failed(self, message: str) -> None:
        """Slot: surface a worker-side error without killing the session."""
        self.statusBar().showMessage("Analysis failed.", 6000)
        QMessageBox.warning(self, APP_NAME, message)

    def on_completed(self, count: int) -> None:
        """Slot: final aggregation once a run finishes."""
        intelligence = self._refresh_panels()
        encoder = str(intelligence.kpis.get("encoder", ""))
        self.statusBar().showMessage(
            f"Completed - {count} report(s) this run  |  dashboard totals: "
            f"{intelligence.kpis.get('total', 0)} processed, "
            f"{intelligence.kpis.get('sif_potential', 0)} SIF-potential, "
            f"{intelligence.kpis.get('needs_review', 0)} awaiting review  |  {encoder}",
            15000,
        )

    def on_row_selected(self) -> None:
        """Slot: render the evidence trail for the selected incident."""
        rows = {index.row() for index in self.table.selectedIndexes()}
        if not rows:
            return
        position = min(rows)
        if position >= len(self.rows):
            return
        result = self.rows[position]
        evidence = result.get("evidence", {}) or {}
        cues = "; ".join(evidence.get("lexical_cues", [])) or "none"
        semantic = ", ".join(
            f"{field} -> {label} ({score:.2f})"
            for field, (label, score) in (evidence.get("semantic_matches", {}) or {}).items()
        ) or "none"
        risk = evidence.get("risk", {}) or {}
        self.evidence_box.setHtml(
            f"<b>{result.get('explanation', '')}</b><br/>"
            f"<span style='color:{COLOR_MUTED}'>Decision path:</span> "
            f"{evidence.get('decision_path', '')}<br/>"
            f"<span style='color:{COLOR_MUTED}'>Risk:</span> "
            f"{result.get('risk_score', 0)}/100 ({result.get('risk_band', '')}) = "
            f"{risk.get('rationale', '')}<br/>"
            f"<span style='color:{COLOR_MUTED}'>Lexical cues:</span> {cues}<br/>"
            f"<span style='color:{COLOR_MUTED}'>Nearest semantic prototypes:</span> {semantic}"
        )

    def _release_worker(self) -> None:
        """Slot: re-enable controls when the thread's event loop exits."""
        self.progress_bar.setVisible(False)
        self._set_controls_enabled(True)
        self.worker = None

    # -- panels ------------------------------------------------------------

    def _aggregate(self):
        """Re-run corpus-level aggregation over the rows collected so far."""
        from sif.pipeline import PipelineResult

        results = []
        for row in self.rows:
            payload = {key: value for key, value in row.items()
                       if key in PipelineResult.__dataclass_fields__}
            results.append(PipelineResult(**payload))
        return self.pipeline.aggregate(results)

    def _refresh_panels(self):
        """Recompute the KPI header, hotspot table and review queue."""
        intelligence = self._aggregate()
        self.header.update_metrics(intelligence.kpis)
        self._fill_table(self.hotspot_table, HOTSPOT_COLUMNS,
                         [spot.to_dict() for spot in intelligence.hotspots])
        self._fill_table(self.review_table, REVIEW_COLUMNS,
                         [item.to_dict() for item in intelligence.review_queue])
        self.tabs.setTabText(1, f"Risk Hotspots ({len(intelligence.hotspots)})")
        self.tabs.setTabText(2, f"Human Review Queue ({len(intelligence.review_queue)})")
        return intelligence

    def _fill_table(self, table: QTableWidget, columns: Sequence[tuple],
                    payloads: Sequence[Dict[str, object]]) -> None:
        """Replace a table's contents with ``payloads``."""
        table.setRowCount(0)
        for payload in payloads:
            row_index = table.rowCount()
            table.insertRow(row_index)
            for column, (_, key, _) in enumerate(columns):
                value = payload.get(key, "")
                if isinstance(value, bool):
                    text = "YES" if value else "no"
                elif isinstance(value, float):
                    text = f"{value:.1f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                if key in {"reports", "sif_reports", "sif_rate", "mean_risk", "max_risk",
                           "risk_score", "sif_potential", "reference"}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if key in {"mean_risk", "max_risk", "risk_score"}:
                    numeric = float(value or 0.0)
                    band = ("Critical" if numeric >= 70 else "High" if numeric >= 50
                            else "Medium" if numeric >= 30 else "Low")
                    item.setForeground(QColor(BAND_COLORS[band]))
                if key == "trigger":
                    item.setForeground(QColor(COLOR_WARN))
                table.setItem(row_index, column, item)

    # -- small helpers -----------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.process_button, self.import_button, self.seed_button,
                       self.clear_button, self.encoder_box):
            widget.setEnabled(enabled)

    @staticmethod
    def _cell_text(result: Dict[str, object], key: str, index: int) -> str:
        """Render one result field as display text."""
        if key == "_index":
            return str(index)
        if key == "sif_potential":
            return "YES" if result.get("sif_potential") else "no"
        if key in {"p_sif", "confidence"}:
            return f"{float(result.get(key, 0.0)):.2f}"
        if key == "risk_score":
            return f"{float(result.get(key, 0.0)):.0f}"
        if key == "review_trigger":
            return str(result.get(key, "") or "-")
        if key == "reference":
            return str(result.get(key, "") or "-")
        if key == "raw_text":
            text = str(result.get("raw_text", ""))
            return text if len(text) <= 300 else text[:297] + "..."
        return str(result.get(key, ""))

    # -- Qt lifecycle ------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop any running worker cleanly before the window closes."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(3000)
        super().closeEvent(event)


def create_application(argv: Optional[List[str]] = None) -> QApplication:
    """Build a configured :class:`QApplication` instance."""
    app = QApplication(argv if argv is not None else [])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Oil India Limited")
    return app
