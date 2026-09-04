"""PyQt6 desktop application for the SIF Insight Console.

Oil India Limited - Problem Statement 26165.

This module is the controller: it owns the pipeline, the MLOps service and the
document extractor, runs all of them on worker threads, and pushes results into
the passive views in :mod:`ui.views`.

Threads
-------
``AnalysisWorker``
    Loads the encoder and runs the pipeline over a batch, streaming one row at a
    time back to the GUI.
``ExtractionWorker``
    Reads PDFs, scans and images through :mod:`sif.ocr` (including PaddleOCR
    model download on first use).
``TrainingWorker``
    Trains the XGBoost model and logs the run to MLflow.

None of the three touches a widget: they emit signals, and the slots that render
them run on the GUI thread.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from sif import SEED_REPORTS, SIFPipeline
from sif.lexical import CSV_TEXT_COLUMNS
from sif.logging_setup import (LOG_LEVELS, active_log_file, configure_logging,
                               log_file_path, set_level)
from sif.mlops import MLOpsService
from sif.ocr import DocumentExtractor
from sif.pipeline import PipelineResult
from ui.components import HeaderBar, Sidebar
from ui.theme import STYLESHEET
from ui.views import (
    HOTSPOT_COLUMNS,
    MATRIX_COLUMNS,
    REVIEW_COLUMNS,
    AnalyticsView,
    BatchUploadView,
    DashboardView,
    SettingsView,
    TableView,
)

__all__ = ["AnalysisWorker", "ExtractionWorker", "TrainingWorker", "OCRProbeWorker",
           "MainWindow", "create_application"]

LOGGER = logging.getLogger("sif.app")

APP_NAME = "SIF Insight Console"
APP_SUBTITLE = "UA/UC & Near-Miss Intelligence   |   PS 26165"

NAV_ITEMS = (
    ("dashboard", "▤", "Dashboard"),
    ("analysis", "✎", "Report Analysis"),
    ("batch", "⬆", "Batch Upload"),
    ("matrix", "▦", "Incident Matrix"),
    ("hotspots", "⌖", "Risk Hotspots"),
    ("review", "👤", "Human Review"),
    ("analytics", "📈", "Analytics"),
    ("settings", "⚙", "Settings"),
)

DOCUMENT_FILTER = ("Documents (*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.txt *.md);;"
                   "All files (*)")


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


class AnalysisWorker(QThread):
    """Runs the pipeline over a batch, streaming results back one row at a time."""

    row_ready = pyqtSignal(dict)
    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    failed = pyqtSignal(str)
    completed = pyqtSignal(int)

    STREAM_DELAY_MS = 25

    def __init__(self, pipeline: SIFPipeline, texts: Optional[Sequence[str]] = None,
                 csv_path: Optional[str] = None, references: Optional[Sequence[str]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._texts = list(texts or [])
        self._references = list(references or [])
        self._csv_path = csv_path

    def run(self) -> None:  # noqa: D102 - documented on the class
        try:
            texts, references = self._texts, self._references
            if self._csv_path:
                self.status.emit("Reading CSV...")
                texts, references = read_csv_reports(self._csv_path)

            pairs = [(text.strip(), references[index] if index < len(references) else "")
                     for index, text in enumerate(texts)
                     if isinstance(text, str) and text.strip()]
            if not pairs:
                self.failed.emit("No usable report text was found in the input.")
                return

            self.status.emit("Loading semantic encoder (first run may download the model)...")
            self.status.emit(f"Encoder ready - {self._pipeline.warm_up()}")

            emitted = 0
            for index, (narrative, reference) in enumerate(pairs, start=1):
                if self.isInterruptionRequested():
                    break
                result = self._pipeline.analyze(narrative, reference)
                payload = result.to_dict()
                payload["_timestamp"] = datetime.now().strftime("%H:%M:%S")
                self.row_ready.emit(payload)
                emitted += 1
                self.progress.emit(index, len(pairs))
                if self.STREAM_DELAY_MS:
                    self.msleep(self.STREAM_DELAY_MS)
            self.completed.emit(emitted)
        except Exception as exc:  # pragma: no cover - defensive GUI guard
            LOGGER.exception("Analysis failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ExtractionWorker(QThread):
    """Reads documents (PDF, scans, images, text) off the GUI thread."""

    document_ready = pyqtSignal(dict)
    failed = pyqtSignal(str)
    completed = pyqtSignal(int)

    def __init__(self, extractor: DocumentExtractor, paths: Sequence[str],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._extractor = extractor
        self._paths = list(paths)

    def run(self) -> None:  # noqa: D102 - documented on the class
        try:
            for document in self._extractor.extract_many(self._paths):
                if self.isInterruptionRequested():
                    break
                payload = document.to_dict()
                payload["name"] = os.path.basename(document.path)
                payload["blocks"] = len(document.blocks())
                payload["text"] = document.text
                payload["note"] = "; ".join(document.warnings) or "-"
                self.document_ready.emit(payload)
            self.completed.emit(len(self._paths))
        except Exception as exc:  # pragma: no cover - defensive GUI guard
            LOGGER.exception("Extraction failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class OCRProbeWorker(QThread):
    """Brings the OCR engine up (downloading models on first use) off the GUI thread."""

    probed = pyqtSignal(bool, str)

    def __init__(self, extractor: DocumentExtractor, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._extractor = extractor

    def run(self) -> None:  # noqa: D102 - documented on the class
        ok, message = self._extractor.probe()
        LOGGER.info("OCR probe: %s", message) if ok else LOGGER.warning("OCR probe: %s", message)
        self.probed.emit(ok, message)


class TrainingWorker(QThread):
    """Trains the XGBoost model and logs the run to MLflow."""

    trained = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, service: MLOpsService, results: Sequence[PipelineResult],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._service = service
        self._results = list(results)

    def run(self) -> None:  # noqa: D102 - documented on the class
        try:
            report = self._service.train(self._results)
            self.trained.emit(report.to_dict())
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            LOGGER.warning("Training failed: %s", exc)
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def read_csv_reports(path: str) -> Tuple[List[str], List[str]]:
    """Extract narratives and references from a CSV export.

    Recognises the column names in :data:`sif.lexical.CSV_TEXT_COLUMNS`; without
    one, the longest text cell in each row is used, which keeps the importer
    usable with arbitrary contractor spreadsheets.
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
            dialect = csv.excel

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
        else:
            handle.seek(0)
            narratives = [line.strip() for line in handle if line.strip()]
            references = [""] * len(narratives)
    return narratives, references


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Sidebar-navigated console wiring the views to the analysis stack."""

    #: How often the Settings page pulls new log lines (ms).
    LOG_REFRESH_MS = 1500

    def __init__(self) -> None:
        super().__init__()
        self.ring = configure_logging("INFO")
        LOGGER.info("%s starting", APP_NAME)

        self.pipeline = SIFPipeline()
        self.mlops = MLOpsService()
        self.extractor = DocumentExtractor()
        self.rows: List[Dict[str, object]] = []
        self.documents: List[Dict[str, object]] = []
        self.pending_blocks: List[str] = []
        self.worker: Optional[QThread] = None
        self.last_run_count = 0

        self.setWindowTitle(APP_NAME)
        self.resize(1600, 980)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._connect_views()

        if self.mlops.load_existing():
            self.pipeline.attach_model(self.mlops)
            LOGGER.info("Attached previously trained model")
        self._refresh_settings()

        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self._refresh_logs)
        self.log_timer.start(self.LOG_REFRESH_MS)

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        self.sidebar = Sidebar(NAV_ITEMS)
        self.sidebar.navigated.connect(self.navigate)
        self.sidebar.select("dashboard")

        self.header = HeaderBar(APP_NAME, APP_SUBTITLE, "HSE Analyst", "Team Member")
        self.header.search_changed.connect(self._apply_filter)

        self.dashboard = DashboardView()
        self.batch_view = BatchUploadView()
        self.matrix_view = TableView(
            "Parsed Incident Matrix",
            "Every analysed report with its verdict, risk score and extracted fields.",
            MATRIX_COLUMNS)
        self.hotspot_view = TableView(
            "Risk Hotspots",
            "Locations, rule-at-location repeats and barrier failures occurring more than "
            "once, ranked by SIF count then mean risk.",
            HOTSPOT_COLUMNS)
        self.review_view = TableView(
            "Human Review Queue",
            "Reports a person must verify: model or rule disagreement, critical risk, thin "
            "evidence, or high energy with no rule match.",
            REVIEW_COLUMNS)
        self.analytics_view = AnalyticsView()
        self.settings_view = SettingsView()

        self.pages = QStackedWidget()
        self._page_index: Dict[str, int] = {}
        for key, widget in (("dashboard", self.dashboard), ("batch", self.batch_view),
                            ("matrix", self.matrix_view), ("hotspots", self.hotspot_view),
                            ("review", self.review_view), ("analytics", self.analytics_view),
                            ("settings", self.settings_view)):
            self._page_index[key] = self.pages.addWidget(widget)
        # "Report Analysis" is the dashboard with the ingestion box focused.
        self._page_index["analysis"] = self._page_index["dashboard"]

        self.status_label = QLabel("Ready - paste a report, load the seed incidents or "
                                   "upload a document.")
        self.status_label.setObjectName("Faint")
        footer = QFrame()
        footer.setObjectName("Footer")
        footer.setFixedHeight(34)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(20, 6, 20, 6)
        self.footer_left = QLabel("Oil India Limited  ·  PS 26165 - UA/UC & Near-Miss "
                                  "Intelligence")
        self.footer_left.setObjectName("Faint")
        footer_layout.addWidget(self.footer_left)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.status_label)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.header)
        right_layout.addWidget(self.pages, stretch=1)
        right_layout.addWidget(footer)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.sidebar)
        layout.addWidget(right, stretch=1)
        self.setCentralWidget(container)
        self._build_menu()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for label, shortcut, slot in (
            ("&Batch Import CSV...", "Ctrl+O", self.import_csv),
            ("Add &Documents...", "Ctrl+D", self.add_documents),
            ("&Export Results CSV...", "Ctrl+S", self.export_csv),
        ):
            action = QAction(label, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _connect_views(self) -> None:
        self.dashboard.analyse_requested.connect(self.analyse_text)
        self.dashboard.seed_requested.connect(self.load_seed_data)
        self.dashboard.csv_requested.connect(self.import_csv)
        self.dashboard.clear_requested.connect(self.clear_dashboard)
        self.dashboard.encoder_changed.connect(self.change_encoder)
        self.dashboard.row_selected.connect(self.select_row)
        self.dashboard.review_requested.connect(self.mark_for_review)

        self.batch_view.files_requested.connect(self.add_documents)
        self.batch_view.csv_requested.connect(self.import_csv)
        self.batch_view.analyse_requested.connect(self.analyse_extracted)
        self.batch_view.clear_requested.connect(self.clear_documents)

        self.settings_view.log_level_changed.connect(self.change_log_level)
        self.settings_view.logs_cleared.connect(self.clear_logs)
        self.settings_view.logs_refreshed.connect(self._refresh_logs)
        self.settings_view.train_requested.connect(self.train_model)
        self.settings_view.tracking_changed.connect(self.change_tracking)
        self.settings_view.ocr_toggled.connect(self.change_ocr)
        self.settings_view.ocr_test_requested.connect(self._check_ocr)

    # -- navigation --------------------------------------------------------

    def navigate(self, key: str) -> None:
        """Switch pages from the sidebar."""
        self.pages.setCurrentIndex(self._page_index.get(key, 0))
        if key == "analysis":
            self.dashboard.input_box.setFocus()
        elif key == "settings":
            self._refresh_settings()
            self._refresh_logs()

    # -- analysis actions --------------------------------------------------

    def analyse_text(self, raw: str) -> None:
        """Analyse the narratives in the ingestion box."""
        blocks = [block.strip() for block in (raw or "").split("\n\n") if block.strip()]
        if not blocks:
            QMessageBox.information(self, APP_NAME,
                                    "Paste at least one report narrative before analysing.")
            return
        self._start(AnalysisWorker(self.pipeline, texts=blocks, parent=self))

    def load_seed_data(self) -> None:
        """Load and analyse the five seed incidents."""
        self.dashboard.input_box.setPlainText("\n\n".join(SEED_REPORTS))
        references = [f"SEED-{number:02d}" for number in range(1, len(SEED_REPORTS) + 1)]
        self._start(AnalysisWorker(self.pipeline, texts=list(SEED_REPORTS),
                                   references=references, parent=self))

    def import_csv(self) -> None:
        """Pick a CSV of reports and analyse every row."""
        path, _ = QFileDialog.getOpenFileName(self, "Batch Import UA/UC Reports", os.getcwd(),
                                              "CSV files (*.csv);;All files (*)")
        if path:
            self._start(AnalysisWorker(self.pipeline, csv_path=path, parent=self))

    def add_documents(self) -> None:
        """Pick documents and extract their text (OCR where needed)."""
        paths, _ = QFileDialog.getOpenFileNames(self, "Add report documents", os.getcwd(),
                                                DOCUMENT_FILTER)
        if not paths:
            return
        self.navigate("batch")
        self.sidebar.select("batch")
        worker = ExtractionWorker(self.extractor, paths, parent=self)
        worker.document_ready.connect(self.on_document_ready)
        worker.failed.connect(self.on_failed)
        worker.completed.connect(lambda count: self._set_status(
            f"Extracted {count} document(s) - {len(self.pending_blocks)} report block(s) ready"))
        self._start(worker, busy_message="Extracting document text...")

    def analyse_extracted(self) -> None:
        """Analyse the blocks recovered from uploaded documents."""
        if not self.pending_blocks:
            QMessageBox.information(self, APP_NAME,
                                    "Add documents first - no extracted text is waiting.")
            return
        references = [f"DOC-{number:03d}" for number in range(1, len(self.pending_blocks) + 1)]
        blocks = list(self.pending_blocks)
        self.pending_blocks.clear()
        self._start(AnalysisWorker(self.pipeline, texts=blocks, references=references,
                                   parent=self))

    def export_csv(self) -> None:
        """Write the incident matrix to a CSV file."""
        if not self.rows:
            QMessageBox.information(self, APP_NAME, "There is nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Parsed Results", os.path.join(os.getcwd(), "sif_results.csv"),
            "CSV files (*.csv)")
        if not path:
            return
        headers = [label for label, _, _ in MATRIX_COLUMNS] + ["Explanation"]
        try:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                for index, row in enumerate(self.rows, start=1):
                    writer.writerow([index] + [str(row.get(key, ""))
                                               for _, key, _ in MATRIX_COLUMNS[1:]]
                                    + [row.get("explanation", "")])
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not write the file:\n{exc}")
            return
        LOGGER.info("Exported %d rows to %s", len(self.rows), path)
        self._set_status(f"Exported {len(self.rows)} rows to {path}")

    def clear_dashboard(self) -> None:
        """Reset every analysis result."""
        self.rows.clear()
        self.last_run_count = 0
        self.dashboard.matrix_table.set_rows([])
        self.dashboard.show_detail(None)
        self._refresh()
        LOGGER.info("Dashboard cleared")
        self._set_status("Dashboard cleared.")

    def clear_documents(self) -> None:
        """Drop the extraction list and any pending blocks."""
        self.documents.clear()
        self.pending_blocks.clear()
        self.batch_view.set_documents([])
        self.batch_view.set_preview("")
        self._set_status("Extraction list cleared.")

    def change_encoder(self, backend: str) -> None:
        """Rebuild the pipeline around a different encoder."""
        self.pipeline = SIFPipeline(backend=backend)
        if self.mlops.model.is_trained:
            self.pipeline.attach_model(self.mlops)
        LOGGER.info("Encoder backend set to '%s'", backend)
        self._set_status(f"Encoder set to '{backend}' - it loads on the next run.")

    def mark_for_review(self) -> None:
        """Force the selected report into the review queue."""
        row = self.dashboard.matrix_table.currentRow()
        if row < 0 or row >= len(self.rows):
            return
        result = self.rows[row]
        result["needs_review"] = True
        result["review_trigger"] = result.get("review_trigger") or "Manual"
        result["review_reason"] = result.get("review_reason") or "Flagged by the analyst"
        LOGGER.info("Report %s marked for review by the analyst",
                    result.get("reference") or row + 1)
        self._refresh()

    # -- settings actions --------------------------------------------------

    def change_log_level(self, level: str) -> None:
        """Change the runtime logging level."""
        if level in LOG_LEVELS:
            set_level(level)
            LOGGER.info("Log level set to %s", level)
            self._refresh_logs()

    def clear_logs(self) -> None:
        """Empty the in-memory log buffer (the file keeps its history)."""
        self.ring.clear()
        self.settings_view.set_log_rows([])

    def change_tracking(self, uri: str, experiment: str) -> None:
        """Point MLflow at a different store or experiment."""
        self.mlops.tracker.tracking_uri = uri.strip() or self.mlops.tracker.tracking_uri
        self.mlops.tracker.experiment = experiment.strip() or self.mlops.tracker.experiment
        LOGGER.info("MLflow tracking set to %s (experiment '%s')",
                    self.mlops.tracker.tracking_uri, self.mlops.tracker.experiment)
        self._refresh_settings()

    def change_ocr(self, enabled: bool, language: str) -> None:
        """Rebuild the document extractor with new OCR settings."""
        self.extractor = DocumentExtractor(language=language, enable_ocr=enabled)
        LOGGER.info("OCR %s (language=%s)", "enabled" if enabled else "disabled", language)
        self._check_ocr()

    def train_model(self) -> None:
        """Train the XGBoost model on the analysed corpus."""
        results = self._as_results()
        if len(results) < 4:
            QMessageBox.information(
                self, APP_NAME,
                "Analyse at least four reports before training - the model needs both "
                "SIF-potential and non-SIF examples.")
            return
        worker = TrainingWorker(self.mlops, results, parent=self)
        worker.trained.connect(self.on_trained)
        worker.failed.connect(self.on_failed)
        self._start(worker, busy_message="Training XGBoost model...")

    # -- worker plumbing ---------------------------------------------------

    def _start(self, worker: QThread, busy_message: str = "Working...") -> None:
        """Connect the common signals of a worker and start it."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, APP_NAME,
                                    "A background task is already running. Please wait.")
            return
        self.worker = worker
        if isinstance(worker, AnalysisWorker):
            worker.row_ready.connect(self.on_row_ready)
            worker.progress.connect(self.dashboard.set_progress)
            worker.status.connect(self._set_status)
            worker.failed.connect(self.on_failed)
            worker.completed.connect(self.on_analysis_completed)
        worker.finished.connect(self._release_worker)
        self.dashboard.set_busy(True)
        self._set_status(busy_message)
        worker.start()

    def _release_worker(self) -> None:
        self.dashboard.set_busy(False)
        self.worker = None

    def on_row_ready(self, payload: Dict[str, object]) -> None:
        """Slot: one analysed report arrived."""
        self.rows.append(payload)
        self.dashboard.matrix_table.append_row(payload)
        self.dashboard.matrix_table.scrollToBottom()
        if len(self.rows) % 5 == 0:
            self._refresh()

    def on_analysis_completed(self, count: int) -> None:
        """Slot: a batch finished."""
        self.last_run_count = count
        intelligence = self._refresh()
        LOGGER.info("Analysed %d report(s); %s SIF-potential, %s awaiting review", count,
                    intelligence.kpis.get("sif_potential"), intelligence.kpis.get("needs_review"))
        self._set_status(
            f"Completed {count} report(s)  ·  {intelligence.kpis.get('sif_potential', 0)} "
            f"SIF-potential  ·  {intelligence.kpis.get('needs_review', 0)} for review  ·  "
            f"{intelligence.kpis.get('encoder', '')}")

    def on_document_ready(self, payload: Dict[str, object]) -> None:
        """Slot: one document was extracted."""
        text = str(payload.pop("text", ""))
        self.documents.append(payload)
        self.batch_view.set_documents(self.documents)
        if text.strip():
            blocks = [block.strip() for block in text.split("\n\n")
                      if len(block.strip()) > 25] or [text.strip()]
            self.pending_blocks.extend(blocks)
            self.batch_view.set_preview(text[:6000])
        self._set_status(f"{payload['name']} read via {payload['backend']}")

    def on_trained(self, report: Dict[str, object]) -> None:
        """Slot: training finished."""
        self.pipeline.attach_model(self.mlops)
        metrics = " ".join(f"{key}={value:.3f}"
                           for key, value in sorted(report.get("metrics", {}).items()))
        run = report.get("run_id") or "not tracked"
        warnings = report.get("warnings") or []
        self._refresh_settings()
        self._refresh()
        message = (f"Trained on {report.get('samples')} reports "
                   f"({report.get('positives')} positive)  ·  {metrics}  ·  run {run[:8]}")
        self._set_status(message)
        if warnings:
            QMessageBox.information(self, APP_NAME, message + "\n\n" + "\n".join(warnings))

    def on_failed(self, message: str) -> None:
        """Slot: a worker reported an error."""
        LOGGER.error("%s", message)
        self._set_status("Task failed - see the Settings log for detail.")
        QMessageBox.warning(self, APP_NAME, message)

    def select_row(self, row: int) -> None:
        """Slot: a matrix row was selected."""
        if 0 <= row < len(self.rows):
            self.dashboard.show_detail(self.rows[row])

    # -- rendering ---------------------------------------------------------

    def _as_results(self) -> List[PipelineResult]:
        """Rebuild dataclass results from the stored row dictionaries."""
        fields = PipelineResult.__dataclass_fields__
        return [PipelineResult(**{key: value for key, value in row.items() if key in fields})
                for row in self.rows]

    def _refresh(self):
        """Recompute aggregates and repaint every view."""
        results = self._as_results()
        intelligence = self.pipeline.aggregate(results)
        kpis = dict(intelligence.kpis)
        kpis["run_count"] = self.last_run_count
        kpis["model_agreement"] = self._model_agreement(results)

        self.dashboard.update_kpis(kpis)
        self.dashboard.update_charts(*self._chart_data(results))
        self.dashboard.update_tabs(len(intelligence.hotspots), len(intelligence.review_queue))
        hotspots = [spot.to_dict() for spot in intelligence.hotspots]
        review = [item.to_dict() for item in intelligence.review_queue]
        self.dashboard.hotspot_table.set_rows(hotspots)
        self.dashboard.review_table.set_rows(review)
        self.matrix_view.set_rows(self.rows)
        self.hotspot_view.set_rows(hotspots)
        self.review_view.set_rows(review)
        self.sidebar.set_badge("review", len(review))

        rules, energies, barriers = self._chart_data(results)
        self.analytics_view.update_charts(rules, energies, barriers,
                                          self._activity_data(results))
        report = self.mlops.last_report
        self.analytics_view.update_model(
            self._model_summary(), report.importances if report else [])
        return intelligence

    @staticmethod
    def _model_agreement(results: Sequence[PipelineResult]) -> Optional[float]:
        """Percentage of scored reports where the model agrees with the pipeline."""
        scored = [item for item in results if item.ml_active]
        if not scored:
            return None
        agree = sum(1 for item in scored if item.ml_flag == item.sif_potential)
        return agree / len(scored) * 100.0

    @staticmethod
    def _chart_data(results: Sequence[PipelineResult]):
        """Build (rule bars, energy slices, barrier bars) from the corpus."""
        rule_total: Counter = Counter()
        rule_sif: Counter = Counter()
        energies: Counter = Counter()
        barrier_total: Counter = Counter()
        barrier_sif: Counter = Counter()

        for item in results:
            rule_total[item.iogp_rule] += 1
            if item.sif_potential:
                rule_sif[item.iogp_rule] += 1
            if item.high_energy:
                for part in item.energy_source.split(" + "):
                    energies[part.strip()] += 1
            if item.barrier_failed:
                for part in item.barrier_failure.split(";"):
                    label = part.strip()
                    if not label:
                        continue
                    barrier_total[label] += 1
                    if item.sif_potential:
                        barrier_sif[label] += 1

        rules = [(label, count, rule_sif.get(label, 0))
                 for label, count in rule_total.most_common(10)]
        barriers = [(label, count, barrier_sif.get(label, 0))
                    for label, count in barrier_total.most_common(10)]
        return rules, energies.most_common(9), barriers

    @staticmethod
    def _activity_data(results: Sequence[PipelineResult]):
        """Activities ranked by how often they appear, SIF share highlighted."""
        total: Counter = Counter()
        flagged: Counter = Counter()
        for item in results:
            total[item.activity] += 1
            if item.sif_potential:
                flagged[item.activity] += 1
        return [(label, count, flagged.get(label, 0)) for label, count in total.most_common(12)]

    def _model_summary(self) -> str:
        status = self.mlops.status()
        report = self.mlops.last_report
        if report is None:
            return f"Model: {status['model']}. {status['tracking']}"
        return (f"Model: {status['model']}  |  labels: {report.label_source}  |  "
                f"MLflow run {report.run_id[:8] or 'not tracked'}. "
                + (" ".join(report.warnings) if report.warnings else ""))

    def _refresh_settings(self) -> None:
        status = self.mlops.status()
        self.settings_view.set_model_status(status["model"], status["tracking"])
        self.settings_view.set_log_path(active_log_file() or log_file_path())
        self.settings_view.set_runs(self.mlops.tracker.recent_runs(10))
        report = self.mlops.last_report
        self.settings_view.set_importances(report.importances if report else [])
        self.settings_view.set_ocr_status(self.extractor.status())
        self.batch_view.set_status(self.extractor.status())

    def _check_ocr(self) -> None:
        """Actually load the OCR engine and report what happened."""
        self.settings_view.set_ocr_status("Checking OCR - loading models, this may download...")
        worker = OCRProbeWorker(self.extractor, parent=self)
        worker.probed.connect(self._on_ocr_probed)
        self._start(worker, busy_message="Checking OCR availability...")

    def _on_ocr_probed(self, ok: bool, message: str) -> None:
        """Slot: the OCR probe finished."""
        text = message if ok else f"OCR unavailable - {message}"
        self.settings_view.set_ocr_status(text)
        self.batch_view.set_status(text)
        self._set_status("OCR ready" if ok else "OCR unavailable - see Settings")

    def _refresh_logs(self) -> None:
        """Pull recent log lines into the Settings table."""
        if self.pages.currentWidget() is not self.settings_view:
            return
        level = self.settings_view.level_box.currentText()
        rows = [{"timestamp": entry.timestamp, "level": entry.level,
                 "logger": entry.logger, "message": entry.message}
                for entry in self.ring.entries(level, limit=400)]
        self.settings_view.set_log_rows(rows)

    def _apply_filter(self, text: str) -> None:
        """Filter the incident matrix from the header search box."""
        needle = (text or "").strip().lower()
        table = self.matrix_view.table
        for row in range(table.rowCount()):
            haystack = " ".join(
                table.item(row, column).text().lower()
                for column in range(table.columnCount()) if table.item(row, column))
            table.setRowHidden(row, bool(needle) and needle not in haystack)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    # -- Qt lifecycle ------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop timers and any running worker before closing."""
        self.log_timer.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(3000)
        LOGGER.info("%s closing", APP_NAME)
        super().closeEvent(event)


def create_application(argv: Optional[List[str]] = None) -> QApplication:
    """Build a configured :class:`QApplication` instance."""
    app = QApplication(argv if argv is not None else [])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Oil India Limited")
    return app
