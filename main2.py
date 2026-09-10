"""Controller for the second build of SENTRA (``app2.py``).

What this build adds over ``main.py`` (which ``app.py`` keeps using, unchanged):

* a **workflow map** that shows every capability, its live status, and runs it;
* **Indian-language ingestion** - PaddleOCR in Hindi, Marathi, Tamil, Telugu,
  Kannada, Urdu and the rest of the Devanagari family, with optional translation
  to English so the analysers can read the report;
* an **optional local LLM analyser** (Ollama) as a fourth opinion that never
  overrides the pipeline;
* a **dashboard on its own page**, separate from ingestion and from the matrix;
* **no pictographic icons** anywhere in the interface.

Every long-running step - OCR, translation, analysis, training, connectivity
probes - happens on a ``QThread``.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from main import read_csv_reports
from sif import SEED_REPORTS, SIFPipeline
from sif.narrative import corpus_bulletin
from sif.llm import OllamaEngine, looks_non_latin
from sif.logging_setup import (LOG_LEVELS, active_log_file, configure_logging,
                               log_file_path, set_level)
from sif.mlops import MLOpsService
from sif import prefs
from sif.audit import AuditLog
from sif.ocr import (LANGUAGE_CHOICES, UNSUPPORTED_LANGUAGES, DocumentExtractor,
                     cache_directory, models_present)
from sif.pipeline import PipelineResult
from sif.review import DECISION_LABELS, DECISION_SHORT, DecisionLog, fingerprint
from sif.updater import UpdateChecker, UpdateInfo
from sif.version import __version__, describe
from ui.theme import C, STYLESHEET
from ui.views import HOTSPOT_COLUMNS, AnalyticsView, TableView
from ui2.components import HeaderBar, Sidebar
from ui2.review import ReviewView
from ui2.views import DashboardView, EnginesView, IngestView, ReportView, SettingsView
from ui2.workflow import WorkflowMap

__all__ = ["AnalysisWorker", "ExtractionWorker", "TrainingWorker", "ProbeWorker",
           "UpdateWorker", "DownloadWorker", "UpdateDialog", "MainWindow",
           "create_application"]

LOGGER = logging.getLogger("sif.app2")

APP_NAME = "SENTRA"
#: The product is SENTRA; "SIF Insight Console" was the working title and
#: survives only where renaming would move an operator's data - the
#: settings directory in :mod:`sif.prefs`, which holds their review
#: decisions, and the ``sif`` package itself.
APP_LONG_NAME = "SENTRA - SIF Insight Console"
APP_SUBTITLE = ("Sense the Risk  ·  Stop the Incident   |   UA/UC and near-miss "
                "intelligence   |   PS 26165   |   build 2")
#: Wait before the start-up update check so it never competes with first paint.
UPDATE_CHECK_DELAY_MS = 4000

NAV_ITEMS = (
    ("workflow", "Workflow map"),
    ("ingest", "Ingest and OCR"),
    ("dashboard", "Dashboard"),
    ("reports", "Reports and evidence"),
    ("hotspots", "Risk hotspots"),
    ("review", "Human review"),
    ("analytics", "Analytics"),
    ("engines", "Engines"),
    ("settings", "Settings"),
)

DOCUMENT_FILTER = ("Documents (*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.txt *.md);;"
                   "All files (*)")


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


class AnalysisWorker(QThread):
    """Translate where needed, then run the pipeline, streaming rows back."""

    row_ready = pyqtSignal(dict)
    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    failed = pyqtSignal(str)
    completed = pyqtSignal(int)

    STREAM_DELAY_MS = 25

    def __init__(self, pipeline: SIFPipeline, texts: Optional[Sequence[str]] = None,
                 csv_path: Optional[str] = None, references: Optional[Sequence[str]] = None,
                 translator: object = None, language: str = "English",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._texts = list(texts or [])
        self._references = list(references or [])
        self._csv_path = csv_path
        self._translator = translator
        self._language = language

    def run(self) -> None:  # noqa: D102 - documented on the class
        try:
            texts, references = self._texts, self._references
            if self._csv_path:
                self.status.emit("Reading CSV export")
                texts, references = read_csv_reports(self._csv_path)

            pairs = [(text.strip(), references[index] if index < len(references) else "")
                     for index, text in enumerate(texts)
                     if isinstance(text, str) and text.strip()]
            if not pairs:
                self.failed.emit("No usable report text was found in the input.")
                return

            self.status.emit("Loading semantic encoder (first run may download the model)")
            self.status.emit(f"Encoder ready - {self._pipeline.warm_up()}")

            emitted = 0
            for index, (narrative, reference) in enumerate(pairs, start=1):
                if self.isInterruptionRequested():
                    break
                translated, language = "", ""
                if self._translator is not None and looks_non_latin(narrative):
                    self.status.emit(f"Translating report {index} to English")
                    translated = self._translator.translate(narrative)
                    language = self._language
                    if not translated:
                        self.status.emit(
                            f"Translation unavailable for report {index} - analysing as written")
                result = self._pipeline.analyze(narrative, reference, translated, language)
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
    """Read documents (PDF text layer, then OCR) off the GUI thread."""

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


class TrainingWorker(QThread):
    """Train the model and log the run."""

    trained = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, service: MLOpsService, results: Sequence[PipelineResult],
                 labels: Optional[Sequence[int]] = None, label_source: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._service = service
        self._results = list(results)
        self._labels = list(labels) if labels is not None else None
        self._label_source = label_source

    def run(self) -> None:  # noqa: D102 - documented on the class
        try:
            self.trained.emit(self._service.train(
                self._results, labels=self._labels,
                label_source=self._label_source or None).to_dict())
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            LOGGER.warning("Training failed: %s", exc)
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ProbeWorker(QThread):
    """Run a connectivity check (OCR models, Ollama) without blocking the GUI."""

    probed = pyqtSignal(str, bool, str)

    def __init__(self, name: str, check, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._name = name
        self._check = check

    def run(self) -> None:  # noqa: D102 - documented on the class
        try:
            ok, message = self._check()
        except Exception as exc:  # noqa: BLE001 - a probe must never raise
            ok, message = False, f"{type(exc).__name__}: {exc}"
        LOGGER.info("%s probe: %s", self._name, message)
        self.probed.emit(self._name, bool(ok), str(message))


class UpdateWorker(QThread):
    """Ask GitHub whether a newer release exists, off the GUI thread."""

    checked = pyqtSignal(object)

    def __init__(self, checker: UpdateChecker, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checker = checker

    def run(self) -> None:  # noqa: D102 - documented on the class
        self.checked.emit(self._checker.check())


class DownloadWorker(QThread):
    """Download and verify an installer, reporting progress."""

    progress = pyqtSignal(int, int)
    finished_path = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, checker: UpdateChecker, info: UpdateInfo,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checker = checker
        self._info = info

    def run(self) -> None:  # noqa: D102 - documented on the class
        try:
            path = self._checker.download(self._info, progress=self.progress.emit)
        except Exception as exc:  # noqa: BLE001 - reported, never raised into Qt
            LOGGER.warning("Update download failed: %s", exc)
            self.failed.emit(str(exc))
            return
        self.finished_path.emit(path)


class UpdateDialog(QDialog):
    """Offers the update, downloads it on request, and never installs silently."""

    def __init__(self, info: UpdateInfo, checker: UpdateChecker,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.info = info
        self.checker = checker
        self.worker: Optional[DownloadWorker] = None
        self.installer_path = ""

        self.setWindowTitle("Update available")
        self.setMinimumWidth(560)

        headline = QLabel(f"Version {info.latest} is available. You are running "
                          f"{info.current}.")
        headline.setStyleSheet("font-size: 14px; font-weight: 700;")
        detail = QLabel(f"{info.asset.name}  ·  {info.asset.megabytes} MB"
                        if info.asset else "No installer was published for this platform.")
        detail.setObjectName("Faint")

        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setPlainText(info.notes or "No release notes were published.")
        notes.setFixedHeight(150)

        self.status = QLabel("The download is checked against the release checksum "
                             "before anything is installed.")
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        buttons = QDialogButtonBox()
        self.install_button = buttons.addButton("Download and install",
                                                QDialogButtonBox.ButtonRole.AcceptRole)
        self.later_button = buttons.addButton("Later", QDialogButtonBox.ButtonRole.RejectRole)
        self.skip_button = buttons.addButton(f"Skip {info.latest}",
                                             QDialogButtonBox.ButtonRole.DestructiveRole)
        self.install_button.setEnabled(bool(info.asset))
        self.install_button.clicked.connect(self.start_download)
        self.later_button.clicked.connect(self.reject)
        self.skip_button.clicked.connect(self._skip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(8)
        layout.addWidget(headline)
        layout.addWidget(detail)
        layout.addWidget(notes)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(buttons)

    def _skip(self) -> None:
        prefs.set_value("skip_version", self.info.latest)
        LOGGER.info("Operator chose to skip version %s", self.info.latest)
        self.reject()

    def start_download(self) -> None:
        """Fetch the installer; the dialog stays open until it is verified."""
        self.install_button.setEnabled(False)
        self.skip_button.setEnabled(False)
        self.progress.setVisible(True)
        self.status.setText("Downloading...")
        self.worker = DownloadWorker(self.checker, self.info, parent=self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_path.connect(self._on_downloaded)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        if total:
            self.status.setText(f"Downloading... {done * 100 // total}%")

    def _on_downloaded(self, path: str) -> None:
        self.installer_path = path
        self.status.setText("Checksum verified. The console will close so the installer "
                            "can replace it.")
        self.accept()

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.install_button.setEnabled(True)
        self.skip_button.setEnabled(True)
        self.status.setText(f"Update refused: {message}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Workflow-led console: the map first, then each capability as its own page."""

    LOG_REFRESH_MS = 1500

    def __init__(self) -> None:
        super().__init__()
        self.ring = configure_logging("INFO")
        LOGGER.info("%s (build 2) starting", APP_NAME)

        self.pipeline = SIFPipeline()
        self.mlops = MLOpsService()
        self.extractor = DocumentExtractor()
        self.llm = OllamaEngine()
        self.llm_enabled = False
        #: Last known reachability, refreshed by the probe worker. The map reads
        #: this instead of opening a socket on the GUI thread every repaint.
        self.llm_online = False
        self.llm_message = "not checked yet"
        self.translate_enabled = True
        self.language = LANGUAGE_CHOICES[0]

        self.updater = UpdateChecker()
        self.update_worker: Optional[UpdateWorker] = None
        self.rows: List[Dict[str, object]] = []
        #: Narrative -> row index, so a report analysed twice replaces itself.
        self._by_narrative: Dict[str, int] = {}
        #: Repeats seen in the batch now running, reported when it finishes.
        self.duplicates_seen = 0
        #: What an auditor reads: append-only, separate from the debug log.
        self.audit = AuditLog(version=__version__)
        self.audit.system("console started", build="2", version=describe())
        #: Set in closeEvent, so deferred work started by a timer can stand down.
        self._closing = False
        self.decisions = DecisionLog().load()
        self.queue_rows: List[Dict[str, object]] = []
        self.outstanding_reviews = 0
        self.documents: List[Dict[str, object]] = []
        self.pending_blocks: List[str] = []
        self.worker: Optional[QThread] = None

        self.setWindowTitle(f"{APP_NAME} - build 2")
        self.resize(1620, 1000)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._connect()

        if self.mlops.load_existing():
            self.pipeline.attach_model(self.mlops)
            LOGGER.info("Attached previously trained model")
            self.audit.system("trained model attached",
                              state=self.mlops.status()["model"])

        self._refresh_engines()
        self._refresh_workflow()
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self._refresh_logs)
        self.log_timer.start(self.LOG_REFRESH_MS)

        # Check for a new release shortly after start-up, quietly: if there is
        # nothing new, or no network, the operator never sees a thing.
        if prefs.get("check_updates", True):
            QTimer.singleShot(UPDATE_CHECK_DELAY_MS, lambda: self.check_for_updates(False))

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        self.sidebar = Sidebar(NAV_ITEMS)
        self.sidebar.navigated.connect(self.navigate)
        self.sidebar.select("workflow")

        self.header = HeaderBar(APP_NAME, APP_SUBTITLE, "HSE Analyst", "Team member")
        self.header.search_changed.connect(self._apply_filter)

        self.workflow = WorkflowMap()
        self.workflow.stage_activated.connect(self.run_stage)
        self.ingest_view = IngestView(LANGUAGE_CHOICES, UNSUPPORTED_LANGUAGES)
        self.dashboard = DashboardView()
        self.report_view = ReportView()
        self.hotspot_view = TableView(
            "Risk hotspots",
            "Sites, activities, rule-at-location repeats and barrier failures occurring "
            "more than once, ranked by SIF-precursor density.",
            HOTSPOT_COLUMNS)
        self.review_view = ReviewView()
        self.review_view.set_reviewer(str(prefs.get("reviewer", "") or ""))
        self.analytics_view = AnalyticsView()
        self.engines_view = EnginesView()
        self.settings_view = SettingsView()

        self.pages = QStackedWidget()
        self._page_index: Dict[str, int] = {}
        for key, widget in (("workflow", self.workflow), ("ingest", self.ingest_view),
                            ("dashboard", self.dashboard), ("reports", self.report_view),
                            ("hotspots", self.hotspot_view), ("review", self.review_view),
                            ("analytics", self.analytics_view),
                            ("engines", self.engines_view), ("settings", self.settings_view)):
            self._page_index[key] = self.pages.addWidget(widget)

        self.status_label = QLabel("Ready. Start on the workflow map.")
        self.status_label.setObjectName("Faint")
        footer = QFrame()
        footer.setObjectName("Footer")
        footer.setFixedHeight(34)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(20, 6, 20, 6)
        left = QLabel("Oil India Limited  ·  PS 26165  ·  prototype output, not a "
                      "statutory record")
        left.setObjectName("Faint")
        footer_layout.addWidget(left)
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

        file_menu = self.menuBar().addMenu("&File")
        for label, shortcut, slot in (
            ("&Import CSV export...", "Ctrl+O", self.import_csv),
            ("Add &documents...", "Ctrl+D", self.add_documents),
            ("&Export results CSV...", "Ctrl+S", self.export_csv),
            ("Generate &safety bulletin...", "Ctrl+B", self.generate_bulletin),
            ("Export the &audit trail...", "", self.export_audit),
            ("&Clear the corpus", "", self.confirm_clear_corpus),
        ):
            action = QAction(label, self)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        update_action = QAction("Check for &updates...", self)
        update_action.triggered.connect(lambda: self.check_for_updates(True))
        help_menu.addAction(update_action)
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _connect(self) -> None:
        self.ingest_view.analyse_requested.connect(self.analyse_text)
        self.ingest_view.seed_requested.connect(self.load_seed_data)
        self.ingest_view.csv_requested.connect(self.import_csv)
        self.ingest_view.files_requested.connect(self.add_documents)
        self.ingest_view.analyse_documents_requested.connect(self.analyse_extracted)
        self.ingest_view.clear_requested.connect(self.clear_documents)
        self.ingest_view.language_changed.connect(self.change_language)
        self.ingest_view.translate_toggled.connect(self.set_translation)

        self.report_view.row_selected.connect(self.select_row)

        self.review_view.row_selected.connect(self.select_review_row)
        self.review_view.decision_made.connect(self.record_decision)
        self.review_view.undo_requested.connect(self.undo_decision)
        self.review_view.export_requested.connect(self.export_decisions)
        self.review_view.reviewer_changed.connect(self.set_reviewer)
        self.review_view.show_decided.stateChanged.connect(lambda _: self._refresh())

        self.engines_view.encoder_changed.connect(self.change_encoder)
        self.engines_view.train_requested.connect(self.train_model)
        self.engines_view.ocr_check_requested.connect(self.check_ocr)
        self.engines_view.ollama_check_requested.connect(self.check_llm)
        self.engines_view.ollama_config_changed.connect(self.configure_llm)
        self.engines_view.ollama_toggled.connect(self.set_llm_enabled)

        self.settings_view.log_level_changed.connect(self.change_log_level)
        self.settings_view.logs_cleared.connect(self.clear_logs)
        self.settings_view.logs_refreshed.connect(self._refresh_logs)
        self.settings_view.tracking_changed.connect(self.change_tracking)
        self.settings_view.audit_refreshed.connect(self._refresh_audit)
        self.settings_view.audit_exported.connect(self.export_audit)
        self.settings_view.audit_filtered.connect(self._set_audit_filter)

    # -- navigation and the workflow map -----------------------------------

    def navigate(self, key: str) -> None:
        """Switch pages."""
        self.pages.setCurrentIndex(self._page_index.get(key, 0))
        self.sidebar.select(key)
        if key == "settings":
            self._refresh_logs()
        elif key == "engines":
            self._refresh_engines()
        elif key == "workflow":
            self._refresh_workflow()

    def run_stage(self, key: str) -> None:
        """A stage card was pressed on the workflow map."""
        actions = {
            "ingest": self.add_documents,
            "ocr": self.check_ocr,
            "translate": self.check_llm,
            "analyse": self.load_seed_data,
            "dashboard": lambda: self.navigate("dashboard"),
            "hotspots": lambda: self.navigate("hotspots"),
            "review": lambda: self.navigate("review"),
            "train": self.train_model,
        }
        action = actions.get(key)
        if action is not None:
            LOGGER.info("Workflow stage '%s' activated", key)
            action()

    def _refresh_workflow(self) -> None:
        """Recompute every stage's live status on the map."""
        ready, waiting, missing = C.OK, C.WARN, C.DANGER

        documents = len(self.documents)
        blocks = len(self.pending_blocks)
        self.workflow.set_status(
            "ingest",
            f"{documents} document(s) read, {blocks} block(s) ready"
            if documents else "no documents added yet",
            ready if documents else waiting)

        ocr_text = self.extractor.status()
        self.workflow.set_status(
            "ocr", f"{self.language}: {ocr_text.split(';')[0]}",
            ready if "ready" in ocr_text.lower() else
            missing if "unavailable" in ocr_text.lower() or "not usable" in ocr_text.lower()
            else waiting)

        if not self.translate_enabled:
            self.workflow.set_status("translate", "translation switched off", waiting)
        elif self.llm_online:
            self.workflow.set_status("translate", f"Ollama ready - {self.llm.model}", ready)
        else:
            self.workflow.set_status(
                "translate", f"Ollama not confirmed - {self.llm_message}", missing)

        analysed = len(self.rows)
        flagged = sum(1 for row in self.rows if row.get("sif_potential"))
        self.workflow.set_status(
            "analyse",
            f"{analysed} report(s) analysed, {flagged} SIF-potential"
            if analysed else "nothing analysed yet",
            ready if analysed else waiting)
        self.workflow.set_status(
            "dashboard", f"{analysed} report(s) on the dashboard" if analysed
            else "waiting for analysed reports", ready if analysed else waiting)

        hotspots = self.hotspot_view.table.rowCount()
        self.workflow.set_status(
            "hotspots", f"{hotspots} cluster(s) above the repeat threshold" if hotspots
            else "no repeats yet - needs at least two related reports",
            ready if hotspots else waiting)

        counts = self.decisions.counts()
        outstanding = self.outstanding_reviews
        if outstanding:
            review_text = (f"{outstanding} report(s) awaiting an expert  ·  "
                           f"{counts['decided']} decided")
        elif counts["decided"]:
            review_text = (f"queue clear  ·  {counts['decided']} decided, "
                           f"{counts['labels']} usable as labels")
        else:
            review_text = "queue empty"
        self.workflow.set_status(
            "review", review_text, waiting if outstanding else ready)

        status = self.mlops.status()["model"]
        self.workflow.set_status(
            "train", status,
            ready if "trained" in status else missing if "not installed" in status else waiting)

        encoder = (self.pipeline.encoder.info.backend
                   if self.pipeline._encoder is not None else "pending")
        self.header.set_engines(
            f"encoder: {encoder}"
            f"  ·  LLM: {'on' if self.llm_enabled else 'off'}"
            f"  ·  model: {'trained' if self.mlops.model.is_trained else 'none'}")

    # -- ingestion ---------------------------------------------------------

    def analyse_text(self, raw: str) -> None:
        """Analyse whatever is in the ingestion box."""
        blocks = [block.strip() for block in (raw or "").split("\n\n") if block.strip()]
        if not blocks:
            QMessageBox.information(self, APP_NAME,
                                    "Paste at least one report narrative before analysing.")
            return
        self._start(self._analysis_worker(texts=blocks))

    def load_seed_data(self) -> None:
        """Load and analyse the five seed incidents."""
        self.ingest_view.input_box.setPlainText("\n\n".join(SEED_REPORTS))
        references = [f"SEED-{number:02d}" for number in range(1, len(SEED_REPORTS) + 1)]
        self.navigate("ingest")
        self._start(self._analysis_worker(texts=list(SEED_REPORTS), references=references))

    def import_csv(self) -> None:
        """Analyse every row of a CSV export."""
        path, _ = QFileDialog.getOpenFileName(self, "Import UA/UC reports", os.getcwd(),
                                              "CSV files (*.csv);;All files (*)")
        if path:
            self._start(self._analysis_worker(csv_path=path))

    def add_documents(self) -> None:
        """Extract text from PDFs, scans and photographs."""
        paths, _ = QFileDialog.getOpenFileNames(self, "Add report documents", os.getcwd(),
                                                DOCUMENT_FILTER)
        if not paths:
            return
        self.navigate("ingest")
        worker = ExtractionWorker(self.extractor, paths, parent=self)
        worker.document_ready.connect(self.on_document_ready)
        worker.failed.connect(self.on_failed)
        worker.completed.connect(lambda count: self._set_status(
            f"Read {count} document(s); {len(self.pending_blocks)} block(s) ready to analyse"))
        self._start(worker, "Extracting document text")

    def analyse_extracted(self) -> None:
        """Analyse the blocks recovered from documents."""
        if not self.pending_blocks:
            QMessageBox.information(self, APP_NAME,
                                    "Add documents first - no extracted text is waiting.")
            return
        blocks = list(self.pending_blocks)
        self.pending_blocks.clear()
        references = [f"DOC-{number:03d}" for number in range(1, len(blocks) + 1)]
        self._start(self._analysis_worker(texts=blocks, references=references))

    def _analysis_worker(self, **kwargs) -> AnalysisWorker:
        self.duplicates_seen = 0
        # Reachability is checked once, in the worker, not per repaint.
        translator = self.llm if (self.translate_enabled and self.llm_online) else None
        return AnalysisWorker(self.pipeline, translator=translator, language=self.language,
                              parent=self, **kwargs)

    def export_csv(self) -> None:
        """Write the incident matrix to CSV."""
        if not self.rows:
            QMessageBox.information(self, APP_NAME, "There is nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results", os.path.join(os.getcwd(), "sif_results.csv"),
            "CSV files (*.csv)")
        if not path:
            return
        keys = ["reference", "sif_potential", "risk_score", "risk_band", "iogp_rule",
                "activity", "location", "barrier_failure", "energy_source",
                "source_language", "ml_probability", "llm_flag", "llm_rule",
                "review_trigger", "explanation", "raw_text"]
        try:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["#"] + keys)
                for index, row in enumerate(self.rows, start=1):
                    writer.writerow([index] + [str(row.get(key, "")) for key in keys])
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not write the file:\n{exc}")
            return
        LOGGER.info("Exported %d rows to %s", len(self.rows), path)
        self.audit.functionality("corpus exported", rows=len(self.rows), path=path)
        self._set_status(f"Exported {len(self.rows)} rows to {path}")

    def generate_bulletin(self) -> None:
        """Write an auto-generated safety bulletin over the analysed corpus.

        Deliberately templated from the same structured fields the dashboard
        renders, not sent through the local LLM: a circulated safety document
        is the wrong place for a model to improvise, and every line here has to
        trace back to a report the way everything else in this console does.
        """
        if not self.rows:
            QMessageBox.information(self, APP_NAME,
                                    "Analyse some reports first - there is nothing to "
                                    "summarise yet.")
            return
        results = self._as_results()
        intelligence = self.pipeline.aggregate(results)
        bulletin = corpus_bulletin(
            intelligence, results,
            period=datetime.now().strftime("Generated from the console, %Y-%m-%d"))

        path, _ = QFileDialog.getSaveFileName(
            self, "Generate safety bulletin",
            os.path.join(os.getcwd(), "sentra_bulletin.txt"), "Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(bulletin)
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not write the file:\n{exc}")
            return
        LOGGER.info("Generated a safety bulletin over %d report(s) to %s",
                    len(self.rows), path)
        self.audit.functionality("safety bulletin generated", reports=len(self.rows),
                                 path=path)
        self._set_status(f"Safety bulletin written to {path}")

    def confirm_clear_corpus(self) -> None:
        """Ask before dropping the corpus - it cannot be undone from here."""
        if not self.rows:
            QMessageBox.information(self, APP_NAME, "There are no reports to clear.")
            return
        answer = QMessageBox.question(
            self, APP_NAME,
            f"Clear all {len(self.rows)} analysed report(s)?\n\n"
            "Recorded review decisions are kept - they belong to the reports, not "
            "to this session - and reappear if the same reports are analysed again.")
        if answer == QMessageBox.StandardButton.Yes:
            self.clear_corpus()

    def clear_corpus(self) -> None:
        """Drop every analysed report, so a fresh corpus starts empty."""
        self.rows.clear()
        self._by_narrative.clear()
        self.duplicates_seen = 0
        self.report_view.table.set_rows([])
        self.report_view.show_detail(None)
        self.audit.functionality("corpus cleared")
        self._refresh()
        self._set_status("Corpus cleared - no reports are loaded.")

    def clear_documents(self) -> None:
        """Drop the extraction list."""
        self.documents.clear()
        self.pending_blocks.clear()
        self.ingest_view.set_documents([])
        self.ingest_view.set_preview("")
        self._refresh_workflow()
        self._set_status("Extraction list cleared.")

    # -- engine configuration ----------------------------------------------

    def change_language(self, language: str) -> None:
        """Rebuild the extractor for a different OCR language."""
        self.language = language
        self.extractor = DocumentExtractor(language=language,
                                           enable_ocr=self.extractor.enable_ocr)
        LOGGER.info("OCR language set to %s (code %s)", language, self.extractor.language)
        self.audit.system("OCR language changed", language=language,
                          code=self.extractor.language)
        status = self.extractor.status()
        self.ingest_view.set_ocr_status(status)
        self.engines_view.set_ocr_status(status)
        self._refresh_workflow()

    def set_translation(self, enabled: bool) -> None:
        """Turn pre-analysis translation on or off."""
        self.translate_enabled = enabled
        LOGGER.info("Translation %s", "enabled" if enabled else "disabled")
        self._refresh_workflow()

    def set_llm_enabled(self, enabled: bool) -> None:
        """Attach or detach the local LLM as an extra analyser."""
        self.llm_enabled = enabled
        self.pipeline.attach_llm(self.llm if enabled else None)
        LOGGER.info("Local LLM analyser %s", "enabled" if enabled else "disabled")
        self._refresh_workflow()

    def configure_llm(self, host: str, model: str) -> None:
        """Point the client at a different host or model."""
        self.llm = OllamaEngine(host=host.strip() or self.llm.host,
                                model=model.strip() or self.llm.model)
        self.llm_online = False
        self.llm_message = "not checked since the host changed"
        if self.llm_enabled:
            self.pipeline.attach_llm(self.llm)
        LOGGER.info("Ollama configured: %s / %s", self.llm.host, self.llm.model)
        self.check_llm()

    def change_encoder(self, backend: str) -> None:
        """Rebuild the pipeline around a different encoder."""
        llm = self.llm if self.llm_enabled else None
        self.pipeline = SIFPipeline(backend=backend)
        if self.mlops.model.is_trained:
            self.pipeline.attach_model(self.mlops)
        if llm is not None:
            self.pipeline.attach_llm(llm)
        LOGGER.info("Encoder backend set to '%s'", backend)
        self.engines_view.set_encoder_status(
            f"Encoder set to '{backend}' - it loads on the next run.")
        self._set_status(f"Encoder set to '{backend}'.")

    def check_ocr(self) -> None:
        """Fetch the OCR models if this machine lacks them, then prove they load.

        The models are a one-time download per machine, kept on disk afterwards,
        so this says which of the two is happening rather than making an operator
        wonder whether it is reinstalling on every run.
        """
        first_time = not models_present()
        LOGGER.info("OCR model cache: %s (%s)", cache_directory(),
                    "empty - will download" if first_time else "already populated")
        message = ("Downloading the OCR models - this happens once on this machine"
                   if first_time else "Loading the OCR models already on this machine")
        self.ingest_view.set_ocr_status(message)
        self.engines_view.set_ocr_status(message)
        self._start(ProbeWorker("OCR", self.extractor.probe, parent=self), message)

    def check_llm(self) -> None:
        """Probe the Ollama host."""
        def probe():
            return self.llm.available(), self.llm.status()

        self.engines_view.set_llm_status("Checking the local LLM host")
        self._start(ProbeWorker("Ollama", probe, parent=self), "Checking Ollama")

    #: Below this many reviewed labels, human decisions are too few to train on
    #: alone and the pipeline's own verdicts are used instead.
    MIN_HUMAN_LABELS = 8

    def train_model(self) -> None:
        """Train the learned model, on reviewed decisions when there are enough.

        This is where the review queue pays for itself. Given enough decided
        reports the model learns from what experts concluded; short of that it
        falls back to the pipeline's own verdicts, which teaches it to reproduce
        the rules and nothing more. Either way the operator is told which it was,
        because the two are worth very different amounts.
        """
        results = self._as_results()
        if len(results) < 4:
            QMessageBox.information(
                self, APP_NAME,
                "Analyse at least four reports before training - the model needs both "
                "SIF-potential and non-SIF examples.")
            return

        reviewed, labels = self.decisions.labels_for(results)
        if len(labels) >= self.MIN_HUMAN_LABELS and len(set(labels)) > 1:
            worker = TrainingWorker(self.mlops, reviewed, labels,
                                    "human review decisions", parent=self)
            note = f"on {len(labels)} reviewed decision(s)"
        else:
            shortfall = (f"only {len(labels)} reviewed label(s); "
                         f"{self.MIN_HUMAN_LABELS} are needed"
                         if len(set(labels)) > 1 or not labels
                         else f"{len(labels)} reviewed label(s), all the same verdict")
            LOGGER.info("Training on pipeline verdicts - %s", shortfall)
            worker = TrainingWorker(self.mlops, results,
                                    label_source="weak (pipeline verdicts)", parent=self)
            note = f"on {len(results)} pipeline verdict(s) - {shortfall}"
        worker.trained.connect(self.on_trained)
        worker.failed.connect(self.on_failed)
        self._start(worker, f"Training the model {note}")

    def change_log_level(self, level: str) -> None:
        if level in LOG_LEVELS:
            set_level(level)
            LOGGER.info("Log level set to %s", level)
            self._refresh_logs()

    def clear_logs(self) -> None:
        self.ring.clear()
        self.settings_view.set_log_rows([])

    def change_tracking(self, uri: str, experiment: str) -> None:
        self.mlops.tracker.tracking_uri = uri.strip() or self.mlops.tracker.tracking_uri
        self.mlops.tracker.experiment = experiment.strip() or self.mlops.tracker.experiment
        LOGGER.info("MLflow tracking set to %s (experiment '%s')",
                    self.mlops.tracker.tracking_uri, self.mlops.tracker.experiment)
        self._refresh_engines()

    # -- updates -----------------------------------------------------------

    def check_for_updates(self, interactive: bool) -> None:
        """Ask GitHub for a newer release.

        ``interactive`` is True when the operator asked, in which case "you are up
        to date" is worth saying; the start-up check stays silent unless there is
        something to offer.
        """
        if self._closing:
            # The start-up check is scheduled four seconds out; a window closed
            # before then must not start a thread nobody is left to wait for.
            LOGGER.debug("Skipping the update check - the window is closing")
            return
        if self.update_worker is not None and self.update_worker.isRunning():
            return
        self._interactive_update = interactive
        if interactive:
            self._set_status("Checking for updates")
        self.update_worker = UpdateWorker(self.updater, parent=self)
        self.update_worker.checked.connect(self.on_update_checked)
        self.update_worker.start()

    def on_update_checked(self, info: UpdateInfo) -> None:
        """Slot: the update check finished."""
        interactive = getattr(self, "_interactive_update", False)
        LOGGER.info("Update check: %s", info.summary())
        self._set_status(info.summary())

        if info.error or not info.available:
            if interactive:
                QMessageBox.information(self, APP_NAME, info.summary())
            return
        if prefs.get("skip_version") == info.latest and not interactive:
            LOGGER.info("Version %s was skipped by the operator", info.latest)
            return
        if not info.installable:
            # A source checkout, or no asset for this platform: say so, do not offer.
            if interactive:
                QMessageBox.information(self, APP_NAME, info.summary())
            return

        dialog = UpdateDialog(info, self.updater, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.installer_path:
            if self.updater.launch(dialog.installer_path):
                LOGGER.info("Closing so the installer can run")
                self.close()
            else:
                QMessageBox.warning(
                    self, APP_NAME,
                    "The installer was downloaded and verified but could not be "
                    f"started. Run it manually:\n{dialog.installer_path}")

    def show_about(self) -> None:
        """Version and provenance, which is what a support call asks for first."""
        QMessageBox.information(
            self, APP_NAME,
            f"{APP_LONG_NAME} (build 2)\n"
            f"Version {describe()}\n\n"
            "Oil India Limited - Problem Statement 26165\n"
            f"Updates: {self.updater.repository}\n\n"
            "Prototype output - for review, not a statutory record.")

    # -- worker plumbing ---------------------------------------------------

    def _start(self, worker: QThread, message: str = "Working") -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, APP_NAME,
                                    "A background task is already running. Please wait.")
            return
        self.worker = worker
        if isinstance(worker, AnalysisWorker):
            worker.row_ready.connect(self.on_row_ready)
            worker.progress.connect(self.ingest_view.set_progress)
            worker.status.connect(self._set_status)
            worker.failed.connect(self.on_failed)
            worker.completed.connect(self.on_analysis_completed)
        if isinstance(worker, ProbeWorker):
            worker.probed.connect(self.on_probed)
        worker.finished.connect(self._release_worker)
        self.ingest_view.set_busy(True)
        self._set_status(message)
        worker.start()

    def _release_worker(self) -> None:
        self.ingest_view.set_busy(False)
        self.worker = None

    @staticmethod
    def _narrative_key(text: str) -> str:
        """A report's identity for duplicate detection: its words, normalised."""
        return " ".join(str(text or "").lower().split())

    def _narrative_index(self) -> Dict[str, int]:
        """Narrative -> row index, rebuilt whenever it has fallen out of step.

        The index is maintained as rows arrive, but anything that replaces
        ``self.rows`` wholesale - restoring a session, a test fixture - would
        otherwise leave duplicate detection silently switched off. Cheaper to
        notice and rebuild than to trust.
        """
        if len(self._by_narrative) != len(self.rows):
            self._by_narrative = {
                self._narrative_key(row.get("raw_text", "")): index
                for index, row in enumerate(self.rows)}
        return self._by_narrative

    def on_row_ready(self, payload: Dict[str, object]) -> None:
        """One analysed report arrived.

        The same narrative analysed twice is one incident, not two. Pressing
        "Load seed incidents" a second time, or re-importing a CSV after adding a
        column, used to append a whole second copy of every report - which
        double-counts the KPIs and, worse, inflates the hotspot densities that
        decide where an asset team spends its week. A repeat now replaces the row
        it repeats, keeping that row's reference and position.
        """
        key = self._narrative_key(payload.get("raw_text", ""))
        existing = self._narrative_index().get(key)
        if existing is not None and existing < len(self.rows):
            payload["reference"] = (self.rows[existing].get("reference")
                                    or payload.get("reference") or "")
            self.rows[existing] = payload
            self.duplicates_seen += 1
            self.report_view.table.set_rows(self.rows)
            return

        if not payload.get("reference"):
            # Never leave a report anonymous: an unreferenced row shows up in the
            # decision log as a bare hash, which nobody can look up afterwards.
            payload["reference"] = f"REP-{len(self.rows) + 1:04d}"
        self._by_narrative[key] = len(self.rows)
        self.rows.append(payload)
        self.report_view.table.append_row(payload)
        self.report_view.table.scrollToBottom()
        if len(self.rows) % 5 == 0:
            self._refresh()

    def on_analysis_completed(self, count: int) -> None:
        """A batch finished."""
        intelligence = self._refresh()
        self.audit.functionality(
            "reports analysed", count=count, corpus=len(self.rows),
            sif_potential=intelligence.kpis.get("sif_potential"),
            awaiting_review=self.outstanding_reviews,
            duplicates_replaced=self.duplicates_seen or None,
            encoder=intelligence.kpis.get("encoder") or "")
        LOGGER.info("Analysed %d report(s); %s SIF-potential, %s awaiting review", count,
                    intelligence.kpis.get("sif_potential"),
                    intelligence.kpis.get("needs_review"))
        repeats = (f"  ·  {self.duplicates_seen} repeat(s) replaced"
                   if self.duplicates_seen else "")
        self._set_status(
            f"Completed {count} report(s)  ·  {intelligence.kpis.get('sif_potential', 0)} "
            f"SIF-potential  ·  {self.outstanding_reviews} for review{repeats}")

    def on_document_ready(self, payload: Dict[str, object]) -> None:
        """One document was extracted."""
        text = str(payload.pop("text", ""))
        self.documents.append(payload)
        self.ingest_view.set_documents(self.documents)
        if text.strip():
            blocks = [block.strip() for block in text.split("\n\n")
                      if len(block.strip()) > 25] or [text.strip()]
            self.pending_blocks.extend(blocks)
            self.ingest_view.set_preview(text[:6000])
        self._set_status(f"{payload['name']} read via {payload['backend']}")
        self.audit.functionality("document read", name=payload.get("name"),
                                 backend=payload.get("backend"),
                                 pages=payload.get("pages"),
                                 characters=payload.get("characters"))
        self._refresh_workflow()

    def on_probed(self, name: str, ok: bool, message: str) -> None:
        """A connectivity probe finished."""
        if name == "OCR":
            self.ingest_view.set_ocr_status(message)
            self.engines_view.set_ocr_status(message)
        else:
            self.llm_online = ok
            self.llm_message = message
            self.engines_view.set_llm_status(message, self.llm.models() if ok else ())
        self._set_status(message[:120])
        self.audit.system(f"{name} check", ok=ok, outcome=message[:160])
        self._refresh_workflow()

    def on_trained(self, report: Dict[str, object]) -> None:
        """Training finished."""
        self.pipeline.attach_model(self.mlops)
        metrics = " ".join(f"{key}={value:.3f}"
                           for key, value in sorted(report.get("metrics", {}).items()))
        self._refresh_engines()
        self._refresh()
        message = (f"Trained on {report.get('samples')} report(s) "
                   f"({report.get('positives')} positive)  ·  {metrics}")
        self._set_status(message)
        self.audit.functionality(
            "model trained", samples=report.get("samples"),
            positives=report.get("positives"), labels=report.get("label_source"),
            metrics=metrics, run_id=report.get("run_id") or "not tracked")
        if report.get("warnings"):
            QMessageBox.information(self, APP_NAME,
                                    message + "\n\n" + "\n".join(report["warnings"]))

    def on_failed(self, message: str) -> None:
        """A worker reported an error."""
        LOGGER.error("%s", message)
        self._set_status("Task failed - see the log in Settings.")
        QMessageBox.warning(self, APP_NAME, message)

    def select_row(self, row: int) -> None:
        if 0 <= row < len(self.rows):
            self.report_view.show_detail(self.rows[row])

    # -- rendering ---------------------------------------------------------

    def _as_results(self) -> List[PipelineResult]:
        fields = PipelineResult.__dataclass_fields__
        return [PipelineResult(**{key: value for key, value in row.items() if key in fields})
                for row in self.rows]

    def _refresh(self):
        """Recompute aggregates and repaint every page."""
        results = self._as_results()
        intelligence = self.pipeline.aggregate(results)
        outstanding = self._refresh_review(results, intelligence)

        kpis = dict(intelligence.kpis)
        kpis["model_agreement"] = self._model_agreement(results)
        kpis["language"] = self.language
        # The header counts what is still owed to a person, not what was ever
        # queued: a decided report is finished work and stops being a number.
        kpis["needs_review"] = outstanding
        kpis["reviewed"] = self.decisions.counts()["decided"]

        rules, energies, barriers = self._chart_data(results)
        activities = self._activity_data(results)
        self.dashboard.update_kpis(kpis)
        self.dashboard.update_charts(rules, energies, barriers, activities)
        self.analytics_view.update_charts(rules, energies, barriers, activities)
        report = self.mlops.last_report
        self.analytics_view.update_model(self.mlops.status()["model"],
                                         report.importances if report else [])
        self.hotspot_view.set_rows([spot.to_dict() for spot in intelligence.hotspots])
        self._refresh_workflow()
        return intelligence

    # -- human review ------------------------------------------------------

    def _result_at(self, position: int) -> Optional[PipelineResult]:
        """The analysed result at a 1-based queue position, or None."""
        if not 1 <= position <= len(self.rows):
            return None
        fields = PipelineResult.__dataclass_fields__
        row = self.rows[position - 1]
        return PipelineResult(**{key: value for key, value in row.items()
                                 if key in fields})

    def _refresh_review(self, results: Sequence[PipelineResult], intelligence) -> int:
        """Repaint the review page; returns how many reports still need a person.

        A decided report leaves the queue but is never deleted: ticking "show
        reports already decided" brings it back, so a call can be checked or
        changed, and the trail tab always holds every entry.
        """
        standing = self.decisions.current()
        include_decided = self.review_view.show_decided.isChecked()
        rows: List[Dict[str, object]] = []
        outstanding = 0
        for item in intelligence.review_queue:
            result = results[item.index - 1] if item.index <= len(results) else None
            entry = standing.get(fingerprint(result)) if result is not None else None
            if entry is None:
                outstanding += 1
            elif not include_decided:
                continue
            payload = item.to_dict()
            payload["decided"] = entry is not None
            payload["status"] = ("awaiting" if entry is None
                                 else DECISION_SHORT.get(entry.decision, entry.decision))
            rows.append(payload)

        self.queue_rows = rows
        self.outstanding_reviews = outstanding
        counts = self.decisions.counts()
        self.review_view.set_queue(rows, outstanding, counts["decided"])
        self.review_view.set_trail(self.decisions.rows())
        self.sidebar.set_badge("review", outstanding)
        return outstanding

    def select_review_row(self, row: int) -> None:
        """Show the case for a queue row."""
        if not 0 <= row < len(self.queue_rows):
            self.review_view.set_case(None)
            return
        payload = self.queue_rows[row]
        position = int(payload.get("index", 0) or 0)
        result = self._result_at(position)
        if result is None:
            self.review_view.set_case(None)
            return
        entry = self.decisions.current().get(fingerprint(result))
        decision = None
        if entry is not None:
            decision = dict(entry.to_dict())
            decision["decision_label"] = DECISION_LABELS.get(entry.decision, entry.decision)
        self.review_view.set_case(self.rows[position - 1], decision)

    def record_decision(self, decision: str, note: str) -> None:
        """The reviewer called the selected report."""
        row = self.review_view.current_row()
        if not 0 <= row < len(self.queue_rows):
            return
        position = int(self.queue_rows[row].get("index", 0) or 0)
        result = self._result_at(position)
        if result is None:
            return
        reviewer = self.review_view.reviewer.text().strip()
        entry = self.decisions.record(result, decision, reviewer=reviewer, note=note)
        if not self.decisions.saved:
            QMessageBox.warning(
                self, APP_NAME,
                "The decision was recorded in this session but could not be written "
                f"to:\n{self.decisions.path}\n\nIt will be lost when the console "
                "closes. Check that the folder is writable.")
        self.audit.functionality(
            "review decision", reference=entry.reference or entry.fingerprint,
            decision=entry.decision, trigger=entry.trigger,
            overturns_engine=entry.overturns_engine, note=entry.note or None)
        self._refresh()
        self.review_view.advance()
        self._set_status(
            f"{DECISION_LABELS[entry.decision]} recorded for "
            f"{entry.reference or 'the selected report'}"
            + ("  ·  overturns the engine" if entry.overturns_engine else "")
            + f"  ·  {self.outstanding_reviews} left to review")

    def undo_decision(self) -> None:
        """Withdraw the most recent decision - for the misclick."""
        entry = self.decisions.undo()
        if entry is None:
            self._set_status("There is no decision to undo.")
            return
        self.audit.functionality("decision withdrawn",
                                 reference=entry.reference or entry.fingerprint,
                                 decision=entry.decision)
        self._refresh()
        self._set_status(
            f"Withdrew the {DECISION_LABELS[entry.decision].lower()} decision on "
            f"{entry.reference or 'a report'}")

    def export_decisions(self) -> None:
        """Write the decision trail out for an auditor."""
        if not self.decisions.entries:
            QMessageBox.information(self, APP_NAME,
                                    "There are no review decisions to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export review decisions", "review_decisions.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            self.decisions.export_csv(path)
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not write {path}:\n{exc}")
            return
        self.audit.functionality("decision trail exported",
                                 entries=len(self.decisions.entries), path=path)
        self._set_status(f"Exported {len(self.decisions.entries)} decision(s) to {path}")

    def set_reviewer(self, name: str) -> None:
        """Remember who is reviewing, so the name is not retyped every session."""
        prefs.set_value("reviewer", name)
        self.audit.reviewer = name
        self.audit.system("reviewer set", reviewer=name or "unnamed")
        LOGGER.info("Reviewer set to %s", name or "unnamed")

    @staticmethod
    def _model_agreement(results: Sequence[PipelineResult]) -> Optional[float]:
        scored = [item for item in results if item.ml_active]
        if not scored:
            return None
        return sum(1 for item in scored
                   if item.ml_flag == item.sif_potential) / len(scored) * 100.0

    @staticmethod
    def _chart_data(results: Sequence[PipelineResult]):
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
        total: Counter = Counter()
        flagged: Counter = Counter()
        for item in results:
            total[item.activity] += 1
            if item.sif_potential:
                flagged[item.activity] += 1
        return [(label, count, flagged.get(label, 0)) for label, count in total.most_common(12)]

    def _refresh_engines(self) -> None:
        status = self.mlops.status()
        self.engines_view.set_model_status(status["model"], status["tracking"])
        self.engines_view.set_runs(self.mlops.tracker.recent_runs(10))
        report = self.mlops.last_report
        self.engines_view.set_importances(report.importances if report else [])
        ocr_status = self.extractor.status()
        self.engines_view.set_ocr_status(ocr_status)
        self.ingest_view.set_ocr_status(ocr_status)
        self.settings_view.set_log_path(active_log_file() or log_file_path())

    def _set_audit_filter(self, category: str) -> None:
        self._audit_filter = category or ""
        self._refresh_audit()

    def _refresh_audit(self) -> None:
        """Repaint the audit trail from disk."""
        category = getattr(self, "_audit_filter", "")
        rows = self.audit.rows(category)
        counts = self.audit.counts()
        note = (f"{counts['total']} entr(ies): {counts['system']} system, "
                f"{counts['functionality']} functionality")
        if not self.audit.writable:
            note += "  ·  MEMORY ONLY - the trail could not be written to disk"
        self.settings_view.set_audit_rows(rows, note)

    def export_audit(self) -> None:
        """Write the audit trail out for an auditor."""
        if not self.audit.counts()["total"]:
            QMessageBox.information(self, APP_NAME, "The audit trail is empty.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export the audit trail", "sentra_audit.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            self.audit.export_csv(path, getattr(self, "_audit_filter", ""))
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not write {path}:\n{exc}")
            return
        self.audit.functionality("audit trail exported", path=path)
        self._set_status(f"Exported the audit trail to {path}")
        self._refresh_audit()

    def _refresh_logs(self) -> None:
        if self.pages.currentWidget() is not self.settings_view:
            return
        self._refresh_audit()
        level = self.settings_view.level_box.currentText()
        self.settings_view.set_log_rows(
            [{"timestamp": entry.timestamp, "level": entry.level,
              "logger": entry.logger, "message": entry.message}
             for entry in self.ring.entries(level, limit=400)])

    def _apply_filter(self, text: str) -> None:
        """Filter the result tables from the header search box.

        The box says "reports, sites, activities", so it searches the reports and
        the hotspots - typing a site name while looking at the hotspot page used
        to do nothing at all. The review queue is deliberately left alone: it is a
        work list, and hiding rows from it would hide work.
        """
        needle = (text or "").strip().lower()
        matched = self._filter_table(self.report_view.table, needle)
        self._filter_table(self.hotspot_view.table, needle)
        if needle:
            self._set_status(f"{matched} of {len(self.rows)} report(s) match {text.strip()!r}")
        elif self.rows:
            self._set_status(f"Showing all {len(self.rows)} report(s)")

    @staticmethod
    def _filter_table(table, needle: str) -> int:
        """Hide the rows that do not contain ``needle``; returns how many remain."""
        visible = 0
        for row in range(table.rowCount()):
            haystack = " ".join(
                table.item(row, column).text().lower()
                for column in range(table.columnCount()) if table.item(row, column))
            hidden = bool(needle) and needle not in haystack
            table.setRowHidden(row, hidden)
            visible += 0 if hidden else 1
        return visible

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    # -- Qt lifecycle ------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop timers and every running worker before closing.

        The update check runs a few seconds after start-up, so closing the window
        early used to leave its thread running and Qt would complain. Every worker
        this window owns is waited for, not just the analysis one.
        """
        self._closing = True
        self.audit.system("console closed", reports=len(self.rows),
                          outstanding=self.outstanding_reviews)
        self.log_timer.stop()
        for worker in (self.worker, self.update_worker):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(3000)
        LOGGER.info("%s (build 2) closing", APP_NAME)
        super().closeEvent(event)


def create_application(argv: Optional[List[str]] = None) -> QApplication:
    """Build a configured :class:`QApplication`."""
    app = QApplication(argv if argv is not None else [])
    app.setApplicationName(f"{APP_NAME} build 2")
    app.setOrganizationName("Oil India Limited")
    return app
