"""The application's pages, one class per navigation entry.

Views are passive: they render whatever state the window hands them and emit
signals for anything that needs work done. All analysis, model training and file
reading happens on worker threads owned by :mod:`main`, never here.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .charts import DonutChart, HBarChart
from .components import DataTable, FieldRow, KpiTile, Panel, Pill
from .theme import BAND_COLORS, C

__all__ = ["MATRIX_COLUMNS", "HOTSPOT_COLUMNS", "REVIEW_COLUMNS", "DashboardView",
           "TableView", "BatchUploadView", "AnalyticsView", "SettingsView"]

MATRIX_COLUMNS: Sequence[Tuple[str, str, int]] = (
    ("#", "_index", 40),
    ("Trigger", "review_trigger", 118),
    ("Ref", "reference", 92),
    ("Risk", "risk_score", 58),
    ("SIF", "sif_potential", 52),
    ("Model", "ml_probability", 58),
    ("IOGP Rule", "iogp_rule", 160),
    ("Activity", "activity", 158),
    ("Location", "location", 158),
    ("Barrier Failure", "barrier_failure", 240),
    ("Energy Source", "energy_source", 180),
    ("Narrative", "raw_text", 320),
)

HOTSPOT_COLUMNS: Sequence[Tuple[str, str, int]] = (
    ("Type", "kind", 150),
    ("Cluster", "label", 330),
    ("Reports", "reports", 78),
    ("SIF", "sif_reports", 60),
    ("SIF %", "sif_rate", 68),
    ("Mean risk", "mean_risk", 86),
    ("Peak risk", "max_risk", 86),
    ("Dominant rule", "top_rule", 190),
    ("Dominant barrier", "top_barrier", 250),
)

REVIEW_COLUMNS: Sequence[Tuple[str, str, int]] = (
    ("Trigger", "trigger", 150),
    ("Ref", "reference", 92),
    ("Risk", "risk_score", 62),
    ("SIF", "sif_potential", 52),
    ("Why a human is needed", "reason", 400),
    ("Report", "summary", 440),
)

LOG_COLUMNS: Sequence[Tuple[str, str, int]] = (
    ("Time", "timestamp", 172),
    ("Level", "level", 84),
    ("Component", "logger", 190),
    ("Message", "message", 640),
)

RUN_COLUMNS: Sequence[Tuple[str, str, int]] = (
    ("Run", "run_id", 96),
    ("Started", "started", 160),
    ("Status", "status", 96),
    ("Samples", "samples", 82),
    ("Labels", "labels", 210),
    ("F1", "f1", 70),
    ("ROC AUC", "roc_auc", 84),
)

IMPORTANCE_COLUMNS: Sequence[Tuple[str, str, int]] = (
    ("Feature", "feature", 330),
    ("Importance", "importance", 110),
)

DOCUMENT_COLUMNS: Sequence[Tuple[str, str, int]] = (
    ("File", "name", 260),
    ("Backend", "backend", 150),
    ("Pages", "pages", 70),
    ("OCR confidence", "confidence", 120),
    ("Characters", "characters", 100),
    ("Blocks", "blocks", 80),
    ("Notes", "note", 420),
)


def _scrollable(widget: QWidget, minimum_height: int = 0) -> QScrollArea:
    """Wrap a page so it stays usable on a small screen.

    ``minimum_height`` is the point below which the page stops compressing and
    starts scrolling instead - without it a Qt layout keeps shrinking its
    children and the scroll bar never appears.
    """
    if minimum_height:
        widget.setMinimumHeight(minimum_height)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(widget)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return area


class DashboardView(QWidget):
    """Landing page: KPIs, the three charts, ingestion, matrix and detail."""

    analyse_requested = pyqtSignal(str)
    seed_requested = pyqtSignal()
    csv_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    encoder_changed = pyqtSignal(str)
    row_selected = pyqtSignal(int)
    review_requested = pyqtSignal()

    #: Below this the dashboard scrolls instead of squeezing its panels.
    MIN_CONTENT_HEIGHT = 860

    def __init__(self) -> None:
        super().__init__()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 12)
        layout.setSpacing(14)
        layout.addLayout(self._build_kpis())
        layout.addLayout(self._build_charts(), stretch=2)
        layout.addLayout(self._build_workspace(), stretch=5)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(content, self.MIN_CONTENT_HEIGHT))

    # -- construction ------------------------------------------------------

    def _build_kpis(self) -> QHBoxLayout:
        self.tile_total = KpiTile("TOTAL REPORTS", "0", C.TEXT, note="Processed this run",
                                  glyph="📄")
        self.tile_sif = KpiTile("SIF-POTENTIAL EVENTS", "0", C.DANGER, note="0.0% of corpus",
                                glyph="⚠")
        self.tile_risk = KpiTile("MEAN RISK SCORE", "0.0", C.ACCENT, unit="/ 100",
                                 note="Ranked exposure", glyph="◎")
        self.tile_review = KpiTile("AWAITING HUMAN REVIEW", "0", C.WARN,
                                   note="Need expert validation", glyph="👤")
        self.tile_model = KpiTile("MODEL AGREEMENT", "-", C.BLUE, note="No model trained",
                                  glyph="🧠")

        row = QHBoxLayout()
        row.setSpacing(12)
        for tile in (self.tile_total, self.tile_sif, self.tile_risk, self.tile_review,
                     self.tile_model):
            row.addWidget(tile)
        return row

    def _build_charts(self) -> QHBoxLayout:
        self.rule_chart = HBarChart(highlight_color=C.DANGER, base_color=C.BLUE)
        self.energy_chart = DonutChart(centre_caption="Energy sources")
        self.barrier_chart = HBarChart(highlight_color=C.WARN, base_color=C.OK)

        rules = Panel("SIF Exposure by IOGP Life-Saving Rule")
        rules.add(self.rule_chart, stretch=1)
        rules.add(self._legend((C.DANGER, "SIF-potential"), (C.BLUE, "Not SIF-potential")))

        energies = Panel("High-Energy Source Distribution")
        energies.add(self.energy_chart, stretch=1)

        barriers = Panel("Top Failed Barrier Controls")
        barriers.add(self.barrier_chart, stretch=1)

        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(rules, stretch=3)
        row.addWidget(energies, stretch=3)
        row.addWidget(barriers, stretch=3)
        return row

    @staticmethod
    def _legend(*entries: Tuple[str, str]) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addStretch(1)
        for colour, text in entries:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {colour}; font-size: 11px;")
            label = QLabel(text)
            label.setObjectName("Faint")
            layout.addWidget(dot)
            layout.addWidget(label)
        layout.addStretch(1)
        return widget

    def _build_workspace(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(self._build_ingestion(), stretch=2)
        row.addWidget(self._build_tabs(), stretch=5)
        row.addWidget(self._build_detail(), stretch=2)
        return row

    def _build_ingestion(self) -> QWidget:
        panel = Panel("Report Ingestion")

        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText(
            "Paste UA/UC or near-miss narrative here...\n\n"
            "e.g. Near miss at Pump Station No. 3, Duliajan: during preventive maintenance "
            "of the booster pump the 11 kV feeder cable was left ungrounded and the breaker "
            "was not racked out.")
        self.input_box.setMinimumHeight(120)

        encoder_caption = QLabel("SEMANTIC ENCODER")
        encoder_caption.setObjectName("Caption")
        self.encoder_box = QComboBox()
        self.encoder_box.addItem("Auto - transformer, fall back offline", "auto")
        self.encoder_box.addItem("Transformer (all-MiniLM-L6-v2)", "transformer")
        self.encoder_box.addItem("Offline - lexical rules only", "hashing")
        self.encoder_box.currentIndexChanged.connect(
            lambda: self.encoder_changed.emit(self.encoder_box.currentData()))

        analyse = QPushButton("Analyse Report  →")
        analyse.setObjectName("Primary")
        analyse.clicked.connect(
            lambda: self.analyse_requested.emit(self.input_box.toPlainText()))

        quick = QLabel("QUICK ACTIONS")
        quick.setObjectName("Caption")

        csv_button = QPushButton("⬆   Batch Import CSV")
        csv_button.clicked.connect(self.csv_requested.emit)
        seed_button = QPushButton("▤   Load 5 Seed Incidents")
        seed_button.clicked.connect(self.seed_requested.emit)
        clear_button = QPushButton("🗑   Clear Dashboard")
        clear_button.clicked.connect(self.clear_requested.emit)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)

        panel.add(self.input_box, stretch=1)
        panel.add(encoder_caption)
        panel.add(self.encoder_box)
        panel.add(analyse)
        panel.add(quick)
        for button in (csv_button, seed_button, clear_button):
            panel.add(button)
        panel.add(self.progress)
        self.buttons = [analyse, csv_button, seed_button, clear_button]
        return panel

    def _build_tabs(self) -> QWidget:
        panel = Panel()
        self.tabs = QTabWidget()
        compact = [(label, key, max(int(width * 0.78), 56))
                   for label, key, width in MATRIX_COLUMNS]
        self.matrix_table = DataTable(compact, on_select=self.row_selected.emit)
        self.hotspot_table = DataTable(HOTSPOT_COLUMNS)
        self.review_table = DataTable(REVIEW_COLUMNS)
        self.tabs.addTab(self.matrix_table, "▦  Parsed Incident Matrix")
        self.tabs.addTab(self.hotspot_table, "⌖  Risk Hotspots (0)")
        self.tabs.addTab(self.review_table, "👤  Human Review Queue (0)")

        self.count_label = QLabel("No reports analysed yet")
        self.count_label.setObjectName("Faint")

        footer = QHBoxLayout()
        footer.addWidget(self.count_label)
        footer.addStretch(1)

        panel.add(self.tabs, stretch=1)
        panel.body.addLayout(footer)
        return panel

    def _build_detail(self) -> QWidget:
        panel = Panel("Selected Report Analysis")

        self.detail_pill = Pill("NO SELECTION", C.TEXT_DIM)
        self.detail_risk = Pill("Risk: -", C.TEXT_DIM)
        pills = QHBoxLayout()
        pills.setSpacing(8)
        pills.addWidget(self.detail_pill)
        pills.addStretch(1)
        pills.addWidget(self.detail_risk)

        self.detail_reference = QLabel("-")
        self.detail_reference.setObjectName("Muted")

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFixedHeight(96)

        self.detail_fields = {
            "rule": FieldRow("⚠", "IOGP Rule", "-", C.WARN),
            "energy": FieldRow("⚡", "Energy Source", "-", C.DANGER),
            "barrier": FieldRow("⛔", "Failed Barrier", "-", C.DANGER),
            "activity": FieldRow("⚙", "Activity", "-", C.BLUE),
            "location": FieldRow("◎", "Location", "-", C.ACCENT),
            "model": FieldRow("🧠", "Model P(SIF)", "-", C.PURPLE),
        }

        self.detail_evidence = QTextEdit()
        self.detail_evidence.setReadOnly(True)
        self.detail_evidence.setPlaceholderText("Evidence and reasoning appear here.")

        review_button = QPushButton("👤  Mark for Review")
        review_button.setObjectName("Warning")
        review_button.clicked.connect(self.review_requested.emit)

        panel.body.addLayout(pills)
        panel.add(self.detail_reference)
        panel.add(self.detail_text)
        for field in self.detail_fields.values():
            panel.add(field)
        evidence_caption = QLabel("EVIDENCE & REASONING")
        evidence_caption.setObjectName("Caption")
        panel.add(evidence_caption)
        panel.add(self.detail_evidence, stretch=1)
        panel.add(review_button)
        return panel

    # -- rendering ---------------------------------------------------------

    def set_busy(self, busy: bool) -> None:
        """Disable the controls while a run is in flight."""
        for button in self.buttons:
            button.setEnabled(not busy)
        self.encoder_box.setEnabled(not busy)
        self.progress.setVisible(busy)

    def set_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)

    def update_charts(self, rules: Sequence[Tuple[str, float, float]],
                      energies: Sequence[Tuple[str, float]],
                      barriers: Sequence[Tuple[str, float, float]]) -> None:
        self.rule_chart.set_data(rules)
        self.energy_chart.set_data(energies)
        self.barrier_chart.set_data(barriers)

    def update_kpis(self, kpis: Dict[str, object]) -> None:
        total = int(kpis.get("total", 0))
        self.tile_total.set_value(str(total), f"{kpis.get('run_count', 0)} in the last run")
        self.tile_sif.set_value(str(kpis.get("sif_potential", 0)),
                                f"{float(kpis.get('sif_rate', 0.0)):.1f}% of corpus")
        self.tile_risk.set_value(f"{float(kpis.get('mean_risk', 0.0)):.1f}",
                                 f"{kpis.get('critical', 0)} in the critical band")
        self.tile_review.set_value(str(kpis.get("needs_review", 0)), "Need expert validation")
        agreement = kpis.get("model_agreement")
        if agreement is None:
            self.tile_model.set_value("-", "No model trained")
        else:
            self.tile_model.set_value(f"{float(agreement):.0f}%",
                                      "Model vs pipeline verdicts")
        self.count_label.setText(
            f"Showing {total} report(s)  ·  encoder: {kpis.get('encoder', 'not loaded')}")

    def update_tabs(self, hotspots: int, review: int) -> None:
        self.tabs.setTabText(1, f"⌖  Risk Hotspots ({hotspots})")
        self.tabs.setTabText(2, f"👤  Human Review Queue ({review})")

    def show_detail(self, result: Optional[Dict[str, object]]) -> None:
        """Render one report in the right-hand analysis panel."""
        if not result:
            self.detail_pill.setText("NO SELECTION")
            self.detail_pill.set_colour(C.TEXT_DIM)
            self.detail_risk.setText("Risk: -")
            self.detail_risk.set_colour(C.TEXT_DIM)
            self.detail_reference.setText("-")
            self.detail_text.clear()
            self.detail_evidence.clear()
            for field in self.detail_fields.values():
                field.set_value("-")
            return

        is_sif = bool(result.get("sif_potential"))
        band = str(result.get("risk_band", "Low"))
        self.detail_pill.setText("SIF-POTENTIAL" if is_sif else "NOT SIF-POTENTIAL")
        self.detail_pill.set_colour(C.DANGER if is_sif else C.OK)
        self.detail_risk.setText(f"Risk: {float(result.get('risk_score', 0.0)):.1f}")
        self.detail_risk.set_colour(BAND_COLORS.get(band, C.OK))
        self.detail_reference.setText(
            f"⛭  {result.get('reference') or 'unreferenced report'}")
        self.detail_text.setPlainText(str(result.get("raw_text", "")))

        self.detail_fields["rule"].set_value(str(result.get("iogp_rule", "-")))
        self.detail_fields["energy"].set_value(str(result.get("energy_source", "-")))
        self.detail_fields["barrier"].set_value(str(result.get("barrier_failure", "-")))
        self.detail_fields["activity"].set_value(str(result.get("activity", "-")))
        self.detail_fields["location"].set_value(str(result.get("location", "-")))
        probability = result.get("ml_probability")
        self.detail_fields["model"].set_value(
            "-" if probability is None else f"{float(probability):.2f}")

        evidence = result.get("evidence", {}) or {}
        cues = "; ".join(evidence.get("lexical_cues", [])) or "none"
        semantic = ", ".join(
            f"{field} → {label} ({score:.2f})"
            for field, (label, score) in (evidence.get("semantic_matches", {}) or {}).items()
        ) or "none"
        risk = evidence.get("risk", {}) or {}
        self.detail_evidence.setHtml(
            f"<b>{result.get('explanation', '')}</b>"
            f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Decision path</p>"
            f"{evidence.get('decision_path', '')}"
            f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Risk</p>"
            f"{risk.get('rationale', '')}"
            f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Lexical cues</p>{cues}"
            f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Nearest prototypes</p>{semantic}")


class TableView(QWidget):
    """A full-width page wrapping one table (matrix, hotspots or review)."""

    def __init__(self, title: str, subtitle: str,
                 columns: Sequence[Tuple[str, str, int]]) -> None:
        super().__init__()
        self.panel = Panel(title)
        self.table = DataTable(columns)
        caption = QLabel(subtitle)
        caption.setObjectName("Faint")
        caption.setWordWrap(True)
        self.panel.add(caption)
        self.panel.add(self.table, stretch=1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(self.panel)

    def set_rows(self, payloads: Sequence[Dict[str, object]]) -> None:
        self.table.set_rows(payloads)


class BatchUploadView(QWidget):
    """Document ingestion: CSV rows, PDFs and scanned images through OCR."""

    files_requested = pyqtSignal()
    csv_requested = pyqtSignal()
    analyse_requested = pyqtSignal()
    clear_requested = pyqtSignal()

    #: Below this the page scrolls instead of squeezing the preview away.
    MIN_CONTENT_HEIGHT = 720

    def __init__(self) -> None:
        super().__init__()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        actions = Panel("Document Ingestion")
        self.status_label = QLabel("OCR status unknown")
        self.status_label.setObjectName("Faint")
        self.status_label.setWordWrap(True)

        explain = QLabel(
            "Add PDFs, scans or photographs of shift logs and permits. Text-layer PDFs are "
            "read directly; pages without one go through PaddleOCR. Extracted text is split "
            "into report-sized blocks, then analysed by the pipeline.")
        explain.setObjectName("Muted")
        explain.setWordWrap(True)

        add_button = QPushButton("📎   Add Documents (PDF / PNG / JPG / TXT)")
        add_button.setObjectName("Primary")
        add_button.clicked.connect(self.files_requested.emit)
        csv_button = QPushButton("⬆   Import CSV of reports")
        csv_button.clicked.connect(self.csv_requested.emit)
        analyse_button = QPushButton("▶   Analyse Extracted Blocks")
        analyse_button.clicked.connect(self.analyse_requested.emit)
        clear_button = QPushButton("🗑   Clear Extraction List")
        clear_button.clicked.connect(self.clear_requested.emit)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        for button in (add_button, csv_button, analyse_button, clear_button):
            buttons.addWidget(button)
        buttons.addStretch(1)

        actions.add(explain)
        actions.add(self.status_label)
        actions.body.addLayout(buttons)

        self.documents = Panel("Extracted Documents")
        self.document_table = DataTable(DOCUMENT_COLUMNS)
        self.documents.add(self.document_table, stretch=1)

        self.preview = Panel("Extracted Text Preview")
        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setPlaceholderText("Extracted text appears here.")
        self.preview.add(self.preview_box, stretch=1)

        layout.addWidget(actions)
        layout.addWidget(self.documents, stretch=3)
        layout.addWidget(self.preview, stretch=2)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(content, self.MIN_CONTENT_HEIGHT))

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_documents(self, rows: Sequence[Dict[str, object]]) -> None:
        self.document_table.set_rows(rows)

    def set_preview(self, text: str) -> None:
        self.preview_box.setPlainText(text)


class AnalyticsView(QWidget):
    """Corpus analytics and the learned model's behaviour."""

    def __init__(self) -> None:
        super().__init__()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        self.rule_chart = HBarChart(highlight_color=C.DANGER, base_color=C.BLUE, max_rows=12)
        self.barrier_chart = HBarChart(highlight_color=C.WARN, base_color=C.OK, max_rows=12)
        self.energy_chart = DonutChart(centre_caption="Energy sources")
        self.activity_chart = HBarChart(highlight_color=C.PURPLE, base_color=C.BLUE,
                                        max_rows=12)

        top = QHBoxLayout()
        top.setSpacing(12)
        rules = Panel("Exposure by IOGP Rule")
        rules.add(self.rule_chart, stretch=1)
        barriers = Panel("Failed Barrier Controls")
        barriers.add(self.barrier_chart, stretch=1)
        top.addWidget(rules, stretch=1)
        top.addWidget(barriers, stretch=1)

        middle = QHBoxLayout()
        middle.setSpacing(12)
        energies = Panel("High-Energy Sources")
        energies.add(self.energy_chart, stretch=1)
        activities = Panel("Activities Most Often Flagged")
        activities.add(self.activity_chart, stretch=1)
        middle.addWidget(energies, stretch=1)
        middle.addWidget(activities, stretch=1)

        self.model_panel = Panel("Learned Model")
        self.model_summary = QLabel("No model trained yet.")
        self.model_summary.setObjectName("Muted")
        self.model_summary.setWordWrap(True)
        self.importance_table = DataTable(IMPORTANCE_COLUMNS)
        self.model_panel.add(self.model_summary)
        self.model_panel.add(self.importance_table, stretch=1)

        layout.addLayout(top, stretch=3)
        layout.addLayout(middle, stretch=3)
        layout.addWidget(self.model_panel, stretch=3)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(content, 900))

    def update_charts(self, rules, energies, barriers, activities) -> None:
        self.rule_chart.set_data(rules)
        self.energy_chart.set_data(energies)
        self.barrier_chart.set_data(barriers)
        self.activity_chart.set_data(activities)

    def update_model(self, summary: str, importances: Sequence[Tuple[str, float]]) -> None:
        self.model_summary.setText(summary)
        self.importance_table.set_rows(
            [{"feature": name, "importance": f"{value:.4f}"} for name, value in importances])


class SettingsView(QWidget):
    """System logging, MLOps controls and ingestion configuration."""

    log_level_changed = pyqtSignal(str)
    logs_cleared = pyqtSignal()
    logs_refreshed = pyqtSignal()
    train_requested = pyqtSignal()
    tracking_changed = pyqtSignal(str, str)
    ocr_toggled = pyqtSignal(bool, str)
    ocr_test_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        layout.addWidget(self._build_logging(), stretch=3)
        layout.addWidget(self._build_mlops(), stretch=4)
        layout.addWidget(self._build_ingestion(), stretch=2)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(content, 980))

    # -- sections ----------------------------------------------------------

    def _build_logging(self) -> QWidget:
        panel = Panel("System Logging")

        self.level_box = QComboBox()
        self.level_box.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_box.setCurrentText("INFO")
        self.level_box.currentTextChanged.connect(self.log_level_changed.emit)

        refresh = QPushButton("↻  Refresh")
        refresh.clicked.connect(self.logs_refreshed.emit)
        clear = QPushButton("🗑  Clear")
        clear.clicked.connect(self.logs_cleared.emit)

        self.log_path_label = QLabel("")
        self.log_path_label.setObjectName("Faint")

        controls = QHBoxLayout()
        controls.setSpacing(10)
        level_caption = QLabel("Minimum level")
        level_caption.setObjectName("Muted")
        controls.addWidget(level_caption)
        controls.addWidget(self.level_box)
        controls.addWidget(refresh)
        controls.addWidget(clear)
        controls.addStretch(1)
        controls.addWidget(self.log_path_label)

        self.log_table = DataTable(LOG_COLUMNS)
        panel.body.addLayout(controls)
        panel.add(self.log_table, stretch=1)
        return panel

    def _build_mlops(self) -> QWidget:
        panel = Panel("MLOps - XGBoost model and MLflow tracking")

        self.tracking_uri = QLineEdit("sqlite:///mlflow.db")
        self.experiment_name = QLineEdit("sif-insight-console")
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(
            lambda: self.tracking_changed.emit(self.tracking_uri.text(),
                                               self.experiment_name.text()))
        self.train_button = QPushButton("⚙  Train XGBoost on analysed corpus")
        self.train_button.setObjectName("Primary")
        self.train_button.clicked.connect(self.train_requested.emit)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        uri_label = QLabel("MLflow tracking URI")
        uri_label.setObjectName("Muted")
        experiment_label = QLabel("Experiment")
        experiment_label.setObjectName("Muted")
        form.addWidget(uri_label, 0, 0)
        form.addWidget(self.tracking_uri, 0, 1)
        form.addWidget(experiment_label, 0, 2)
        form.addWidget(self.experiment_name, 0, 3)
        form.addWidget(apply_button, 0, 4)
        form.addWidget(self.train_button, 1, 0, 1, 5)

        self.model_status = QLabel("Model status unknown")
        self.model_status.setObjectName("Muted")
        self.model_status.setWordWrap(True)
        self.tracking_status = QLabel("Tracking status unknown")
        self.tracking_status.setObjectName("Faint")
        self.tracking_status.setWordWrap(True)

        tables = QHBoxLayout()
        tables.setSpacing(12)
        runs = Panel("Recent training runs")
        self.run_table = DataTable(RUN_COLUMNS)
        runs.add(self.run_table, stretch=1)
        importances = Panel("Feature importance")
        self.importance_table = DataTable(IMPORTANCE_COLUMNS)
        importances.add(self.importance_table, stretch=1)
        tables.addWidget(runs, stretch=3)
        tables.addWidget(importances, stretch=2)

        panel.body.addLayout(form)
        panel.add(self.model_status)
        panel.add(self.tracking_status)
        panel.body.addLayout(tables, stretch=1)
        return panel

    def _build_ingestion(self) -> QWidget:
        panel = Panel("Document Ingestion - PaddleOCR")

        self.ocr_enabled = QCheckBox("Use OCR for scanned pages and images")
        self.ocr_enabled.setChecked(True)
        self.ocr_language = QComboBox()
        self.ocr_language.addItems(["en", "hi", "ch", "fr", "de", "es", "ar"])
        self.ocr_enabled.toggled.connect(
            lambda checked: self.ocr_toggled.emit(checked, self.ocr_language.currentText()))
        self.ocr_language.currentTextChanged.connect(
            lambda language: self.ocr_toggled.emit(self.ocr_enabled.isChecked(), language))

        test_button = QPushButton("Check OCR availability")
        test_button.clicked.connect(self.ocr_test_requested.emit)

        self.ocr_status = QLabel("OCR status unknown")
        self.ocr_status.setObjectName("Muted")
        self.ocr_status.setWordWrap(True)

        row = QHBoxLayout()
        row.setSpacing(10)
        language_label = QLabel("Language")
        language_label.setObjectName("Muted")
        row.addWidget(self.ocr_enabled)
        row.addWidget(language_label)
        row.addWidget(self.ocr_language)
        row.addWidget(test_button)
        row.addStretch(1)

        panel.body.addLayout(row)
        panel.add(self.ocr_status)
        return panel

    # -- rendering ---------------------------------------------------------

    def set_log_rows(self, rows: Sequence[Dict[str, object]]) -> None:
        self.log_table.set_rows(rows)
        self.log_table.scrollToBottom()

    def set_log_path(self, path: str) -> None:
        self.log_path_label.setText(f"Log file: {path}")

    def set_model_status(self, model: str, tracking: str) -> None:
        self.model_status.setText(f"Model: {model}")
        self.tracking_status.setText(tracking)

    def set_runs(self, rows: Sequence[Dict[str, object]]) -> None:
        self.run_table.set_rows(rows)

    def set_importances(self, importances: Sequence[Tuple[str, float]]) -> None:
        self.importance_table.set_rows(
            [{"feature": name, "importance": f"{value:.4f}"} for name, value in importances])

    def set_ocr_status(self, text: str) -> None:
        self.ocr_status.setText(text)
