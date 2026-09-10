"""Pages for the second build: dashboard, ingestion, engines and settings.

Views are passive - they render what the controller gives them and emit signals.
No pictographs anywhere: statuses are words, separators are typographic marks.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui2.components import scrollable
from ui.charts import DonutChart, HBarChart
from ui.components import DataTable, FieldRow, KpiTile, Panel, Pill
from ui.theme import BAND_COLORS, C
from ui.views import (
    DOCUMENT_COLUMNS,
    IMPORTANCE_COLUMNS,
    LOG_COLUMNS,
    MATRIX_COLUMNS,
    RUN_COLUMNS,
)

__all__ = ["DashboardView", "IngestView", "EnginesView", "SettingsView",
           "ReportView", "scrollable", "AUDIT_COLUMNS"]

AUDIT_COLUMNS: Sequence[Tuple[str, str, int]] = (
    ("Time", "when", 158),
    ("Kind", "category", 116),
    ("What happened", "action", 210),
    ("Who", "actor", 110),
    ("Detail", "summary", 460),
)


class DashboardView(QWidget):
    """Analytics only - metrics and charts for the whole corpus."""

    MIN_CONTENT_HEIGHT = 760

    def __init__(self) -> None:
        super().__init__()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(14)

        self.tile_total = KpiTile("TOTAL REPORTS", "0", C.TEXT, note="analysed so far")
        self.tile_sif = KpiTile("SIF-POTENTIAL", "0", C.DANGER, note="0.0% of corpus")
        self.tile_risk = KpiTile("MEAN RISK SCORE", "0.0", C.ACCENT, unit="/ 100",
                                 note="ranked exposure")
        self.tile_review = KpiTile("AWAITING REVIEW", "0", C.WARN, note="expert validation")
        self.tile_engine = KpiTile("ENGINE AGREEMENT", "-", C.BLUE, note="model vs pipeline")

        kpis = QHBoxLayout()
        kpis.setSpacing(12)
        for tile in (self.tile_total, self.tile_sif, self.tile_risk, self.tile_review,
                     self.tile_engine):
            kpis.addWidget(tile)

        self.rule_chart = HBarChart(highlight_color=C.DANGER, base_color=C.BLUE)
        self.energy_chart = DonutChart(centre_caption="energy sources")
        self.barrier_chart = HBarChart(highlight_color=C.WARN, base_color=C.OK)
        self.activity_chart = HBarChart(highlight_color=C.PURPLE, base_color=C.BLUE)

        rules = Panel("SIF exposure by IOGP Life-Saving Rule")
        rules.add(self.rule_chart, stretch=1)
        rules.add(self._legend((C.DANGER, "SIF-potential"), (C.BLUE, "not SIF-potential")))
        energies = Panel("High-energy source distribution")
        energies.add(self.energy_chart, stretch=1)
        barriers = Panel("Failed barrier controls")
        barriers.add(self.barrier_chart, stretch=1)
        activities = Panel("Activities most often flagged")
        activities.add(self.activity_chart, stretch=1)

        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(rules, stretch=1)
        top.addWidget(energies, stretch=1)
        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        bottom.addWidget(barriers, stretch=1)
        bottom.addWidget(activities, stretch=1)

        self.summary = QLabel("No reports analysed yet.")
        self.summary.setObjectName("Faint")

        layout.addLayout(kpis)
        layout.addLayout(top, stretch=3)
        layout.addLayout(bottom, stretch=3)
        layout.addWidget(self.summary)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrollable(content, self.MIN_CONTENT_HEIGHT))

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

    def update_kpis(self, kpis: Dict[str, object]) -> None:
        total = int(kpis.get("total", 0))
        self.tile_total.set_value(str(total), "analysed so far")
        self.tile_sif.set_value(str(kpis.get("sif_potential", 0)),
                                f"{float(kpis.get('sif_rate', 0.0)):.1f}% of corpus")
        self.tile_risk.set_value(f"{float(kpis.get('mean_risk', 0.0)):.1f}",
                                 f"{kpis.get('critical', 0)} in the critical band")
        reviewed = int(kpis.get("reviewed", 0) or 0)
        self.tile_review.set_value(
            str(kpis.get("needs_review", 0)),
            f"{reviewed} decided by an expert" if reviewed else "expert validation")
        agreement = kpis.get("model_agreement")
        self.tile_engine.set_value(
            "-" if agreement is None else f"{float(agreement):.0f}%",
            "no model trained" if agreement is None else "model vs pipeline")
        self.summary.setText(
            f"{total} report(s)  ·  encoder: {kpis.get('encoder', 'not loaded')}"
            f"  ·  language: {kpis.get('language', 'English')}")

    def update_charts(self, rules, energies, barriers, activities) -> None:
        self.rule_chart.set_data(rules)
        self.energy_chart.set_data(energies)
        self.barrier_chart.set_data(barriers)
        self.activity_chart.set_data(activities)


class IngestView(QWidget):
    """Everything that gets a report into the system, on one page."""

    analyse_requested = pyqtSignal(str)
    seed_requested = pyqtSignal()
    csv_requested = pyqtSignal()
    files_requested = pyqtSignal()
    analyse_documents_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    language_changed = pyqtSignal(str)
    translate_toggled = pyqtSignal(bool)

    MIN_CONTENT_HEIGHT = 760

    def __init__(self, languages: Sequence[str], unsupported: Sequence[str]) -> None:
        super().__init__()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(self._build_text_panel(), stretch=3)
        top.addWidget(self._build_document_panel(languages, unsupported), stretch=4)

        self.document_table = DataTable(DOCUMENT_COLUMNS)
        documents = Panel("Extracted documents")
        documents.add(self.document_table, stretch=1)

        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Extracted text appears here.")
        preview_panel = Panel("Extracted text preview")
        preview_panel.add(self.preview, stretch=1)

        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        bottom.addWidget(documents, stretch=3)
        bottom.addWidget(preview_panel, stretch=2)

        layout.addLayout(top, stretch=3)
        layout.addLayout(bottom, stretch=3)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrollable(content, self.MIN_CONTENT_HEIGHT))

    def _build_text_panel(self) -> QWidget:
        panel = Panel("Report text")
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText(
            "Paste one UA/UC or near-miss narrative per blank-line-separated block.\n\n"
            "Reports in Hindi, Marathi, Tamil, Telugu, Kannada or Urdu are translated "
            "first when the local model is available.")
        self.input_box.setMinimumHeight(200)

        analyse = QPushButton("Analyse text")
        analyse.setObjectName("Primary")
        analyse.clicked.connect(
            lambda: self.analyse_requested.emit(self.input_box.toPlainText()))
        seed = QPushButton("Load 5 seed incidents")
        seed.clicked.connect(self.seed_requested.emit)
        csv_button = QPushButton("Import CSV export")
        csv_button.clicked.connect(self.csv_requested.emit)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)

        panel.add(self.input_box, stretch=1)
        panel.add(analyse)
        panel.add(seed)
        panel.add(csv_button)
        panel.add(self.progress)
        self.text_buttons = [analyse, seed, csv_button]
        return panel

    def _build_document_panel(self, languages, unsupported) -> QWidget:
        panel = Panel("Documents and OCR")

        note = QLabel(
            "PDFs with a text layer are read exactly; scans and photographs go "
            "through PaddleOCR in the selected language.")
        note.setObjectName("Muted")
        note.setWordWrap(True)

        language_caption = QLabel("OCR LANGUAGE")
        language_caption.setObjectName("Caption")
        self.language_box = QComboBox()
        self.language_box.addItems(list(languages))
        self.language_box.currentTextChanged.connect(self.language_changed.emit)

        self.translate_box = QCheckBox(
            "Translate non-English reports to English before analysis (needs Ollama)")
        self.translate_box.setChecked(True)
        self.translate_box.toggled.connect(self.translate_toggled.emit)

        self.ocr_status = QLabel("OCR status unknown")
        self.ocr_status.setObjectName("Muted")
        self.ocr_status.setWordWrap(True)

        unsupported_label = QLabel(
            "No recogniser in this PaddleOCR build: " + ", ".join(unsupported)
            + ". Those reports must be typed in or translated at source.")
        unsupported_label.setObjectName("Faint")
        unsupported_label.setWordWrap(True)

        add = QPushButton("Add documents (PDF, PNG, JPG, TIFF, TXT)")
        add.setObjectName("Primary")
        add.clicked.connect(self.files_requested.emit)
        run = QPushButton("Analyse extracted blocks")
        run.clicked.connect(self.analyse_documents_requested.emit)
        clear = QPushButton("Clear extraction list")
        clear.clicked.connect(self.clear_requested.emit)

        panel.add(note)
        panel.add(language_caption)
        panel.add(self.language_box)
        panel.add(self.translate_box)
        panel.add(self.ocr_status)
        panel.add(unsupported_label)
        panel.add(add)
        panel.add(run)
        panel.add(clear)
        self.document_buttons = [add, run, clear]
        return panel

    def set_busy(self, busy: bool) -> None:
        for button in self.text_buttons + self.document_buttons:
            button.setEnabled(not busy)
        self.progress.setVisible(busy)

    def set_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)

    def set_ocr_status(self, text: str) -> None:
        self.ocr_status.setText(text)

    def set_documents(self, rows) -> None:
        self.document_table.set_rows(rows)

    def set_preview(self, text: str) -> None:
        self.preview.setPlainText(text)


class ReportView(QWidget):
    """The incident matrix beside the full reasoning for the selected report."""

    row_selected = pyqtSignal(int)

    #: Below this the detail column scrolls rather than squeezing its rows.
    MIN_DETAIL_HEIGHT = 620

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        matrix = Panel("Parsed incident matrix")
        self.table = DataTable(MATRIX_COLUMNS, on_select=self.row_selected.emit)
        matrix.add(self.table, stretch=1)

        detail = Panel("Selected report")
        self.verdict = Pill("NO SELECTION", C.TEXT_DIM)
        self.risk = Pill("Risk: -", C.TEXT_DIM)
        pills = QHBoxLayout()
        pills.setSpacing(8)
        pills.addWidget(self.verdict)
        pills.addStretch(1)
        pills.addWidget(self.risk)

        self.reference = QLabel("-")
        self.reference.setObjectName("Muted")
        self.narrative = QTextEdit()
        self.narrative.setReadOnly(True)
        self.narrative.setFixedHeight(104)

        self.fields = {
            "rule": FieldRow("", "IOGP rule", "-", C.WARN),
            "energy": FieldRow("", "Energy source", "-", C.DANGER),
            "barrier": FieldRow("", "Failed barrier", "-", C.DANGER),
            "activity": FieldRow("", "Activity", "-", C.BLUE),
            "location": FieldRow("", "Location", "-", C.ACCENT),
            "language": FieldRow("", "Source language", "-", C.ACCENT),
            "model": FieldRow("", "Model P(SIF)", "-", C.PURPLE),
            "llm": FieldRow("", "Local LLM", "-", C.PURPLE),
        }

        self.evidence = QTextEdit()
        self.evidence.setReadOnly(True)
        self.evidence.setPlaceholderText("Evidence and reasoning appear here.")

        detail.body.addLayout(pills)
        detail.add(self.reference)
        detail.add(self.narrative)
        for field in self.fields.values():
            detail.add(field)
        caption = QLabel("EVIDENCE AND REASONING")
        caption.setObjectName("Caption")
        detail.add(caption)
        detail.add(self.evidence, stretch=1)

        # Eight stacked field rows plus the evidence box do not survive a short
        # window: without this the rows compress until the values are unreadable.
        layout.addWidget(matrix, stretch=5)
        layout.addWidget(scrollable(detail, self.MIN_DETAIL_HEIGHT), stretch=3)

    def show_detail(self, result: Optional[Dict[str, object]]) -> None:
        """Render one report, or clear the panel."""
        if not result:
            self.verdict.setText("NO SELECTION")
            self.verdict.set_colour(C.TEXT_DIM)
            self.risk.setText("Risk: -")
            self.risk.set_colour(C.TEXT_DIM)
            self.reference.setText("-")
            self.narrative.clear()
            self.evidence.clear()
            for field in self.fields.values():
                field.set_value("-")
            return

        is_sif = bool(result.get("sif_potential"))
        band = str(result.get("risk_band", "Low"))
        self.verdict.setText("SIF-POTENTIAL" if is_sif else "NOT SIF-POTENTIAL")
        self.verdict.set_colour(C.DANGER if is_sif else C.OK)
        self.risk.setText(f"Risk: {float(result.get('risk_score', 0.0)):.1f}")
        self.risk.set_colour(BAND_COLORS.get(band, C.OK))
        self.reference.setText(str(result.get("reference") or "unreferenced report"))
        # English is what was analysed and what a reviewer reads; the original
        # stays in the evidence panel below, which is the audit record.
        english = str(result.get("translated_text", ""))
        language = str(result.get("source_language", "")) or "another language"
        self.narrative.setPlainText(english or str(result.get("raw_text", "")))
        self.narrative.setToolTip(
            f"English rendering, translated from {language}. The original is under "
            "EVIDENCE AND REASONING." if english else "")

        self.fields["rule"].set_value(str(result.get("iogp_rule", "-")))
        self.fields["energy"].set_value(str(result.get("energy_source", "-")))
        self.fields["barrier"].set_value(str(result.get("barrier_failure", "-")))
        self.fields["activity"].set_value(str(result.get("activity", "-")))
        self.fields["location"].set_value(str(result.get("location", "-")))
        self.fields["language"].set_value(str(result.get("source_language") or "English"))
        probability = result.get("ml_probability")
        self.fields["model"].set_value(
            "-" if probability is None else f"{float(probability):.2f}")
        self.fields["llm"].set_value(
            f"{'SIF' if result.get('llm_flag') else 'not SIF'} - {result.get('llm_rule', '')}"
            if result.get("llm_active") else "not consulted")

        evidence = result.get("evidence", {}) or {}
        cues = "; ".join(evidence.get("lexical_cues", [])) or "none"
        semantic = ", ".join(
            f"{field} to {label} ({score:.2f})"
            for field, (label, score) in (evidence.get("semantic_matches", {}) or {}).items()
        ) or "none"
        risk = evidence.get("risk", {}) or {}
        llm = evidence.get("llm", {}) or {}
        original = str(result.get("raw_text", ""))
        translated = str(result.get("translated_text", ""))
        html = [
            f"<b>{result.get('explanation', '')}</b>",
            f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Decision path</p>"
            f"{evidence.get('decision_path', '')}",
            f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Risk</p>{risk.get('rationale', '')}",
            f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Lexical cues</p>{cues}",
            f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Nearest prototypes</p>{semantic}",
        ]
        if translated:
            html.append(f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Original as "
                        f"written ({result.get('source_language') or 'source language'})"
                        f"</p>{original}")
        if llm:
            detail = llm.get("rationale") or llm.get("error", "")
            html.append(f"<p style='color:{C.TEXT_DIM};margin:6px 0 0 0'>Local LLM "
                        f"({llm.get('model', 'n/a')})</p>{detail}")
        self.evidence.setHtml("".join(html))


class EnginesView(QWidget):
    """One page for the four engines, their state and their controls."""

    ollama_check_requested = pyqtSignal()
    ollama_config_changed = pyqtSignal(str, str)
    ollama_toggled = pyqtSignal(bool)
    encoder_changed = pyqtSignal(str)
    train_requested = pyqtSignal()
    ocr_check_requested = pyqtSignal()

    MIN_CONTENT_HEIGHT = 860

    def __init__(self) -> None:
        super().__init__()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        layout.addWidget(self._build_analysers())
        layout.addWidget(self._build_llm())
        layout.addWidget(self._build_model(), stretch=1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrollable(content, self.MIN_CONTENT_HEIGHT))

    def _build_analysers(self) -> QWidget:
        panel = Panel("Analysis engines")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        encoder_label = QLabel("Semantic encoder")
        encoder_label.setObjectName("Muted")
        self.encoder_box = QComboBox()
        self.encoder_box.addItem("Auto - transformer, fall back offline", "auto")
        self.encoder_box.addItem("Transformer (all-MiniLM-L6-v2)", "transformer")
        self.encoder_box.addItem("Offline - deterministic rules only", "hashing")
        self.encoder_box.currentIndexChanged.connect(
            lambda: self.encoder_changed.emit(self.encoder_box.currentData()))

        # The models are a one-time download per machine, so the button says so:
        # an operator who reads "check" every session assumes it is re-installing.
        ocr_button = QPushButton("Download / verify OCR models (once)")
        ocr_button.setToolTip(
            "Fetches the PaddleOCR models if this machine does not have them, then "
            "proves they load. They are kept on disk and are not downloaded again.")
        ocr_button.clicked.connect(self.ocr_check_requested.emit)

        grid.addWidget(encoder_label, 0, 0)
        grid.addWidget(self.encoder_box, 0, 1)
        grid.addWidget(ocr_button, 0, 2)
        grid.setColumnStretch(1, 1)

        self.encoder_status = QLabel("Encoder resolves on the next run.")
        self.encoder_status.setObjectName("Faint")
        self.encoder_status.setWordWrap(True)
        self.ocr_status = QLabel("OCR status unknown")
        self.ocr_status.setObjectName("Faint")
        self.ocr_status.setWordWrap(True)

        panel.body.addLayout(grid)
        panel.add(self.encoder_status)
        panel.add(self.ocr_status)
        return panel

    def _build_llm(self) -> QWidget:
        panel = Panel("Local LLM analyser (Ollama)")

        note = QLabel(
            "Optional. Runs on this machine, so no report leaves it. It gives a second "
            "reading of each narrative and translates non-English reports. It never "
            "overrides the pipeline: where it disagrees, the report goes to review.")
        note.setObjectName("Muted")
        note.setWordWrap(True)

        self.llm_enabled = QCheckBox("Use the local LLM as an additional analyser")
        self.llm_enabled.toggled.connect(self.ollama_toggled.emit)

        self.host_edit = QLineEdit("http://localhost:11434")
        self.model_edit = QLineEdit("llama3.2")
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(
            lambda: self.ollama_config_changed.emit(self.host_edit.text(),
                                                    self.model_edit.text()))
        check_button = QPushButton("Check connection")
        check_button.clicked.connect(self.ollama_check_requested.emit)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        host_label = QLabel("Host")
        host_label.setObjectName("Muted")
        model_label = QLabel("Model")
        model_label.setObjectName("Muted")
        grid.addWidget(host_label, 0, 0)
        grid.addWidget(self.host_edit, 0, 1)
        grid.addWidget(model_label, 0, 2)
        grid.addWidget(self.model_edit, 0, 3)
        grid.addWidget(apply_button, 0, 4)
        grid.addWidget(check_button, 0, 5)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(3, 1)

        self.llm_status = QLabel("Not checked yet.")
        self.llm_status.setObjectName("Muted")
        self.llm_status.setWordWrap(True)
        self.llm_models = QLabel("")
        self.llm_models.setObjectName("Faint")
        self.llm_models.setWordWrap(True)

        panel.add(note)
        panel.add(self.llm_enabled)
        panel.body.addLayout(grid)
        panel.add(self.llm_status)
        panel.add(self.llm_models)
        return panel

    def _build_model(self) -> QWidget:
        panel = Panel("Learned model and experiment tracking")
        self.train_button = QPushButton("Train XGBoost on the analysed corpus")
        self.train_button.setObjectName("Primary")
        self.train_button.clicked.connect(self.train_requested.emit)

        self.model_status = QLabel("Model status unknown")
        self.model_status.setObjectName("Muted")
        self.model_status.setWordWrap(True)
        self.tracking_status = QLabel("")
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

        panel.add(self.train_button)
        panel.add(self.model_status)
        panel.add(self.tracking_status)
        panel.body.addLayout(tables, stretch=1)
        return panel

    def set_llm_status(self, status: str, models: Sequence[str] = ()) -> None:
        self.llm_status.setText(status)
        self.llm_models.setText(
            "Models on this host: " + ", ".join(models) if models else "")

    def set_encoder_status(self, text: str) -> None:
        self.encoder_status.setText(text)

    def set_ocr_status(self, text: str) -> None:
        self.ocr_status.setText(text)

    def set_model_status(self, model: str, tracking: str) -> None:
        self.model_status.setText(f"Model: {model}")
        self.tracking_status.setText(tracking)

    def set_runs(self, rows) -> None:
        self.run_table.set_rows(rows)

    def set_importances(self, importances) -> None:
        self.importance_table.set_rows(
            [{"feature": name, "importance": f"{value:.4f}"} for name, value in importances])


class SettingsView(QWidget):
    """System logging and MLflow configuration."""

    log_level_changed = pyqtSignal(str)
    logs_cleared = pyqtSignal()
    logs_refreshed = pyqtSignal()
    tracking_changed = pyqtSignal(str, str)
    audit_refreshed = pyqtSignal()
    audit_exported = pyqtSignal()
    #: The chosen category, or "" for everything.
    audit_filtered = pyqtSignal(str)

    #: Below this the page scrolls instead of squeezing the log view away.
    MIN_CONTENT_HEIGHT = 640

    def __init__(self) -> None:
        super().__init__()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        tracking = Panel("MLflow tracking")
        self.tracking_uri = QLineEdit("sqlite:///mlflow.db")
        self.experiment_name = QLineEdit("sif-insight-console")
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(
            lambda: self.tracking_changed.emit(self.tracking_uri.text(),
                                               self.experiment_name.text()))
        row = QHBoxLayout()
        row.setSpacing(10)
        uri_label = QLabel("Tracking URI")
        uri_label.setObjectName("Muted")
        experiment_label = QLabel("Experiment")
        experiment_label.setObjectName("Muted")
        row.addWidget(uri_label)
        row.addWidget(self.tracking_uri, stretch=2)
        row.addWidget(experiment_label)
        row.addWidget(self.experiment_name, stretch=1)
        row.addWidget(apply_button)
        tracking.body.addLayout(row)

        logging_panel = Panel("System logging")
        self.level_box = QComboBox()
        self.level_box.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_box.setCurrentText("INFO")
        self.level_box.currentTextChanged.connect(self.log_level_changed.emit)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.logs_refreshed.emit)
        clear = QPushButton("Clear buffer")
        clear.clicked.connect(self.logs_cleared.emit)
        self.log_path = QLabel("")
        self.log_path.setObjectName("Faint")

        controls = QHBoxLayout()
        controls.setSpacing(10)
        level_label = QLabel("Minimum level")
        level_label.setObjectName("Muted")
        controls.addWidget(level_label)
        controls.addWidget(self.level_box)
        controls.addWidget(refresh)
        controls.addWidget(clear)
        controls.addStretch(1)
        controls.addWidget(self.log_path)

        self.log_table = DataTable(LOG_COLUMNS)
        logging_panel.body.addLayout(controls)
        logging_panel.add(self.log_table, stretch=1)

        layout.addWidget(tracking)
        layout.addWidget(logging_panel, stretch=1)
        layout.addWidget(self._build_audit_panel(), stretch=1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrollable(content, self.MIN_CONTENT_HEIGHT))

    def _build_audit_panel(self) -> Panel:
        """The trail: what the console did, and what it was asked to do.

        Separate from the log above it on purpose. The log is diagnostics and
        rotates away; this is the record an auditor reads, and it is append-only.
        """
        panel = Panel("Audit trail")
        caption = QLabel(
            "Append-only record of what happened on this machine. SYSTEM is what the "
            "software did by itself; FUNCTIONALITY is what an operator asked for and "
            "what came back. The debug log above rotates - this does not.")
        caption.setObjectName("Faint")
        caption.setWordWrap(True)

        self.audit_filter = QComboBox()
        self.audit_filter.addItem("Everything", "")
        self.audit_filter.addItem("System only", "system")
        self.audit_filter.addItem("Functionality only", "functionality")
        self.audit_filter.currentIndexChanged.connect(
            lambda: self.audit_filtered.emit(self.audit_filter.currentData()))

        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.audit_refreshed.emit)
        export = QPushButton("Export the trail as CSV")
        export.clicked.connect(self.audit_exported.emit)

        self.audit_summary = QLabel("No audit entries yet.")
        self.audit_summary.setObjectName("Faint")

        controls = QHBoxLayout()
        controls.setSpacing(10)
        show_label = QLabel("Show")
        show_label.setObjectName("Muted")
        controls.addWidget(show_label)
        controls.addWidget(self.audit_filter)
        controls.addWidget(refresh)
        controls.addWidget(export)
        controls.addStretch(1)
        controls.addWidget(self.audit_summary)

        self.audit_table = DataTable(AUDIT_COLUMNS)
        panel.add(caption)
        panel.body.addLayout(controls)
        panel.add(self.audit_table, stretch=1)
        return panel

    def set_log_rows(self, rows) -> None:
        self.log_table.set_rows(rows)
        self.log_table.scrollToBottom()

    def set_log_path(self, path: str) -> None:
        self.log_path.setText(f"Log file: {path}")

    def set_audit_rows(self, rows, note: str = "") -> None:
        """Render the audit trail, newest first, with a one-line summary."""
        self.audit_table.set_rows(rows)
        self.audit_summary.setText(note or f"{len(rows)} entr(ies)")
