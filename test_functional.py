"""End-to-end functional tests: the console driven the way an operator drives it.

The unit suites prove each part in isolation. This one proves the parts still
work *together*, by running the real windows against the real sample files in
`samples/` and checking what an operator would look at afterwards - the KPI
tiles, the tables, the queue, the exports.

What is deliberately not mocked: the pipeline, the workers, the aggregation, the
review log, the CSV importer and the document extractor. What is: the network
(never touched), file dialogs (paths are passed straight in) and modal message
boxes (recorded instead of shown, because a modal dialog in a headless run hangs
the suite rather than failing it).

Runs under both ``python -m unittest test_functional`` and ``python -m pytest
test_functional.py``. The GUI classes skip themselves when PyQt6 is absent, and
the learned-model class skips when xgboost is absent, so a partial install still
gets a useful answer instead of an error.
"""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from typing import List

from sif.ocr import DocumentExtractor
from sif.pipeline import SIFPipeline
from sif.review import DecisionLog, ReviewQueue

try:  # pragma: no cover - import guard, exercised by the skip decorators
    from PyQt6.QtWidgets import QApplication

    HAS_PYQT = True
except Exception:  # noqa: BLE001
    HAS_PYQT = False

try:  # pragma: no cover - the learned layer is optional by design
    import importlib.util

    HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None
except Exception:  # noqa: BLE001
    HAS_XGBOOST = False

SAMPLES = "samples"
CSV_SAMPLE = os.path.join(SAMPLES, "near_miss_reports.csv")


def _application():
    """One offscreen QApplication for the whole run."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class TestIngestionPathsEndToEnd(unittest.TestCase):
    """Every way a report gets into the system, against the real sample files."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = SIFPipeline(backend="hashing")
        cls.extractor = DocumentExtractor()

    def test_the_csv_becomes_analysed_reports(self) -> None:
        from main import read_csv_reports

        narratives, references = read_csv_reports(CSV_SAMPLE)
        results = [self.pipeline.analyze(text, reference=reference)
                   for text, reference in zip(narratives, references)]
        self.assertEqual(len(results), 18)
        self.assertTrue(all(result.iogp_rule for result in results))
        self.assertTrue(all(result.explanation for result in results))
        self.assertTrue(all(0.0 <= result.risk_score <= 100.0 for result in results))
        self.assertGreaterEqual(sum(1 for r in results if r.sif_potential), 4)

    def test_a_text_document_splits_into_report_sized_blocks(self) -> None:
        document = self.extractor.extract(os.path.join(SAMPLES, "shift_log.txt"))
        self.assertEqual(document.backend, "text")
        blocks = document.blocks()
        self.assertGreaterEqual(len(blocks), 5)
        results = [self.pipeline.analyze(block, reference=f"LOG-{index:02d}")
                   for index, block in enumerate(blocks[1:], start=1)]
        self.assertTrue(all(result.iogp_rule for result in results))

    def test_a_pdf_is_read_through_its_text_layer_without_ocr(self) -> None:
        document = self.extractor.extract(os.path.join(SAMPLES, "permit_observation.pdf"))
        self.assertEqual(document.backend, "pdf-text")
        self.assertGreater(len(document.text), 400)
        result = self.pipeline.analyze(document.text, reference="PTW-0142")
        self.assertTrue(result.sif_potential)
        self.assertGreater(result.risk_score, 50)

    def test_a_non_english_report_is_carried_through_not_dropped(self) -> None:
        """Without a translator the text still analyses; it is never discarded."""
        document = self.extractor.extract(os.path.join(SAMPLES, "multilingual_report.txt"))
        blocks = [block for block in document.blocks() if len(block) > 60]
        self.assertGreaterEqual(len(blocks), 3)
        for block in blocks:
            result = self.pipeline.analyze(block, reference="ML")
            self.assertTrue(result.iogp_rule, "every report gets a verdict, even untranslated")

    def test_corpus_aggregation_produces_the_dashboard_numbers(self) -> None:
        from main import read_csv_reports

        narratives, references = read_csv_reports(CSV_SAMPLE)
        results = [self.pipeline.analyze(text, reference=reference)
                   for text, reference in zip(narratives, references)]
        intelligence = self.pipeline.aggregate(results)
        kpis = intelligence.kpis
        self.assertEqual(kpis["total"], 18)
        self.assertEqual(kpis["sif_potential"],
                         sum(1 for result in results if result.sif_potential))
        self.assertGreater(kpis["mean_risk"], 0)
        self.assertGreaterEqual(len(intelligence.hotspots), 1)
        # The queue is exactly the reports that met a trigger. The count moves as
        # the knowledge base improves; the identity does not.
        self.assertEqual(len(intelligence.review_queue),
                         sum(1 for result in results if result.needs_review))
        self.assertGreater(len(intelligence.review_queue), 0)


class TestReviewToLabelsEndToEnd(unittest.TestCase):
    """The loop the whole product turns on: queue, decide, train on the decisions."""

    def setUp(self) -> None:
        from main import read_csv_reports

        self.folder = tempfile.mkdtemp(prefix="sif-functional-")
        self.pipeline = SIFPipeline(backend="hashing")
        narratives, references = read_csv_reports(CSV_SAMPLE)
        self.results = [self.pipeline.analyze(text, reference=reference)
                        for text, reference in zip(narratives, references)]
        self.log = DecisionLog(os.path.join(self.folder, "decisions.json"))

    def test_working_the_queue_empties_it_and_leaves_a_trail(self) -> None:
        queue = ReviewQueue()
        outstanding = queue.build(self.results, skip=self.log.decided())
        self.assertEqual(len(outstanding),
                         sum(1 for result in self.results if result.needs_review))
        self.assertGreater(len(outstanding), 0)

        by_reference = {result.reference: result for result in self.results}
        for position, item in enumerate(outstanding):
            result = by_reference[item.reference]
            decision = ("confirmed" if result.sif_potential else
                        "unclear" if position % 7 == 0 else "rejected")
            self.log.record(result, decision, reviewer="Functional test")

        self.assertEqual(queue.build(self.results, skip=self.log.decided()), [])
        counts = self.log.counts()
        self.assertEqual(counts["decided"], len(outstanding))
        self.assertEqual(counts["labels"], counts["decided"] - counts["unclear"])
        self.assertEqual(len(DecisionLog(self.log.path).load().entries), len(outstanding))

    def test_the_trail_exports_for_an_auditor(self) -> None:
        self.log.record(self.results[0], "confirmed", reviewer="A", note="verified on site")
        self.log.record(self.results[1], "rejected", reviewer="B")
        path = self.log.export_csv(os.path.join(self.folder, "trail.csv"))
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["reviewer"] for row in rows}, {"A", "B"})

    @unittest.skipUnless(HAS_XGBOOST, "xgboost is not installed")
    def test_the_model_trains_on_the_decisions_a_person_made(self) -> None:
        from sif.mlops import MLOpsService

        for result in self.results:
            self.log.record(result, "confirmed" if result.sif_potential else "rejected")
        reviewed, labels = self.log.labels_for(self.results)
        self.assertEqual(len(labels), len(self.results))
        self.assertEqual(set(labels), {0, 1})

        service = MLOpsService(model_directory=os.path.join(self.folder, "models"),
                               tracking_uri="", experiment="functional")
        service.tracker.tracking_uri = ""          # no MLflow store in a test run
        service.tracker.installed = staticmethod(lambda: False)
        report = service.train(reviewed, labels=labels, label_source="human review decisions")
        self.assertEqual(report.label_source, "human review decisions")
        self.assertEqual(report.samples, len(labels))
        self.assertTrue(os.path.isfile(report.model_path))

        probability = service.predict(self.results[0])
        self.assertIsNotNone(probability)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestBuildTwoWindowEndToEnd(unittest.TestCase):
    """The build-2 console, driven stage by stage down its own workflow map."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _application()

    def setUp(self) -> None:
        import main2

        self.main2 = main2
        self.dialogs: List[str] = []
        self._boxes = (main2.QMessageBox.information, main2.QMessageBox.warning)
        main2.QMessageBox.information = lambda *args, **kw: self.dialogs.append(args[-1])
        main2.QMessageBox.warning = lambda *args, **kw: self.dialogs.append(args[-1])

        self.folder = tempfile.mkdtemp(prefix="sif-window-")
        self.window = main2.MainWindow()
        self.window.decisions = DecisionLog(os.path.join(self.folder, "decisions.json"))

    def tearDown(self) -> None:
        self.main2.QMessageBox.information, self.main2.QMessageBox.warning = self._boxes
        self.window.close()

    def _analyse_samples(self) -> None:
        from main import read_csv_reports

        narratives, references = read_csv_reports(CSV_SAMPLE)
        self.window._start(self.window._analysis_worker(texts=narratives,
                                                        references=references))
        self.assertTrue(self.window.worker.wait(180_000), "analysis did not finish")
        self.app.processEvents()

    def test_the_whole_console_fills_in_from_one_import(self) -> None:
        self._analyse_samples()

        # Reports and evidence
        self.assertEqual(len(self.window.rows), 18)
        self.assertEqual(self.window.report_view.table.rowCount(), 18)
        self.window.select_row(0)
        self.assertTrue(self.window.report_view.evidence.toPlainText())

        # Dashboard
        self.assertEqual(self.window.dashboard.tile_total._value.text(), "18")
        self.assertNotEqual(self.window.dashboard.tile_risk._value.text(), "0.0")

        # Hotspots and review
        self.assertGreaterEqual(self.window.hotspot_view.table.rowCount(), 1)
        self.assertGreater(self.window.outstanding_reviews, 0)
        self.assertEqual(self.window.review_view.table.rowCount(),
                         self.window.outstanding_reviews)

        # Every page renders without raising
        for key in ("workflow", "ingest", "dashboard", "reports", "hotspots",
                    "review", "analytics", "engines", "settings"):
            self.window.navigate(key)
            self.app.processEvents()

    def test_the_workflow_map_reports_what_has_actually_happened(self) -> None:
        self._analyse_samples()
        cards = self.window.workflow.cards
        self.assertIn("18", cards["analyse"].status.text())
        self.assertIn(str(self.window.outstanding_reviews), cards["review"].status.text())
        self.assertIn("report(s)", cards["dashboard"].status.text())

    def test_reviewing_a_report_moves_it_out_of_the_queue_and_into_the_trail(self) -> None:
        self._analyse_samples()
        self.window.navigate("review")
        view = self.window.review_view
        view.reviewer.setText("Functional test")
        before = self.window.outstanding_reviews
        view.select(0)
        first = view.reference.text()
        view._decide("confirmed")

        self.assertEqual(self.window.outstanding_reviews, before - 1)
        self.assertEqual(view.trail_table.rowCount(), 1)
        self.assertNotEqual(view.reference.text(), first, "the bench must advance")
        self.assertEqual(self.window.dashboard.tile_review._value.text(), str(before - 1))

    def _visible(self, table) -> int:
        """Rows the operator can actually see - filtering hides, it does not delete."""
        return sum(1 for row in range(table.rowCount()) if not table.isRowHidden(row))

    def test_the_search_box_filters_reports_and_hotspots_alike(self) -> None:
        self._analyse_samples()
        matrix = self.window.report_view.table
        hotspots = self.window.hotspot_view.table

        # "permit" matches several reports, in the narrative or in an extracted
        # field such as the failed barrier. Counted from the rows rather than
        # hard-coded, because what the engine extracts changes as it improves;
        # what must hold is that filtering shows exactly the matching rows.
        self.window._apply_filter("permit")
        expected = sum(1 for row in self.window.rows
                       if "permit" in " ".join(str(value) for value in row.values()).lower())
        self.assertEqual(self._visible(matrix), expected)
        self.assertGreater(expected, 0)
        self.assertLess(self._visible(matrix), 18)
        self.assertIn("match", self.window.status_label.text())
        if hotspots.rowCount():
            self.assertLessEqual(self._visible(hotspots), hotspots.rowCount())

        self.window._apply_filter("")
        self.assertEqual(self._visible(matrix), 18)
        self.assertEqual(self._visible(hotspots), hotspots.rowCount())
        self.assertEqual(len(self.window.rows), 18, "filtering must not drop reports")

    def test_a_search_that_matches_nothing_says_so_rather_than_looking_broken(self) -> None:
        self._analyse_samples()
        self.window._apply_filter("zzzz-no-such-site")
        self.assertEqual(self._visible(self.window.report_view.table), 0)
        self.assertIn("0 of 18", self.window.status_label.text())

    def test_a_closed_window_does_not_start_an_update_check(self) -> None:
        """The start-up check fires on a timer; the window may be gone by then."""
        self.window.close()
        self.window.check_for_updates(False)
        self.assertIsNone(self.window.update_worker)

    def test_documents_are_read_and_queued_for_analysis(self) -> None:
        """Drives Add documents itself, with the file chooser answered for it."""
        paths = [os.path.join(SAMPLES, "shift_log.txt"),
                 os.path.join(SAMPLES, "permit_observation.pdf")]
        original = self.main2.QFileDialog.getOpenFileNames
        self.main2.QFileDialog.getOpenFileNames = lambda *args, **kw: (paths, "")
        try:
            self.window.add_documents()
        finally:
            self.main2.QFileDialog.getOpenFileNames = original
        self.assertTrue(self.window.worker.wait(120_000))
        self.app.processEvents()

        self.assertEqual(len(self.window.documents), 2)
        self.assertGreaterEqual(len(self.window.pending_blocks), 6)
        backends = {document["backend"] for document in self.window.documents}
        self.assertEqual(backends, {"text", "pdf-text"})

    def test_importing_a_csv_goes_through_the_real_import_path(self) -> None:
        original = self.main2.QFileDialog.getOpenFileName
        self.main2.QFileDialog.getOpenFileName = lambda *args, **kw: (CSV_SAMPLE, "")
        try:
            self.window.import_csv()
        finally:
            self.main2.QFileDialog.getOpenFileName = original
        self.assertTrue(self.window.worker.wait(180_000))
        self.app.processEvents()
        self.assertEqual(len(self.window.rows), 18)
        self.assertEqual(self.window.rows[0]["reference"], "NM-2601")

    def test_exporting_the_corpus_writes_every_row(self) -> None:
        self._analyse_samples()
        path = os.path.join(self.folder, "export.csv")
        original = self.main2.QFileDialog.getSaveFileName
        self.main2.QFileDialog.getSaveFileName = lambda *args, **kw: (path, "")
        try:
            self.window.export_csv()
        finally:
            self.main2.QFileDialog.getSaveFileName = original

        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 18)
        self.assertIn("iogp_rule", rows[0])
        self.assertIn("sif_potential", rows[0])
        self.assertEqual(rows[0]["reference"], "NM-2601")

    def test_generating_a_bulletin_writes_a_traceable_document(self) -> None:
        """The auto-written bulletin over the real corpus, exercised end to end."""
        self._analyse_samples()
        path = os.path.join(self.folder, "bulletin.txt")
        original = self.main2.QFileDialog.getSaveFileName
        self.main2.QFileDialog.getSaveFileName = lambda *args, **kw: (path, "")
        try:
            self.window.generate_bulletin()
        finally:
            self.main2.QFileDialog.getSaveFileName = original

        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("Safety Intelligence Bulletin", text)
        self.assertIn("18 report(s) analysed", text)
        self.assertIn("WHAT IS DRIVING RISK", text)
        self.assertIn("NEEDS ATTENTION NOW", text)
        # Every reference the bulletin names must be a report that was actually
        # analysed - a generated document is worthless if it cites nothing real.
        references = {row["reference"] for row in self.window.rows}
        cited = {line.split()[1] for line in text.splitlines()
                 if line.startswith("- ") and line.split()[1] in references}
        self.assertTrue(cited, "the bulletin named no real report references")
        self.assertTrue(cited.issubset(references))
        self.assertIn("Prototype output", text)

    def test_a_corpus_imported_twice_stays_one_corpus(self) -> None:
        """The whole CSV, twice - the second pass must replace, not duplicate."""
        self._analyse_samples()
        self.assertEqual(len(self.window.rows), 18)
        first_references = [row["reference"] for row in self.window.rows]

        self._analyse_samples()
        self.assertEqual(len(self.window.rows), 18, "the corpus doubled")
        self.assertEqual(self.window.duplicates_seen, 18)
        self.assertEqual([row["reference"] for row in self.window.rows], first_references)
        self.assertEqual(self.window.dashboard.tile_total._value.text(), "18")

    def test_the_audit_trail_carries_the_session_end_to_end(self) -> None:
        from sif.audit import FUNCTIONALITY, SYSTEM, AuditLog

        self.window.audit = AuditLog(os.path.join(self.folder, "audit.jsonl"))
        self._analyse_samples()
        self.window.navigate("review")
        self.window.review_view.select(0)
        self.window.review_view._decide("confirmed")

        actions = [entry.action for entry in self.window.audit.entries()]
        self.assertIn("reports analysed", actions)
        self.assertIn("review decision", actions)
        analysed = next(entry for entry in self.window.audit.entries()
                        if entry.action == "reports analysed")
        self.assertEqual(analysed.category, FUNCTIONALITY)
        self.assertEqual(analysed.detail.get("count"), 18)
        self.assertEqual(self.window.audit.counts()[FUNCTIONALITY], len(actions)
                         - sum(1 for entry in self.window.audit.entries()
                               if entry.category == SYSTEM))

        path = self.window.audit.export_csv(os.path.join(self.folder, "audit.csv"))
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(actions))

    def test_every_language_sample_is_read_and_analysed(self) -> None:
        """Tamil, Hindi and the rest reach a verdict rather than being dropped."""
        folder = os.path.join(SAMPLES, "languages")
        names = sorted(name for name in os.listdir(folder) if name.endswith(".txt"))
        self.assertGreaterEqual(len(names), 6)
        extractor = DocumentExtractor()
        pipeline = SIFPipeline(backend="hashing")
        for name in names:
            document = extractor.extract(os.path.join(folder, name))
            self.assertGreater(len(document.text), 200, name)
            result = pipeline.analyze(document.text, reference=name)
            self.assertTrue(result.iogp_rule, f"{name} produced no verdict")
            self.assertTrue(result.needs_review,
                            f"{name}: an untranslated report must reach a person")

    def test_the_log_view_shows_what_the_session_did(self) -> None:
        self._analyse_samples()
        self.window.navigate("settings")          # the log only refreshes when visible
        self.window._refresh_logs()
        self.assertGreater(self.window.settings_view.log_table.rowCount(), 0)


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestBuildOneStillWorks(unittest.TestCase):
    """Build 1 is a shipped product in its own right - build 2 must not disturb it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _application()

    def test_the_original_window_analyses_the_seed_corpus(self) -> None:
        import main

        window = main.MainWindow()
        try:
            window.load_seed_data()
            self.assertTrue(window.worker.wait(180_000))
            self.app.processEvents()
            self.assertEqual(len(window.rows), 5)
            self.assertEqual(window.matrix_view.table.rowCount(), 5)
            self.assertGreaterEqual(window.review_view.table.rowCount(), 1)
        finally:
            window.close()

    def test_the_two_builds_coexist_in_one_process(self) -> None:
        import main
        import main2

        self.assertIsNot(main.MainWindow, main2.MainWindow)
        self.assertTrue(hasattr(main2.MainWindow, "record_decision"))
        self.assertFalse(hasattr(main.MainWindow, "record_decision"),
                         "the review bench belongs to build 2 only")


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestExtractedDocumentActions(unittest.TestCase):
    """Each extracted document carries its own Preview, Analyse and Remove."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _application()

    def setUp(self) -> None:
        import main2

        # A modal dialog never returns in a headless run - it hangs the suite
        # rather than failing it - so record the call instead of showing one.
        self._boxes = (main2.QMessageBox.information, main2.QMessageBox.warning)
        main2.QMessageBox.information = lambda *args, **kw: None
        main2.QMessageBox.warning = lambda *args, **kw: None
        self.addCleanup(self._restore_boxes)

        self.window = main2.MainWindow()
        self.addCleanup(self.window.close)
        self.window.show()
        self.app.processEvents()
        for name, text in (
                ("first.txt", "A scaffold at six metres with no harness anchored.\n\n"
                              "An 11 kV feeder left ungrounded with no LOTO applied."),
                ("second.pdf", "A confined space entry with no gas test carried out.")):
            self.window.on_document_ready(
                {"name": name, "backend": "text", "pages": 1, "confidence": None,
                 "characters": len(text), "blocks": 2, "note": "-", "text": text})
        self.app.processEvents()

    def _restore_boxes(self) -> None:
        import main2

        main2.QMessageBox.information, main2.QMessageBox.warning = self._boxes

    def test_the_controls_sit_on_the_row_and_are_not_hidden_off_the_edge(self) -> None:
        from ui2.views import DOCUMENT_ACTION_COLUMNS

        table = self.window.ingest_view.document_table
        column = next(index for index, (_, key, _) in enumerate(DOCUMENT_ACTION_COLUMNS)
                      if key == "_actions")
        self.assertLess(column, 3, "the controls must not sit behind a horizontal scroll")

        for row in range(table.rowCount()):
            holder = table.cellWidget(row, column)
            self.assertIsNotNone(holder, f"row {row} has no controls")
            labels = [child.text() for child in holder.children()
                      if hasattr(child, "text") and child.text()]
            self.assertEqual(labels, ["Preview", "Analyse", "Remove"])

    def test_preview_shows_the_document_whose_button_was_pressed(self) -> None:
        self.window.preview_document(1)
        self.app.processEvents()
        self.assertIn("confined space",
                      self.window.ingest_view.preview.toPlainText().lower())

    def test_remove_drops_only_that_document_and_its_blocks(self) -> None:
        self.assertEqual(len(self.window.pending_blocks), 3)

        self.window.remove_document(0)
        self.app.processEvents()

        self.assertEqual([d["name"] for d in self.window.documents], ["second.pdf"])
        self.assertEqual(len(self.window.pending_blocks), 1,
                         "removing one document must not strand the other's blocks")

    def test_analysing_one_document_analyses_only_its_blocks(self) -> None:
        self.window.analyse_document(1)
        self.assertTrue(self.window.worker.wait(120_000))
        self.app.processEvents()

        self.assertEqual(len(self.window.rows), 1)
        self.assertIn("confined space", self.window.rows[0]["raw_text"].lower())


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestHotspotsPage(unittest.TestCase):
    """The hotspots page says why it is empty instead of showing a blank grid."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _application()

    def setUp(self) -> None:
        import main2

        self._boxes = (main2.QMessageBox.information, main2.QMessageBox.warning)
        main2.QMessageBox.information = lambda *args, **kw: None
        main2.QMessageBox.warning = lambda *args, **kw: None

    def tearDown(self) -> None:
        import main2

        main2.QMessageBox.information, main2.QMessageBox.warning = self._boxes

    def test_an_empty_page_explains_itself_and_does_not_repeat_its_title(self) -> None:
        import main2

        window = main2.MainWindow()
        self.addCleanup(window.close)
        window.navigate("hotspots")
        self.app.processEvents()

        # isHidden(), not isVisible(): the latter is false for any page that is
        # not the current one in the stack, whatever this page asked for.
        self.assertTrue(window.hotspot_view.table.isHidden(),
                        "an empty grid reads as a broken page")
        note = window.hotspot_view._empty_note
        self.assertFalse(note.isHidden())
        self.assertIn("at least two", note.text())
        # titled() supplies the page heading; the panel must not repeat it.
        from PyQt6.QtWidgets import QLabel

        repeated = [label for label in window.hotspot_view.panel.findChildren(QLabel)
                    if label.objectName() == "SectionTitle"]
        self.assertEqual(repeated, [], "the page heading is shown twice")

    def test_the_table_comes_back_once_there_are_hotspots(self) -> None:
        from main import read_csv_reports
        import main2

        window = main2.MainWindow()
        self.addCleanup(window.close)
        window.show()
        self.app.processEvents()
        narratives, references = read_csv_reports(CSV_SAMPLE)
        window._start(window._analysis_worker(texts=narratives, references=references))
        self.assertTrue(window.worker.wait(180_000))
        self.app.processEvents()

        self.assertGreater(window.hotspot_view.table.rowCount(), 0)
        self.assertFalse(window.hotspot_view.table.isHidden())
        self.assertTrue(window.hotspot_view._empty_note.isHidden())


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestImportSpeed(unittest.TestCase):
    """Importing a corpus must cost about what analysing it costs.

    The engine analyses a report in roughly 3ms. The console around it used to
    add ~90ms on top of every one - a 25ms sleep put there to make rows stream
    visibly, plus a full re-aggregation and chart repaint every fifth row, plus
    a scroll-to-bottom that laid the table out again each time. None of that is
    analysis, and on a real shift's corpus it was the whole wait.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _application()

    def test_the_console_adds_little_to_the_cost_of_analysis(self) -> None:
        import time

        from main import read_csv_reports
        import main2

        narratives, _ = read_csv_reports(CSV_SAMPLE)
        texts = [f"{text} Distinct marker {index}."
                 for index, text in enumerate(narratives * 6)]
        references = [f"SPEED-{index}" for index in range(len(texts))]

        window = main2.MainWindow()
        self.addCleanup(window.close)
        window.show()
        self.app.processEvents()

        # One warm run first: the encoder resolves once per process, and that
        # cost belongs to start-up rather than to the import being measured.
        window._start(window._analysis_worker(texts=texts[:6], references=references[:6]))
        self.assertTrue(window.worker.wait(180_000))
        self.app.processEvents()

        started = time.perf_counter()
        window._start(window._analysis_worker(texts=texts, references=references))
        self.assertTrue(window.worker.wait(180_000), "import did not finish")
        self.app.processEvents()
        per_report = (time.perf_counter() - started) / len(texts) * 1000

        self.assertEqual(len(window.rows), len(texts))
        self.assertLess(per_report, 40,
                        f"{per_report:.0f}ms a report - the console is adding more "
                        "than the analysis costs")


class _StubOllama:
    """A local LLM that is reachable, or isn't, without a server."""

    def __init__(self, usable: bool = True) -> None:
        self._usable = usable
        self.model = "llama3.2"
        self.host = "http://localhost:11434"
        self.translated: List[str] = []
        self.readiness_checks = 0

    def ready(self) -> bool:
        self.readiness_checks += 1
        return self._usable

    def status(self) -> str:
        return (f"Ollama ready at {self.host} - model '{self.model}'" if self._usable
                else f"Ollama not reachable at {self.host} - start it with 'ollama serve'")

    def models(self) -> List[str]:
        return ["llama3.2:latest"] if self._usable else []

    def translate(self, text: str) -> str:
        self.translated.append(text)
        return "Booster pump 11 kV feeder was earthed" if self._usable else ""


#: A Tamil narrative, so ``looks_non_latin`` routes it to the translator.
TAMIL_REPORT = ("பூஸ்டர் பம்பின் தடுப்பு பராமரிப்பின் போது 11 கிலோ வோல்ட் "
                "ஊட்டி கேபிள் எர்த் செய்யப்படவில்லை")


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestTranslationWithoutAManualProbe(unittest.TestCase):
    """Translation must not depend on anyone having pressed "Check Ollama".

    The defect this pins down: ``llm_online`` started False on every launch and
    was only ever set by the manual probe, while the analysis worker took its
    translator from that flag. A running, reachable Ollama therefore sat unused
    and every non-English report was filed "NOT TRANSLATED - ... START OLLAMA",
    telling the operator to start something that was already running.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _application()

    def setUp(self) -> None:
        import main2

        self.window = main2.MainWindow()
        self.addCleanup(self.window.close)

    def _analyse(self, text: str) -> None:
        self.window._start(self.window._analysis_worker(texts=[text],
                                                        references=["DOC-001"]))
        self.assertTrue(self.window.worker.wait(120_000), "analysis did not finish")
        self.app.processEvents()

    def test_a_fresh_session_translates_with_no_probe_first(self) -> None:
        stub = _StubOllama(usable=True)
        self.window.llm = stub
        self.assertFalse(self.window.llm_online,
                         "a fresh session has probed nothing - that is the whole point")

        self._analyse(TAMIL_REPORT)

        self.assertEqual(stub.translated, [TAMIL_REPORT],
                         "the report was never even handed to the translator")
        self.assertTrue(self.window.rows[0]["translated_text"],
                        "an English rendering must reach the review bench")
        self.assertTrue(self.window.llm_online,
                        "readiness learned from a real attempt must be kept")

    def test_an_unreachable_host_is_named_rather_than_left_silent(self) -> None:
        stub = _StubOllama(usable=False)
        self.window.llm = stub

        self._analyse(TAMIL_REPORT)

        self.assertEqual(self.window.rows[0]["translated_text"], "",
                         "nothing may be passed off as a translation")
        self.assertFalse(self.window.llm_online)
        self.assertIn("not reachable", self.window.llm_message,
                      "the map must carry the real reason, not 'not checked yet'")

    def test_an_english_corpus_never_asks_the_translator_anything(self) -> None:
        """Readiness is resolved lazily, so an English corpus opens no socket."""
        stub = _StubOllama(usable=True)
        self.window.llm = stub

        self._analyse("No harness worn while working at 6 m on the scaffold; "
                      "the lanyard was clipped to the handrail.")

        self.assertEqual(stub.readiness_checks, 0)
        self.assertEqual(stub.translated, [])

    def test_readiness_is_settled_once_per_run_not_once_per_report(self) -> None:
        stub = _StubOllama(usable=True)
        self.window.llm = stub

        self.window._start(self.window._analysis_worker(
            texts=[TAMIL_REPORT, TAMIL_REPORT + " மீண்டும்", TAMIL_REPORT + " மூன்று"],
            references=["DOC-001", "DOC-002", "DOC-003"]))
        self.assertTrue(self.window.worker.wait(120_000), "analysis did not finish")
        self.app.processEvents()

        self.assertEqual(stub.readiness_checks, 1)
        self.assertEqual(len(stub.translated), 3)

    def test_switching_translation_off_is_still_honoured(self) -> None:
        stub = _StubOllama(usable=True)
        self.window.llm = stub
        self.window.set_translation(False)

        self._analyse(TAMIL_REPORT)

        self.assertEqual(stub.readiness_checks, 0)
        self.assertEqual(self.window.rows[0]["translated_text"], "")


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestDeepNavyBuild(unittest.TestCase):
    """app.py: the same console as app2.py, wearing the second skin.

    The point of the re-skin being an entry-point concern rather than a second
    controller is that neither build can quietly lose a capability the other
    has. These tests hold that: the window app.py builds is the full one, and
    the palette really does move.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _application()

    def setUp(self) -> None:
        from ui.theme import C

        # apply_palette rebinds module-level state, so put it back afterwards or
        # every later test in this process inherits the second skin.
        self._palette = {name: getattr(C, name) for name in vars(C)
                         if name.isupper() and isinstance(getattr(C, name), str)}

    def tearDown(self) -> None:
        from ui.theme import apply_palette

        apply_palette(self._palette)

    def test_it_carries_every_capability_of_the_untouched_build(self) -> None:
        import app

        window = app.build_window()
        self.addCleanup(window.close)

        for capability in ("workflow", "ingest_view", "dashboard", "report_view",
                           "hotspot_view", "review_view", "analytics_view",
                           "engines_view", "settings_view", "audit", "decisions",
                           "mlops", "pipeline", "llm", "extractor"):
            self.assertTrue(hasattr(window, capability),
                            f"app.py lost {capability}, which app2.py has")
        for action in ("generate_bulletin", "export_audit", "clear_corpus",
                       "record_decision", "train_model", "check_llm"):
            self.assertTrue(callable(getattr(window, action, None)),
                            f"app.py lost {action}()")

    def test_the_skin_actually_changes_the_palette(self) -> None:
        import app
        from ui import gov_theme
        from ui.theme import C

        before = C.ACCENT
        window = app.build_window()
        self.addCleanup(window.close)

        self.assertEqual(C.ACCENT, gov_theme.PALETTE["ACCENT"])
        self.assertNotEqual(C.ACCENT, before, "the second skin must not be the first")
        self.assertIn("14b8a6", window.styleSheet())

    def test_a_long_field_value_cannot_widen_the_detail_panel(self) -> None:
        """A long barrier list must elide, not push the panel past its pane.

        FieldRow elides to fit, but a plain label still reports its full text as
        its minimum width, so before this was pinned the "Failed barrier" row
        could force the whole case wider than the pane holding it - taking the
        wrapped brief and the narrative off the right-hand edge with it. It fit
        in one skin only by luck, on one corpus.
        """
        from PyQt6.QtWidgets import QScrollArea

        import app

        window = app.build_window()
        self.addCleanup(window.close)
        window.resize(1600, 950)
        window.show()
        self.app.processEvents()

        window.rows = [dict(
            reference="LONG-1", raw_text="Cable left ungrounded", sif_potential=True,
            risk_score=99.0, risk_band="Critical", iogp_rule="Energy Isolation",
            activity="maintenance", location="Pump station", energy_source="Electrical energy",
            barrier_failure="Energy isolation / LOTO not applied or verified; Permit to "
                            "work / JSA absent, expired or not followed; Exclusion zone / "
                            "barricading absent; Gas testing / ventilation missing",
            high_energy=True, barrier_failed=True, confidence=0.9)]
        window._refresh()
        window.navigate("review")
        self.app.processEvents()
        window.select_review_row(0)
        self.app.processEvents()

        for area in window.review_view.findChildren(QScrollArea):
            self.assertLessEqual(
                area.widget().width(), area.viewport().width() + 1,
                "a field value has pushed the case wider than the pane holding it")

    def test_the_header_ornament_is_stripped_here_but_not_in_the_other_build(self) -> None:
        """The mark, avatar and search box are hidden for this build only."""
        import app
        import main2

        navy = app.build_window()
        self.addCleanup(navy.close)
        navy.show()
        self.app.processEvents()
        for name in ("mark", "avatar", "search"):
            self.assertTrue(getattr(navy.header, name).isHidden(),
                            f"{name} should be hidden in the deep-navy build")

        blue = main2.MainWindow()
        self.addCleanup(blue.close)
        blue.show()
        self.app.processEvents()
        for name in ("mark", "avatar", "search"):
            self.assertFalse(getattr(blue.header, name).isHidden(),
                             f"app2.py must keep its {name}")

    def test_the_two_builds_are_told_apart_in_the_title_bar(self) -> None:
        import app

        window = app.build_window()
        self.addCleanup(window.close)
        self.assertNotEqual(window.windowTitle(), "SENTRA - build 2")
        self.assertIn("SENTRA", window.windowTitle())


if __name__ == "__main__":
    unittest.main(verbosity=2)
