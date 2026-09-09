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
        self.assertEqual(len(intelligence.review_queue), 18)


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
        self.assertEqual(len(outstanding), 18)

        by_reference = {result.reference: result for result in self.results}
        for position, item in enumerate(outstanding):
            result = by_reference[item.reference]
            decision = ("confirmed" if result.sif_potential else
                        "unclear" if position % 7 == 0 else "rejected")
            self.log.record(result, decision, reviewer="Functional test")

        self.assertEqual(queue.build(self.results, skip=self.log.decided()), [])
        counts = self.log.counts()
        self.assertEqual(counts["decided"], 18)
        self.assertEqual(counts["labels"], 18 - counts["unclear"])
        self.assertEqual(len(DecisionLog(self.log.path).load().entries), 18)

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
        self.assertEqual(self.window.outstanding_reviews, 18)
        self.assertEqual(self.window.review_view.table.rowCount(), 18)

        # Every page renders without raising
        for key in ("workflow", "ingest", "dashboard", "reports", "hotspots",
                    "review", "analytics", "engines", "settings"):
            self.window.navigate(key)
            self.app.processEvents()

    def test_the_workflow_map_reports_what_has_actually_happened(self) -> None:
        self._analyse_samples()
        cards = self.window.workflow.cards
        self.assertIn("18", cards["analyse"].status.text())
        self.assertIn("18", cards["review"].status.text())
        self.assertIn("report(s)", cards["dashboard"].status.text())

    def test_reviewing_a_report_moves_it_out_of_the_queue_and_into_the_trail(self) -> None:
        self._analyse_samples()
        self.window.navigate("review")
        view = self.window.review_view
        view.reviewer.setText("Functional test")
        view.select(0)
        first = view.reference.text()
        view._decide("confirmed")

        self.assertEqual(self.window.outstanding_reviews, 17)
        self.assertEqual(view.trail_table.rowCount(), 1)
        self.assertNotEqual(view.reference.text(), first, "the bench must advance")
        self.assertEqual(self.window.dashboard.tile_review._value.text(), "17")

    def _visible(self, table) -> int:
        """Rows the operator can actually see - filtering hides, it does not delete."""
        return sum(1 for row in range(table.rowCount()) if not table.isRowHidden(row))

    def test_the_search_box_filters_reports_and_hotspots_alike(self) -> None:
        self._analyse_samples()
        matrix = self.window.report_view.table
        hotspots = self.window.hotspot_view.table

        # "permit" appears in four narratives. A site name would not: the CSV
        # importer reads the narrative column only, so `site` never enters the
        # system - see the note in samples/README.md.
        self.window._apply_filter("permit")
        self.assertEqual(self._visible(matrix), 4)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
