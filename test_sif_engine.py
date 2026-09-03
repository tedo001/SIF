"""Unit tests for the SIF parsing engine and the analysis worker thread.

Run with::

    python -m unittest -v
"""

from __future__ import annotations

import csv
import os
import tempfile
import threading
import unittest

from sif_engine import (
    NO_BARRIER_FAILURE,
    SEED_REPORTS,
    UNCLASSIFIED_RULE,
    UNKNOWN_ACTIVITY,
    UNKNOWN_LOCATION,
    SIFEngine,
)

try:  # The GUI-layer tests are skipped when PyQt6 is not installed.
    from PyQt6.QtCore import QCoreApplication, QTimer

    from main import AnalysisWorker

    HAS_PYQT = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_PYQT = False


class TestSIFEngine(unittest.TestCase):
    """Behavioural tests for :class:`sif_engine.SIFEngine`."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = SIFEngine()

    def test_returns_all_required_keys(self) -> None:
        result = self.engine.analyze("Routine inspection at the workshop.")
        for key in ("sif_potential", "iogp_rule", "activity", "location", "barrier_failure"):
            self.assertIn(key, result)
        self.assertIsInstance(result["sif_potential"], bool)

    def test_height_with_failed_fall_protection_is_sif(self) -> None:
        result = self.engine.analyze(
            "Worker on an incomplete scaffold at 6 metres near the crude tank; "
            "his lanyard was not anchored and the guardrail was missing."
        )
        self.assertTrue(result["sif_potential"])
        self.assertEqual(result["iogp_rule"], "Working at Height")
        self.assertIn("Fall protection", result["barrier_failure"])

    def test_electrical_isolation_failure_is_sif(self) -> None:
        result = self.engine.analyze(
            "The 11 kV feeder at the pump station was left ungrounded and no LOTO "
            "was applied before cable jointing."
        )
        self.assertTrue(result["sif_potential"])
        self.assertEqual(result["iogp_rule"], "Energy Isolation")
        self.assertEqual(result["location"], "Pump station")

    def test_confined_space_without_gas_test_is_sif(self) -> None:
        result = self.engine.analyze(
            "Two workers entered the separator sump for cleaning without gas testing "
            "and with no hole watch posted."
        )
        self.assertTrue(result["sif_potential"])
        self.assertEqual(result["iogp_rule"], "Confined Space")

    def test_line_of_fire_under_suspended_load_is_sif(self) -> None:
        result = self.engine.analyze(
            "A helper stood directly under the suspended load while the crane slewed; "
            "the area was not barricaded."
        )
        self.assertTrue(result["sif_potential"])
        self.assertIn(result["iogp_rule"], {"Line of Fire", "Safe Mechanical Lifting"})

    def test_low_energy_observation_is_not_sif(self) -> None:
        result = self.engine.analyze(
            "Minor oil spillage on the walkway near the manifold made the plate "
            "slippery. Poor housekeeping, no injury."
        )
        self.assertFalse(result["sif_potential"])
        self.assertEqual(result["severity_hint"], "Low")

    def test_high_energy_with_controls_in_place_is_not_sif(self) -> None:
        result = self.engine.analyze(
            "Crane lift of the casing bundle completed at the drill site. Lift plan "
            "approved, exclusion zone barricaded and a certified rigger supervised."
        )
        self.assertFalse(result["sif_potential"])
        self.assertEqual(result["barrier_failure"], NO_BARRIER_FAILURE)

    def test_blank_input_returns_fallbacks(self) -> None:
        for value in ("", "   ", None):
            result = self.engine.analyze(value)  # type: ignore[arg-type]
            self.assertFalse(result["sif_potential"])
            self.assertEqual(result["iogp_rule"], UNCLASSIFIED_RULE)
            self.assertEqual(result["activity"], UNKNOWN_ACTIVITY)
            self.assertEqual(result["location"], UNKNOWN_LOCATION)
            self.assertEqual(result["barrier_failure"], NO_BARRIER_FAILURE)

    def test_unmatched_text_degrades_gracefully(self) -> None:
        result = self.engine.analyze("Lorem ipsum dolor sit amet, consectetur adipiscing.")
        self.assertFalse(result["sif_potential"])
        self.assertEqual(result["iogp_rule"], UNCLASSIFIED_RULE)

    def test_confidence_is_bounded(self) -> None:
        for report in SEED_REPORTS:
            confidence = self.engine.analyze(report)["confidence"]
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_seed_dataset_shape(self) -> None:
        self.assertEqual(len(SEED_REPORTS), 5)
        results = self.engine.analyze_many(SEED_REPORTS)
        self.assertEqual(len(results), 5)
        # Four seeds are engineered SIF precursors; one is a housekeeping issue.
        self.assertEqual(sum(1 for item in results if item["sif_potential"]), 4)


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestAnalysisWorker(unittest.TestCase):
    """Tests for the CSV importer and the off-GUI-thread execution guarantee."""

    def test_reads_named_description_column(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "reports.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "description"])
                writer.writerow(["1", "No harness used on the scaffold at 5 m."])
                writer.writerow(["2", "Breaker left closed and line still live."])
            rows = AnalysisWorker._read_csv(path)
        self.assertEqual(len(rows), 2)
        self.assertIn("harness", rows[0])

    def test_falls_back_to_longest_cell_without_known_column(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "odd.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["ref", "free_form_note"])
                writer.writerow(["A1", "Worker stood under the suspended load, no barricading."])
            rows = AnalysisWorker._read_csv(path)
        self.assertEqual(len(rows), 1)
        self.assertIn("suspended load", rows[0])

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            AnalysisWorker._read_csv("/nonexistent/path/reports.csv")

    def test_parsing_runs_off_the_main_thread(self) -> None:
        app = QCoreApplication.instance() or QCoreApplication([])
        worker = AnalysisWorker(SIFEngine(), texts=list(SEED_REPORTS))
        worker.STREAM_DELAY_MS = 0

        main_thread = threading.get_ident()
        seen_threads: list[int] = []
        rows: list[dict] = []

        worker.row_ready.connect(lambda row: rows.append(row))
        worker.started.connect(lambda: None)
        worker.completed.connect(lambda _count: app.quit())
        # Sample the worker's thread identity from inside its own run loop.
        worker.row_ready.connect(lambda _row: None)
        original_run = worker.run

        def instrumented_run() -> None:
            seen_threads.append(threading.get_ident())
            original_run()

        worker.run = instrumented_run  # type: ignore[method-assign]

        QTimer.singleShot(10000, app.quit)
        worker.start()
        app.exec()
        worker.wait(2000)

        self.assertEqual(len(rows), 5)
        self.assertTrue(seen_threads)
        self.assertNotEqual(seen_threads[0], main_thread)


if __name__ == "__main__":
    unittest.main(verbosity=2)
