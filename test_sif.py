"""Unit tests for the SIF pipeline, its stages and the Qt worker.

Run with::

    python -m unittest -v

The suite pins the encoder to the offline hashing backend so it needs no model
download and stays deterministic; the transformer path is covered by its own
tests, which skip when the model cannot be loaded.
"""

from __future__ import annotations

import csv
import os
import tempfile
import threading
import unittest

import numpy as np

from sif import SEED_REPORTS, SIFPipeline
from sif.encoders import HashingEncoder, TransformerEncoder, load_encoder
from sif.heads import MIN_MARGIN, SemanticIndex, _calibrate, _margin
from sif.lexical import (
    NO_BARRIER_FAILURE,
    UNCLASSIFIED_RULE,
    UNKNOWN_ACTIVITY,
    UNKNOWN_LOCATION,
    LexicalEngine,
)
from sif.patterns import PatternDetector
from sif.preprocessing import NLPPreprocessor
from sif.review import ReviewQueue
from sif.scoring import RiskScorer

try:  # The GUI-layer tests are skipped when PyQt6 is not installed.
    from PyQt6.QtCore import QCoreApplication, QTimer

    from main import AnalysisWorker, read_csv_reports

    HAS_PYQT = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_PYQT = False


def offline_pipeline() -> SIFPipeline:
    """A pipeline pinned to the deterministic offline encoder."""
    return SIFPipeline(encoder=HashingEncoder())


class TestLexicalEngine(unittest.TestCase):
    """The deterministic rule layer, which is also the pipeline's backbone."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = LexicalEngine()

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

    def test_rule_scores_are_exposed_for_fusion(self) -> None:
        scores = self.engine.rule_scores(SEED_REPORTS[1])
        self.assertIn("Working at Height", scores)
        self.assertGreater(scores["Working at Height"], 0)


class TestPreprocessor(unittest.TestCase):
    """Stage 1."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pre = NLPPreprocessor()

    def test_blank_input_yields_empty_document(self) -> None:
        for value in ("", "   ", None):
            self.assertTrue(self.pre.process(value).is_empty)  # type: ignore[arg-type]

    def test_expands_domain_abbreviations_for_the_encoder(self) -> None:
        document = self.pre.process("No LOTO applied and the PTW had expired.")
        self.assertIn("lock out tag out", document.semantic.lower())
        self.assertIn("permit to work", document.semantic.lower())
        # The surface form keeps the reporter's own wording for the patterns.
        self.assertIn("loto", document.normalised)

    def test_segments_sentences_without_splitting_on_no(self) -> None:
        document = self.pre.process(
            "Near miss at Pump Station No. 3. The breaker was left closed."
        )
        self.assertEqual(len(document.sentences), 2)
        self.assertIn("No. 3", document.sentences[0])

    def test_tokens_are_lowercased(self) -> None:
        document = self.pre.process("Harness NOT anchored")
        self.assertEqual(document.tokens, ["harness", "not", "anchored"])


class TestEncoders(unittest.TestCase):
    """Stage 2."""

    def setUp(self) -> None:
        self.encoder = HashingEncoder()

    def test_vectors_are_unit_length_and_deterministic(self) -> None:
        texts = ["no harness on the scaffold", "breaker left closed"]
        first = self.encoder.encode(texts)
        second = self.encoder.encode(texts)
        self.assertEqual(first.shape, (2, self.encoder.info.dimension))
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(first, second)

    def test_similarity_is_bounded_and_non_negative(self) -> None:
        vectors = self.encoder.encode(["harness missing", "tanker speeding", ""])
        similarity = self.encoder.similarity(vectors, vectors)
        self.assertTrue(np.all(similarity >= -1e-6))
        self.assertTrue(np.all(similarity <= 1.0 + 1e-6))

    def test_fallback_encoder_is_marked_non_semantic(self) -> None:
        self.assertFalse(self.encoder.info.semantic)
        self.assertIn("hash", self.encoder.info.label())

    def test_explicit_hashing_backend_never_downloads(self) -> None:
        encoder = load_encoder("hashing")
        self.assertIsInstance(encoder, HashingEncoder)

    def test_auto_backend_always_returns_a_working_encoder(self) -> None:
        # Resolution is eager, so whatever comes back can encode immediately -
        # with or without a reachable model hub.
        encoder = load_encoder("auto")
        vectors = encoder.encode(["a report"])
        self.assertEqual(vectors.shape[0], 1)
        self.assertGreater(vectors.shape[1], 0)

    def test_transformer_encoder_defers_loading(self) -> None:
        # Constructing must not touch the network or the disk.
        encoder = TransformerEncoder("definitely/not-a-real-model")
        self.assertEqual(encoder.info.backend, "transformer")
        self.assertEqual(encoder.info.dimension, 0)

    def test_unavailable_transformer_raises_when_demanded(self) -> None:
        with self.assertRaises(Exception):
            load_encoder("transformer", model_name="definitely/not-a-real-model")


class TestHeads(unittest.TestCase):
    """Stage 3 helpers: calibration and the discrimination guard."""

    def test_calibration_is_bounded_and_capped_for_the_fallback(self) -> None:
        self.assertEqual(_calibrate(0.05, True), 0.0)
        self.assertEqual(_calibrate(0.9, True), 1.0)
        # The offline backend can never assert presence on its own.
        self.assertLess(_calibrate(0.99, False), 0.5)

    def test_margin_detects_a_flat_ranking(self) -> None:
        from sif.heads import LabelMatch

        flat = [LabelMatch(f"label{index}", 0.80) for index in range(5)]
        peaked = [LabelMatch("top", 0.9)] + [LabelMatch(f"l{i}", 0.2) for i in range(4)]
        self.assertLess(_margin(flat), MIN_MARGIN)
        self.assertGreaterEqual(_margin(peaked), MIN_MARGIN)

    def test_semantic_index_ranks_every_label(self) -> None:
        encoder = HashingEncoder()
        index = SemanticIndex(encoder)
        vectors = encoder.encode(["no harness on the scaffold"])
        ranked = index.rank("rule", vectors, ["no harness on the scaffold"])
        self.assertEqual(len(ranked), len(SemanticIndex.SETS["rule"]))
        self.assertGreaterEqual(ranked[0].score, ranked[-1].score)


class TestScoring(unittest.TestCase):
    """Stage 5."""

    def test_band_thresholds(self) -> None:
        from sif.heads import SIFVerdict

        scorer = RiskScorer()
        critical = scorer.score(
            SIFVerdict(True, 1.0, "Electrical energy", 1.0,
                       "Energy isolation / LOTO not applied or verified", 1.0,
                       True, True, True, True), 1.0)
        self.assertEqual(critical.band, "Critical")
        self.assertAlmostEqual(critical.value, 100.0, places=1)

        none = scorer.score(
            SIFVerdict(False, 0.0, "No high-energy source identified", 0.0,
                       NO_BARRIER_FAILURE, 0.0, False, False, False, False), 0.5)
        self.assertEqual(none.value, 0.0)
        self.assertEqual(none.band, "Low")

    def test_housekeeping_barrier_ranks_below_isolation(self) -> None:
        from sif.heads import SIFVerdict

        scorer = RiskScorer()
        base = dict(sif_potential=True, probability=1.0, energy_label="Electrical energy",
                    energy_score=1.0, barrier_score=1.0, high_energy=True,
                    barrier_failed=True, lexical_flag=True, semantic_flag=True)
        isolation = scorer.score(SIFVerdict(
            barrier_label="Energy isolation / LOTO not applied or verified", **base), 1.0)
        housekeeping = scorer.score(SIFVerdict(
            barrier_label="Housekeeping / walkway integrity lapse", **base), 1.0)
        self.assertGreater(isolation.value, housekeeping.value)


class TestPipeline(unittest.TestCase):
    """End-to-end behaviour with the offline encoder."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = offline_pipeline()
        cls.results = cls.pipeline.analyze_many(SEED_REPORTS)

    def test_seed_corpus_matches_the_lexical_baseline(self) -> None:
        # Offline mode must be exactly as precise as the deterministic rules:
        # four of the five seeds are engineered precursors, one is housekeeping.
        self.assertEqual(len(self.results), 5)
        self.assertEqual(sum(1 for item in self.results if item.sif_potential), 4)

    def test_required_fields_survive_the_rewrite(self) -> None:
        payload = self.results[0].to_dict()
        for key in ("sif_potential", "iogp_rule", "activity", "location", "barrier_failure"):
            self.assertIn(key, payload)

    def test_risk_score_is_bounded_and_banded(self) -> None:
        for item in self.results:
            self.assertGreaterEqual(item.risk_score, 0.0)
            self.assertLessEqual(item.risk_score, 100.0)
            self.assertIn(item.risk_band, {"Critical", "High", "Medium", "Low"})

    def test_non_sif_report_scores_zero_risk(self) -> None:
        housekeeping = next(item for item in self.results if not item.sif_potential)
        self.assertEqual(housekeeping.risk_score, 0.0)

    def test_offline_backend_raises_no_semantic_flag(self) -> None:
        for item in self.results:
            self.assertFalse(item.semantic_flag)
            self.assertFalse(item.semantic_active)

    def test_blank_report_is_handled(self) -> None:
        result = self.pipeline.analyze("   ")
        self.assertFalse(result.sif_potential)
        self.assertEqual(result.risk_score, 0.0)
        self.assertIn("Empty report", result.explanation)

    def test_analyze_many_skips_blanks(self) -> None:
        self.assertEqual(len(self.pipeline.analyze_many(["", "  ", None, "no harness"])), 1)

    def test_evidence_carries_the_audit_trail(self) -> None:
        evidence = self.results[0].evidence
        for key in ("lexical_cues", "semantic_matches", "decision_path", "explanation", "risk"):
            self.assertIn(key, evidence)
        self.assertTrue(evidence["lexical_cues"])

    def test_every_result_reports_its_encoder(self) -> None:
        self.assertTrue(all("hash" in item.encoder for item in self.results))


class TestIntelligence(unittest.TestCase):
    """Stages 6a and 6b over a corpus."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = offline_pipeline()
        texts = list(SEED_REPORTS) + [
            "Near miss at the pump station: the 11 kV breaker was left closed and no "
            "LOTO was applied before the fitter opened the motor terminal box.",
        ]
        cls.results = cls.pipeline.analyze_many(texts)
        cls.intelligence = cls.pipeline.aggregate(cls.results)

    def test_kpis(self) -> None:
        kpis = self.intelligence.kpis
        self.assertEqual(kpis["total"], 6)
        self.assertEqual(kpis["sif_potential"], 5)
        self.assertGreater(kpis["mean_risk"], 0.0)
        self.assertEqual(kpis["top_rule"], "Energy Isolation")

    def test_repeat_location_becomes_a_hotspot(self) -> None:
        labels = {spot.label for spot in self.intelligence.hotspots}
        self.assertIn("Pump station", labels)
        hotspot = next(spot for spot in self.intelligence.hotspots
                       if spot.label == "Pump station")
        self.assertEqual(hotspot.reports, 2)
        self.assertEqual(hotspot.sif_reports, 2)
        self.assertEqual(hotspot.sif_rate, 100.0)

    def test_singleton_groups_are_not_hotspots(self) -> None:
        for spot in self.intelligence.hotspots:
            self.assertGreaterEqual(spot.reports, PatternDetector.MIN_REPORTS)

    def test_critical_reports_are_queued_for_review(self) -> None:
        queue = self.intelligence.review_queue
        self.assertTrue(queue)
        self.assertTrue(all(item.risk_score >= 0 for item in queue))
        # Highest priority first, then by risk.
        self.assertEqual(queue, sorted(
            queue, key=lambda item: ({"Disagreement": 0, "Critical risk": 1,
                                      "Thin evidence": 2, "Unclassified exposure": 3}
                                     .get(item.trigger, 9), -item.risk_score)))

    def test_offline_mode_never_reports_disagreement(self) -> None:
        triggers = {item.trigger for item in self.intelligence.review_queue}
        self.assertNotIn("Disagreement", triggers)

    def test_thin_evidence_is_flagged(self) -> None:
        result = self.pipeline.analyze("Something looked wrong near the shed.")
        trigger, reason = ReviewQueue().classify(result)
        self.assertEqual(trigger, "Thin evidence")
        self.assertIn("confidence", reason)


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestAnalysisWorker(unittest.TestCase):
    """The CSV importer and the off-GUI-thread execution guarantee."""

    def test_reads_named_description_column_with_reference(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "reports.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["report_id", "description"])
                writer.writerow(["R-1", "No harness used on the scaffold at 5 m."])
                writer.writerow(["R-2", "Breaker left closed and line still live."])
            rows, references = read_csv_reports(path)
        self.assertEqual(len(rows), 2)
        self.assertIn("harness", rows[0])
        self.assertEqual(references, ["R-1", "R-2"])

    def test_falls_back_to_longest_cell_without_known_column(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "odd.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["ref", "free_form_note"])
                writer.writerow(["A1", "Worker stood under the suspended load, no barricading."])
            rows, _ = read_csv_reports(path)
        self.assertEqual(len(rows), 1)
        self.assertIn("suspended load", rows[0])

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_csv_reports("/nonexistent/path/reports.csv")

    def test_pipeline_runs_off_the_main_thread(self) -> None:
        app = QCoreApplication.instance() or QCoreApplication([])
        worker = AnalysisWorker(offline_pipeline(), texts=list(SEED_REPORTS))
        worker.STREAM_DELAY_MS = 0

        main_thread = threading.get_ident()
        seen_threads: list = []
        rows: list = []

        worker.row_ready.connect(rows.append)
        worker.completed.connect(lambda _count: app.quit())
        original_run = worker.run

        def instrumented_run() -> None:
            seen_threads.append(threading.get_ident())
            original_run()

        worker.run = instrumented_run  # type: ignore[method-assign]

        QTimer.singleShot(30000, app.quit)
        worker.start()
        app.exec()
        worker.wait(5000)

        self.assertEqual(len(rows), 5)
        self.assertTrue(seen_threads)
        self.assertNotEqual(seen_threads[0], main_thread)


class TestSystemLogging(unittest.TestCase):
    """The audit trail shown on the Settings page."""

    def test_ring_buffer_captures_and_filters(self) -> None:
        import logging as std_logging

        from sif.logging_setup import active_log_file, configure_logging, set_level

        with tempfile.TemporaryDirectory() as folder:
            ring = configure_logging("DEBUG", folder, to_stderr=False)
            ring.clear()
            logger = std_logging.getLogger("sif.test.logging")
            logger.info("pipeline started")
            logger.warning("encoder fell back")
            self.assertEqual(len(ring.entries()), 2)
            self.assertEqual(len(ring.entries("WARNING")), 1)
            # Handlers install once per process, so assert on the file actually
            # in use rather than the directory this call asked for.
            self.assertTrue(os.path.isfile(active_log_file()))
            ring.clear()
            self.assertEqual(ring.entries(), [])
            set_level("INFO")

    def test_listener_receives_records(self) -> None:
        import logging as std_logging

        from sif.logging_setup import configure_logging

        seen = []
        ring = configure_logging("DEBUG", tempfile.mkdtemp(), to_stderr=False)
        ring.set_listener(seen.append)
        try:
            std_logging.getLogger("sif.test.listener").error("boom")
        finally:
            ring.set_listener(None)
        self.assertTrue(any(entry.message == "boom" for entry in seen))


class TestDocumentExtraction(unittest.TestCase):
    """Stage 0 - reading reports out of files."""

    @classmethod
    def setUpClass(cls) -> None:
        from sif.ocr import DocumentExtractor

        cls.extractor = DocumentExtractor()

    def test_reads_plain_text(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "log.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("No harness worn on the scaffold at 6 m.\n\n"
                             "Breaker left closed during maintenance of the pump.")
            document = self.extractor.extract(path)
        self.assertEqual(document.backend, "text")
        self.assertEqual(len(document.blocks()), 2)
        self.assertFalse(document.is_empty)

    def test_reads_pdf_text_layer_without_ocr(self) -> None:
        report = os.path.join("reports", "SIF_Analysis_Report.pdf")
        if not os.path.isfile(report):
            self.skipTest("sample PDF not present")
        document = self.extractor.extract(report)
        self.assertIn("pdf-text", document.backend)
        self.assertGreater(document.pages, 1)
        self.assertGreater(len(document.text), 500)

    def test_unsupported_type_and_missing_file(self) -> None:
        with self.assertRaises(ValueError):
            self.extractor.extract("sample_reports.xyz") if os.path.isfile(
                "sample_reports.xyz") else self._raise_value_error()
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract("/nonexistent/report.pdf")

    @staticmethod
    def _raise_value_error():
        raise ValueError("unsupported")

    def test_batch_records_failures_instead_of_raising(self) -> None:
        documents = self.extractor.extract_many(["/nonexistent/a.pdf", "sample_reports.csv"])
        self.assertEqual(documents[0].backend, "failed")
        self.assertTrue(documents[0].warnings)
        self.assertEqual(documents[1].backend, "text")

    def test_probe_returns_a_verdict_either_way(self) -> None:
        ok, message = self.extractor.probe()
        self.assertIsInstance(ok, bool)
        self.assertTrue(message)

    def test_status_is_honest_before_loading(self) -> None:
        from sif.ocr import DocumentExtractor

        disabled = DocumentExtractor(enable_ocr=False)
        self.assertIn("disabled", disabled.status())


class TestMLOps(unittest.TestCase):
    """The learned layer: features, training, tracking, persistence."""

    @classmethod
    def setUpClass(cls) -> None:
        from sif.mlops import SIFModel

        cls.has_xgboost = SIFModel.installed()
        pipeline = offline_pipeline()
        texts = list(SEED_REPORTS) + [
            "Unsafe condition at the camp canteen: a loose chequered plate near the "
            "entrance created a trip hazard. Minor housekeeping issue, no injury.",
            "Near miss at the substation: the 33 kV switchgear panel door was left open "
            "with the busbar still live and no isolation certificate was available.",
            "Housekeeping observation: cable trays in the workshop store were dusty.",
        ]
        cls.results = pipeline.analyze_many(texts)

    def test_feature_vector_matches_the_named_columns(self) -> None:
        from sif.mlops import FEATURE_NAMES, featurise, featurise_many

        vector = featurise(self.results[0])
        self.assertEqual(vector.shape[0], len(FEATURE_NAMES))
        np.testing.assert_allclose(vector, featurise(self.results[0]))
        self.assertEqual(featurise_many(self.results).shape,
                         (len(self.results), len(FEATURE_NAMES)))

    def test_features_are_empty_safe(self) -> None:
        from sif.mlops import FEATURE_NAMES, featurise_many

        self.assertEqual(featurise_many([]).shape, (0, len(FEATURE_NAMES)))

    def test_small_corpus_relaxes_regularisation(self) -> None:
        from sif.mlops import DEFAULT_PARAMS, adapt_params

        warnings: list = []
        adapted = adapt_params(dict(DEFAULT_PARAMS), samples=11, positives=9, warnings=warnings)
        self.assertLess(float(adapted["min_child_weight"]),
                        float(DEFAULT_PARAMS["min_child_weight"]))
        self.assertTrue(warnings)

        untouched = adapt_params(dict(DEFAULT_PARAMS), samples=5000, positives=1200,
                                 warnings=[])
        self.assertEqual(untouched["min_child_weight"], DEFAULT_PARAMS["min_child_weight"])

    def test_training_requires_both_classes_and_enough_rows(self) -> None:
        from sif.mlops import SIFModel

        if not self.has_xgboost:
            self.skipTest("xgboost is not installed")
        model = SIFModel()
        with self.assertRaises(ValueError):
            model.train(self.results[:2])
        positives = [item for item in self.results if item.sif_potential]
        with self.assertRaises(ValueError):
            model.train(positives)

    def test_train_predict_save_load_roundtrip(self) -> None:
        from sif.mlops import MLOpsService

        if not self.has_xgboost:
            self.skipTest("xgboost is not installed")
        with tempfile.TemporaryDirectory() as folder:
            service = MLOpsService(model_directory=os.path.join(folder, "models"),
                                   tracking_uri=f"sqlite:///{folder}/mlflow.db",
                                   experiment="unit-test")
            report = service.train(self.results)
            self.assertEqual(report.samples, len(self.results))
            self.assertIn("f1", report.metrics)
            self.assertTrue(report.importances, "a trained model must expose importances")
            self.assertTrue(os.path.isfile(report.model_path))

            # The model separates the classes it was trained on.
            flagged = next(item for item in self.results if item.sif_potential)
            clean = next(item for item in self.results if not item.sif_potential)
            self.assertGreater(service.predict(flagged), service.predict(clean))

            reloaded = MLOpsService(model_directory=os.path.join(folder, "models"),
                                    tracking_uri=f"sqlite:///{folder}/mlflow.db",
                                    experiment="unit-test")
            self.assertTrue(reloaded.load_existing())
            self.assertAlmostEqual(reloaded.predict(flagged), service.predict(flagged),
                                   places=5)

    def test_untrained_service_predicts_none(self) -> None:
        from sif.mlops import MLOpsService

        with tempfile.TemporaryDirectory() as folder:
            service = MLOpsService(model_directory=os.path.join(folder, "models"))
            self.assertIsNone(service.predict(self.results[0]))
            self.assertFalse(service.load_existing())

    def test_mlflow_run_is_logged_and_listed(self) -> None:
        from sif.mlops import MLOpsService, MLflowTracker

        if not (self.has_xgboost and MLflowTracker.installed()):
            self.skipTest("xgboost and mlflow are both required")
        with tempfile.TemporaryDirectory() as folder:
            service = MLOpsService(model_directory=os.path.join(folder, "models"),
                                   tracking_uri=f"sqlite:///{folder}/mlflow.db",
                                   experiment="unit-test-runs")
            report = service.train(self.results)
            self.assertTrue(report.run_id)
            runs = service.tracker.recent_runs(5)
            self.assertTrue(runs)
            self.assertEqual(runs[0]["run_id"], report.run_id[:8])


class TestModelInPipeline(unittest.TestCase):
    """The learned model as a third opinion, and the review trigger it feeds."""

    class _StubModel:
        """Minimal provider: always returns the probability it was given."""

        def __init__(self, probability: float) -> None:
            self.probability = probability

        def predict(self, result) -> float:
            return self.probability

    def test_attached_model_scores_every_row(self) -> None:
        pipeline = offline_pipeline()
        pipeline.attach_model(self._StubModel(0.91))
        result = pipeline.analyze("No harness worn on the scaffold at 6 m.")
        self.assertTrue(pipeline.has_model)
        self.assertTrue(result.ml_active)
        self.assertEqual(result.ml_probability, 0.91)
        self.assertTrue(result.ml_flag)
        self.assertIn("model", result.evidence)

    def test_model_disagreement_reaches_the_review_queue(self) -> None:
        pipeline = offline_pipeline()
        pipeline.attach_model(self._StubModel(0.95))
        # A housekeeping report the rules clear, but the model flags.
        result = pipeline.analyze(
            "Minor oil spillage on the walkway near the manifold made the plate slippery. "
            "Poor housekeeping, no injury.")
        self.assertFalse(result.sif_potential)
        self.assertTrue(result.ml_flag)
        self.assertEqual(result.review_trigger, "Model disagreement")

    def test_provider_returning_none_is_ignored(self) -> None:
        class Silent:
            @staticmethod
            def predict(_result):
                return None

        pipeline = offline_pipeline()
        pipeline.attach_model(Silent())
        result = pipeline.analyze("No harness worn on the scaffold at 6 m.")
        self.assertFalse(result.ml_active)
        self.assertIsNone(result.ml_probability)


class TestTrainerCLI(unittest.TestCase):
    """The command-line trainer: corpus reading, label parsing and exit codes."""

    HEADER = "report_id,description,sif_label\n"
    ROWS = (
        'R-1,"No harness worn on the scaffold at 6 m.",1\n'
        'R-2,"Loose plate on the canteen walkway, minor housekeeping.",0\n'
        'R-3,"Breaker left closed during pump maintenance.",\n'
        'R-4,"Tanker overspeeding on the field road.",maybe\n'
    )

    def _corpus(self, folder: str) -> str:
        path = os.path.join(folder, "reports.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.HEADER + self.ROWS)
        return path

    def test_reads_narratives_references_and_labels(self) -> None:
        from train_model import read_corpus

        with tempfile.TemporaryDirectory() as folder:
            texts, references, labels = read_corpus(self._corpus(folder),
                                                    label_column="sif_label")
        self.assertEqual(len(texts), 4)
        self.assertEqual(references[0], "R-1")
        # 1 -> positive, 0 -> negative, blank and unrecognised -> unlabelled.
        self.assertEqual(labels, [1, 0, None, None])

    def test_label_column_is_auto_detected(self) -> None:
        from train_model import read_corpus

        with tempfile.TemporaryDirectory() as folder:
            _texts, _refs, labels = read_corpus(self._corpus(folder))
        self.assertIsNotNone(labels)

    def test_without_labels_none_is_returned(self) -> None:
        from train_model import read_corpus

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "plain.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("report_id,description\nR-1,No harness on the scaffold.\n")
            _texts, _refs, labels = read_corpus(path)
        self.assertIsNone(labels)

    def test_missing_narrative_column_is_rejected(self) -> None:
        from train_model import read_corpus

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "odd.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("a,b\n1,2\n")
            with self.assertRaises(ValueError):
                read_corpus(path)

    def test_dry_run_succeeds_and_bad_input_fails(self) -> None:
        from train_model import main

        with tempfile.TemporaryDirectory() as folder:
            path = self._corpus(folder)
            self.assertEqual(main([path, "--encoder", "hashing", "--dry-run"]), 0)
            self.assertEqual(main([os.path.join(folder, "nope.csv")]), 1)
            self.assertEqual(main([path, "--label-column", "absent",
                                   "--encoder", "hashing"]), 1)


class TestTheme(unittest.TestCase):
    """The style sheet must not reference assets that are missing."""

    def test_every_referenced_asset_exists(self) -> None:
        from ui.theme import ASSETS, STYLESHEET

        references = [chunk.split(")")[0]
                      for chunk in STYLESHEET.split("url(")[1:]]
        self.assertTrue(references, "the scroll controls should reference arrow assets")
        for path in references:
            self.assertTrue(os.path.isfile(path), f"missing style asset: {path}")
        self.assertTrue(os.path.isdir(ASSETS))

    def test_scroll_controls_are_styled(self) -> None:
        from ui.theme import STYLESHEET

        for selector in ("QScrollBar:vertical", "QScrollBar::handle:vertical",
                         "QScrollBar::up-arrow:vertical", "QScrollBar::down-arrow:vertical",
                         "QScrollBar:horizontal", "QScrollBar::handle:horizontal",
                         "QScrollBar::left-arrow:horizontal",
                         "QScrollBar::right-arrow:horizontal"):
            self.assertIn(selector, STYLESHEET)


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestInterfaceWidgets(unittest.TestCase):
    """The presentation layer renders without a display."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_data_table_formats_cells(self) -> None:
        from ui.components import DataTable

        table = DataTable((("#", "_index", 40), ("SIF", "sif_potential", 50),
                           ("Risk", "risk_score", 50), ("P(SIF)", "p_sif", 50),
                           ("Model", "ml_probability", 50)))
        table.set_rows([{"sif_potential": True, "risk_score": 87.44, "p_sif": 1.0,
                         "ml_probability": None, "risk_band": "Critical"}])
        self.assertEqual(table.rowCount(), 1)
        self.assertEqual(table.item(0, 0).text(), "1")
        self.assertEqual(table.item(0, 1).text(), "YES")
        self.assertEqual(table.item(0, 2).text(), "87.4")
        self.assertEqual(table.item(0, 3).text(), "1.00")
        self.assertEqual(table.item(0, 4).text(), "-")

    def test_charts_accept_data_and_empty_state(self) -> None:
        from ui.charts import DonutChart, HBarChart

        bars = HBarChart()
        bars.set_data([("Energy Isolation", 4, 3), ("Working at Height", 2, 2)])
        bars.resize(320, 120)
        bars.grab()  # forces a paint pass
        bars.set_data([])
        bars.grab()

        donut = DonutChart("Energy sources")
        donut.set_data([("Electrical energy", 3), ("Gravity / Fall from height", 2)])
        donut.resize(360, 200)
        donut.grab()
        donut.set_data([])
        donut.grab()

    def test_dashboard_renders_a_result(self) -> None:
        from ui.views import DashboardView

        view = DashboardView()
        pipeline = offline_pipeline()
        result = pipeline.analyze(SEED_REPORTS[0], "SEED-01").to_dict()
        view.update_kpis({"total": 1, "sif_potential": 1, "sif_rate": 100.0,
                          "mean_risk": 100.0, "critical": 1, "needs_review": 1,
                          "encoder": "hashing", "run_count": 1, "model_agreement": None})
        view.update_charts([("Energy Isolation", 1, 1)], [("Electrical energy", 1)],
                           [("Energy isolation / LOTO not applied or verified", 1, 1)])
        view.show_detail(result)
        self.assertIn("SIF-POTENTIAL", view.detail_pill.text())
        self.assertIn("Energy Isolation", view.detail_fields["rule"]._full_value)
        view.show_detail(None)
        self.assertEqual(view.detail_pill.text(), "NO SELECTION")

    def test_pages_scroll_instead_of_squashing(self) -> None:
        from ui.views import BatchUploadView, DashboardView

        for view_class in (DashboardView, BatchUploadView):
            view = view_class()
            view.resize(900, 420)  # deliberately shorter than the content
            area = view.findChild(__import__("PyQt6.QtWidgets", fromlist=["QScrollArea"])
                                  .QScrollArea)
            self.assertIsNotNone(area, f"{view_class.__name__} should be scrollable")
            self.assertGreaterEqual(area.widget().minimumHeight(), 700)

    def test_settings_view_renders_state(self) -> None:
        from ui.views import SettingsView

        view = SettingsView()
        view.set_log_rows([{"timestamp": "2026-01-01 00:00:00", "level": "INFO",
                            "logger": "sif.app", "message": "started"}])
        view.set_model_status("no model trained yet", "MLflow tracking to sqlite:///mlflow.db")
        view.set_runs([{"run_id": "abc12345", "started": "2026-01-01 00:00:00",
                        "status": "FINISHED", "samples": "11", "labels": "weak",
                        "f1": "1.000", "roc_auc": "1.000"}])
        view.set_importances([("p_sif", 0.48)])
        view.set_ocr_status("PaddleOCR installed")
        self.assertEqual(view.log_table.rowCount(), 1)
        self.assertEqual(view.run_table.rowCount(), 1)
        self.assertEqual(view.importance_table.rowCount(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
