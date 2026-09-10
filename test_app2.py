"""Tests for the second build: languages, the local LLM, and the workflow map.

These cover what ``app2.py`` adds. The original build keeps its own suite in
``test_sif.py``; both must pass, since the two share the analysis stack.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import re
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List

from sif.encoders import HashingEncoder
from sif.llm import OllamaEngine, looks_non_latin
from sif.ocr import (LANGUAGE_CHOICES, LANGUAGES, UNSUPPORTED_LANGUAGES,
                     DocumentExtractor, resolve_language)
from sif.pipeline import PipelineResult, SIFPipeline
from sif.review import (DECISION_SHORT, DecisionLog, ReviewDecision, ReviewQueue,
                        fingerprint)

try:
    import PyQt6.QtWidgets  # noqa: F401

    HAS_PYQT = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_PYQT = False

#: Anything in these ranges is a pictograph rather than a typographic mark.
EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U0000FE0F]")


class MockOllamaHandler(BaseHTTPRequestHandler):
    """Answers the two endpoints the client uses."""

    reply = {"sif_potential": True, "iogp_rule": "Working at Height",
             "activity": "Light fitting replacement", "location": "GGS-4 tank farm",
             "barrier_failure": "Fall protection not anchored",
             "rationale": "Unanchored lanyard at 6 m."}

    def _send(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        self._send({"models": [{"name": "llama3.2:latest"}]})

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/generate":
            if "Translate this workplace safety report" in payload.get("prompt", ""):
                self._send({"response": "Near miss at the pump station: the 11 kV feeder "
                                        "was left ungrounded and no LOTO was applied."})
            else:
                self._send({"response": json.dumps(self.reply)})
        else:
            self._send({"models": [{"name": "llama3.2:latest"}]})

    def log_message(self, *args):  # silence the test output
        pass


class TestOCRLanguages(unittest.TestCase):
    """Indian-language OCR selection."""

    def test_the_indian_languages_asked_for_are_present(self) -> None:
        names = " ".join(LANGUAGE_CHOICES).lower()
        for language in ("hindi", "marathi", "tamil", "telugu", "kannada", "urdu"):
            self.assertIn(language, names)

    def test_codes_match_what_paddleocr_accepts(self) -> None:
        # Read from the installed package rather than assumed: Tamil/Telugu/Kannada
        # have their own recognisers, the rest of the Indian set is Devanagari.
        self.assertEqual(LANGUAGES["Tamil / தமிழ்"][0], "ta")
        self.assertEqual(LANGUAGES["Telugu / తెలుగు"][0], "te")
        self.assertEqual(LANGUAGES["Kannada / ಕನ್ನಡ"][0], "ka")
        self.assertEqual(LANGUAGES["Hindi / हिन्दी"][0], "hi")
        self.assertEqual(LANGUAGES["Marathi / मराठी"][0], "mr")
        for name, (_code, script) in LANGUAGES.items():
            if name.startswith(("Hindi", "Marathi", "Nepali", "Sanskrit")):
                self.assertEqual(script, "Devanagari")

    def test_resolve_accepts_names_codes_and_aliases(self) -> None:
        self.assertEqual(resolve_language("Tamil / தமிழ்"), "ta")
        self.assertEqual(resolve_language("tamil"), "ta")
        self.assertEqual(resolve_language("ta"), "ta")
        self.assertEqual(resolve_language("kn"), "ka")   # ISO vs PaddleOCR spelling
        self.assertEqual(resolve_language("marathi"), "mr")

    def test_unknown_language_falls_back_to_english(self) -> None:
        self.assertEqual(resolve_language("klingon"), "en")
        self.assertEqual(resolve_language(""), "en")

    def test_unsupported_indian_languages_are_declared(self) -> None:
        # Naming them keeps the interface honest rather than reading them wrongly.
        joined = " ".join(UNSUPPORTED_LANGUAGES).lower()
        for language in ("bengali", "gujarati", "malayalam", "odia"):
            self.assertIn(language, joined)

    def test_capability_check_does_not_import_the_paddle_runtime(self) -> None:
        """The status line runs while the window opens; importing paddle costs

        about twenty seconds on Windows, so availability is resolved from the
        module search path instead.
        """
        import sys

        from sif.ocr import PaddleOCRBackend

        before = {name for name in sys.modules if name.split(".")[0]
                  in ("paddle", "paddleocr")}
        started = time.monotonic()
        result = PaddleOCRBackend.installed()
        elapsed = time.monotonic() - started
        after = {name for name in sys.modules if name.split(".")[0]
                 in ("paddle", "paddleocr")}
        self.assertIsInstance(result, bool)
        self.assertEqual(before, after, "the capability check must not import paddle")
        self.assertLess(elapsed, 1.0, "the capability check must be cheap")

    def test_extractor_reports_the_language_it_will_use(self) -> None:
        extractor = DocumentExtractor(language="Kannada / ಕನ್ನಡ")
        self.assertEqual(extractor.language, "ka")
        self.assertIn("Kannada", extractor.status())


class TestOCRModelCache(unittest.TestCase):
    """The models are a one-time download per machine - and must be seen as one."""

    def setUp(self) -> None:
        from sif import ocr, prefs

        self.ocr, self.prefs = ocr, prefs
        self.cache = tempfile.mkdtemp(prefix="sif-paddlex-")
        self._previous = os.environ.get("PADDLE_PDX_CACHE_HOME")
        os.environ["PADDLE_PDX_CACHE_HOME"] = self.cache
        # Keep the operator's real preferences out of the tests.
        self.stored: dict = {}
        self._real = (prefs.get, prefs.set_value)
        prefs.get = lambda key, default=None: self.stored.get(key, default)
        prefs.set_value = lambda key, value: self.stored.__setitem__(key, value)

    def tearDown(self) -> None:
        self.prefs.get, self.prefs.set_value = self._real
        if self._previous is None:
            os.environ.pop("PADDLE_PDX_CACHE_HOME", None)
        else:
            os.environ["PADDLE_PDX_CACHE_HOME"] = self._previous

    def _download(self, *names: str) -> None:
        """Pretend PaddleOCR has fetched these models."""
        for name in names:
            folder = os.path.join(self.cache, "official_models", name)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "inference.pdmodel"), "w") as handle:
                handle.write("weights")

    def test_the_cache_directory_follows_paddle_s_own_variable(self) -> None:
        self.assertEqual(self.ocr.cache_directory(), self.cache)

    def test_an_empty_machine_reports_nothing_cached(self) -> None:
        self.assertEqual(self.ocr.cached_models(), [])
        self.assertFalse(self.ocr.models_present())

    def test_a_half_finished_download_does_not_count_as_present(self) -> None:
        os.makedirs(os.path.join(self.cache, "official_models", "PP-OCRv5_det"))
        self.assertEqual(self.ocr.cached_models(), [],
                         "an empty folder is an interrupted download, not a model")
        self.assertFalse(self.ocr.models_present())

    def test_detection_alone_is_not_enough_to_read_a_page(self) -> None:
        self._download("PP-OCRv5_server_det")
        self.assertFalse(self.ocr.models_present())
        self._download("latin_PP-OCRv5_mobile_rec")
        self.assertTrue(self.ocr.models_present())

    def test_a_machine_that_has_the_models_is_told_so_every_start_up(self) -> None:
        extractor = self.ocr.DocumentExtractor(language="English")
        self.assertIn("download", extractor.status().lower())
        self._download("PP-OCRv5_server_det", "latin_PP-OCRv5_mobile_rec")
        self.assertIn("already cached", extractor.status())
        self.ocr.remember_verified("en")
        status = self.ocr.DocumentExtractor(language="English").status()
        self.assertIn("No download needed", status)
        self.assertIn(self.cache, status)

    def test_verification_is_remembered_per_language(self) -> None:
        self.ocr.remember_verified("ta")
        self.assertIn("ta", self.ocr.verified_languages())
        self.assertNotIn("hi", self.ocr.verified_languages())

    def test_switching_language_does_not_throw_the_engine_away(self) -> None:
        """Rebuilding the extractor is how language switching works - it must be free."""
        english = self.ocr.DocumentExtractor(language="English")
        tamil = self.ocr.DocumentExtractor(language="Tamil / \u0ba4\u0bae\u0bbf\u0bb4\u0bcd")
        again = self.ocr.DocumentExtractor(language="English")
        self.assertIs(english._ocr, again._ocr)
        self.assertIsNot(english._ocr, tamil._ocr)

    def test_a_failure_is_retried_when_the_operator_asks_again(self) -> None:
        backend = self.ocr.PaddleOCRBackend.shared("en")
        backend.failure = "the network was down"
        self.assertIn("not usable", self.ocr.DocumentExtractor(language="English").status())
        backend.reset()
        self.assertIsNone(backend.failure)

    def test_prefetch_reports_every_language_and_never_raises(self) -> None:
        outcomes = self.ocr.prefetch(["English", "Tamil / \u0ba4\u0bae\u0bbf\u0bb4\u0bcd"])
        self.assertEqual([code for code, _ok, _message in outcomes], ["en", "ta"])
        for _code, ok, message in outcomes:
            self.assertIsInstance(ok, bool)
            self.assertTrue(message)

    def test_the_listing_command_runs_without_touching_the_network(self) -> None:
        self._download("PP-OCRv5_server_det", "latin_PP-OCRv5_mobile_rec")
        self.assertEqual(self.ocr._main(["--list"]), 0)


class TestOllamaEngine(unittest.TestCase):
    """The local LLM client, against a stand-in server."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), MockOllamaHandler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.host = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def engine(self, model: str = "llama3.2:latest") -> OllamaEngine:
        return OllamaEngine(host=self.host, model=model)

    def test_availability_and_models(self) -> None:
        engine = self.engine()
        self.assertTrue(engine.available())
        self.assertIn("llama3.2:latest", engine.models())
        self.assertIn("ready", engine.status())

    def test_missing_model_is_reported_with_the_pull_command(self) -> None:
        status = self.engine("not-pulled").status()
        self.assertIn("not pulled", status)
        self.assertIn("ollama pull not-pulled", status)

    def test_analysis_returns_a_structured_opinion(self) -> None:
        opinion = self.engine().analyze("Worker on the scaffold, lanyard not anchored.")
        self.assertTrue(opinion.ok)
        self.assertTrue(opinion.sif_potential)
        self.assertEqual(opinion.iogp_rule, "Working at Height")
        self.assertIn("lanyard", opinion.rationale.lower())

    def test_translation_returns_english(self) -> None:
        english = self.engine().translate("பம்ப் நிலையத்தில் அபாயகரமான சம்பவம்")
        self.assertIn("pump station", english.lower())

    def test_unreachable_host_degrades_quietly(self) -> None:
        engine = OllamaEngine(host="http://127.0.0.1:9", model="llama3.2")
        self.assertFalse(engine.available())
        self.assertIn("not reachable", engine.status())
        opinion = engine.analyze("no harness on the scaffold")
        self.assertFalse(opinion.ok)
        self.assertTrue(opinion.error)
        self.assertEqual(engine.translate("ஒரு அறிக்கை"), "")

    def test_replies_wrapped_in_prose_are_still_parsed(self) -> None:
        opinion = OllamaEngine._parse(
            'Sure! {"sif_potential": "yes", "rule": "Hot Work"} hope that helps')
        self.assertTrue(opinion.sif_potential)
        self.assertEqual(opinion.iogp_rule, "Hot Work")

    def test_unparsable_reply_is_an_error_not_a_crash(self) -> None:
        opinion = OllamaEngine._parse("I cannot answer that")
        self.assertFalse(opinion.ok)
        self.assertIn("unparsable", opinion.error)

    def test_script_detection_decides_when_to_translate(self) -> None:
        self.assertTrue(looks_non_latin("பம்ப் நிலையத்தில் 11 kV ஊட்டி"))
        self.assertTrue(looks_non_latin("पंप स्टेशन पर 11 kV फीडर"))
        self.assertFalse(looks_non_latin("No harness worn on the scaffold at 6 m."))
        self.assertFalse(looks_non_latin("11 kV / LOTO / PTW"))


class TestPipelineWithLLM(unittest.TestCase):
    """The LLM is additive: attaching it changes no existing verdict."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), MockOllamaHandler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.host = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def pipeline(self) -> SIFPipeline:
        return SIFPipeline(encoder=HashingEncoder())

    def test_verdict_is_unchanged_when_the_llm_is_attached(self) -> None:
        report = ("Near miss at Pump Station No. 3: the 11 kV feeder was left ungrounded "
                  "and no LOTO was applied.")
        without = self.pipeline().analyze(report)
        pipeline = self.pipeline()
        pipeline.attach_llm(OllamaEngine(host=self.host, model="llama3.2:latest"))
        with_llm = pipeline.analyze(report)
        for field in ("sif_potential", "iogp_rule", "risk_score", "barrier_failure"):
            self.assertEqual(getattr(without, field), getattr(with_llm, field))
        self.assertTrue(with_llm.llm_active)

    def test_llm_disagreement_reaches_the_review_queue(self) -> None:
        pipeline = self.pipeline()
        pipeline.attach_llm(OllamaEngine(host=self.host, model="llama3.2:latest"))
        # The mock always answers "SIF"; this report is housekeeping.
        result = pipeline.analyze("Loose chequered plate at the canteen entrance. "
                                  "Minor housekeeping issue, no injury.")
        self.assertFalse(result.sif_potential)
        self.assertTrue(result.llm_flag)
        self.assertEqual(result.review_trigger, "LLM disagreement")

    def test_unreachable_llm_leaves_the_result_untouched(self) -> None:
        pipeline = self.pipeline()
        pipeline.attach_llm(OllamaEngine(host="http://127.0.0.1:9", model="llama3.2"))
        result = pipeline.analyze("No harness worn on the scaffold at 6 m.")
        self.assertFalse(result.llm_active)
        self.assertIn("error", result.evidence.get("llm", {}))
        self.assertTrue(result.sif_potential)   # the pipeline still decided

    def test_translated_report_keeps_the_original_as_evidence(self) -> None:
        tamil = "பம்ப் நிலையத்தில் 11 kV ஊட்டி கம்பி கிரவுண்ட் செய்யப்படவில்லை; LOTO இல்லை."
        engine = OllamaEngine(host=self.host, model="llama3.2:latest")
        english = engine.translate(tamil)
        result = self.pipeline().analyze(tamil, "TA-1", translated=english,
                                         source_language="Tamil / தமிழ்")
        self.assertTrue(result.sif_potential)
        self.assertEqual(result.iogp_rule, "Energy Isolation")
        self.assertEqual(result.raw_text, tamil)         # the reporter's own words
        self.assertIn("pump station", result.translated_text.lower())
        self.assertEqual(result.source_language, "Tamil / தமிழ்")

    def test_analyze_many_accepts_translations(self) -> None:
        results = self.pipeline().analyze_many(
            ["अहवाल एक", "no harness on the scaffold at 6 m"],
            references=["A", "B"],
            translations=["Report one: no gas test before vessel entry.", ""],
            source_language="Marathi / मराठी")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].source_language, "Marathi / मराठी")
        self.assertEqual(results[1].translated_text, "")

    def test_review_priority_places_disagreements_first(self) -> None:
        from sif.review import PRIORITY_ORDER

        self.assertLess(PRIORITY_ORDER["Disagreement"], PRIORITY_ORDER["LLM disagreement"])
        self.assertLess(PRIORITY_ORDER["LLM disagreement"], PRIORITY_ORDER["Critical risk"])
        self.assertTrue(hasattr(ReviewQueue, "classify"))


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestWorkflowAndInterface(unittest.TestCase):
    """The functional map, and the no-pictograph rule."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_map_covers_every_capability_in_order(self) -> None:
        from ui2.workflow import STAGES, WorkflowMap

        keys = [key for key, _title, _body, _action in STAGES]
        self.assertEqual(keys, ["ingest", "ocr", "translate", "analyse", "dashboard",
                                "hotspots", "review", "train"])
        widget = WorkflowMap()
        self.assertEqual(set(widget.cards), set(keys))

    def test_map_statuses_are_settable(self) -> None:
        from ui2.workflow import WorkflowMap

        widget = WorkflowMap()
        widget.set_status("ocr", "Tamil ready")
        self.assertIn("Tamil ready", widget.cards["ocr"].status.text())

    def test_stage_button_emits_its_key(self) -> None:
        from ui2.workflow import WorkflowMap

        widget = WorkflowMap()
        seen = []
        widget.stage_activated.connect(seen.append)
        widget.cards["train"].button.click()
        self.assertEqual(seen, ["train"])

    def test_the_interface_carries_no_pictographs(self) -> None:
        """Build 2 must render on a workstation with no emoji font."""
        import main2
        from ui2 import components, review, views, workflow

        for module in (main2, components, views, workflow, review):
            with open(module.__file__, encoding="utf-8") as handle:
                source = handle.read()
            found = EMOJI.findall(source)
            self.assertEqual(found, [], f"{os.path.basename(module.__file__)}: {found}")

    def test_dashboard_is_its_own_page(self) -> None:
        from ui2.views import DashboardView, IngestView

        dashboard = DashboardView()
        self.assertTrue(hasattr(dashboard, "tile_total"))
        self.assertFalse(hasattr(dashboard, "input_box"),
                         "ingestion belongs on the ingest page, not the dashboard")
        ingest = IngestView(LANGUAGE_CHOICES, UNSUPPORTED_LANGUAGES)
        self.assertTrue(hasattr(ingest, "input_box"))
        self.assertEqual(ingest.language_box.count(), len(LANGUAGE_CHOICES))

    def test_every_page_can_scroll_rather_than_squeeze(self) -> None:
        """A 1366x768 plant laptop must reach every control on every page."""
        from PyQt6.QtWidgets import QAbstractScrollArea, QScrollArea

        import main2

        window = main2.MainWindow()
        try:
            window.resize(1280, 720)
            for key in window._page_index:
                window.navigate(key)
                page = window.pages.currentWidget()
                scrollers = page.findChildren(QAbstractScrollArea)
                self.assertTrue(scrollers, f"{key}: nothing on this page can scroll")
                if key not in {"hotspots"}:   # a bare table scrolls itself
                    self.assertTrue(page.findChildren(QScrollArea),
                                    f"{key}: stacked content with no scroll area")
        finally:
            window.close()

    def test_the_decision_buttons_are_never_scrolled_out_of_reach(self) -> None:
        """The case detail scrolls; the three decisions stay where the eye expects."""
        from ui2.review import ReviewView

        view = ReviewView()
        inside = view.case_scroll.findChildren(type(view.buttons["confirmed"]))
        self.assertEqual(inside, [], "the decision bar must sit outside the scroll area")
        self.assertIn(view.evidence, view.case_scroll.findChildren(type(view.evidence)))

    def test_the_scroll_helper_sets_the_policies_the_theme_expects(self) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QFrame, QLabel

        from ui2.components import scrollable

        area = scrollable(QLabel("content"), 640)
        self.assertTrue(area.widgetResizable())
        self.assertEqual(area.frameShape(), QFrame.Shape.NoFrame)
        self.assertEqual(area.verticalScrollBarPolicy(),
                         Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.assertEqual(area.widget().minimumHeight(), 640)

    def test_original_build_is_untouched(self) -> None:
        """app.py must keep working exactly as before."""
        import app
        import main as build_one

        self.assertTrue(hasattr(app, "main"))
        self.assertTrue(hasattr(build_one, "MainWindow"))
        with open(app.__file__, encoding="utf-8") as handle:
            self.assertNotIn("main2", handle.read())


def _result(reference: str, text: str, **fields) -> PipelineResult:
    """A minimal analysed result, for tests that only care about review."""
    defaults = dict(sif_potential=False, iogp_rule="Energy Isolation", activity="maintenance",
                    location="Pump station", barrier_failure="none", energy_source="electrical")
    defaults.update(fields)
    return PipelineResult(reference=reference, raw_text=text, **defaults)


class TestReviewDecisions(unittest.TestCase):
    """The record an expert leaves behind, and what training reads back."""

    def setUp(self) -> None:
        self.folder = tempfile.mkdtemp(prefix="sif-review-")
        self.log = DecisionLog(os.path.join(self.folder, "decisions.json"))
        self.critical = _result("NM-1", "Cable left ungrounded with no LOTO applied",
                                sif_potential=True, risk_score=91.0, risk_band="Critical")
        self.thin = _result("NM-2", "Observation raised during the round", confidence=0.1)

    def test_fingerprint_survives_reformatting_but_separates_reports(self) -> None:
        """A decision must follow the report, not its row number or its whitespace."""
        same = _result("NM-1", "  Cable left   ungrounded with no LOTO   applied\n")
        self.assertEqual(fingerprint(self.critical), fingerprint(same))
        self.assertNotEqual(fingerprint(self.critical), fingerprint(self.thin))

    def test_a_decision_reaches_disk_immediately(self) -> None:
        self.log.record(self.critical, "confirmed", reviewer="A. Reviewer", note="no isolation")
        self.assertTrue(os.path.isfile(self.log.path))
        reloaded = DecisionLog(self.log.path).load()
        standing = reloaded.for_result(self.critical)
        self.assertEqual(standing.decision, "confirmed")
        self.assertEqual(standing.reviewer, "A. Reviewer")
        self.assertEqual(standing.note, "no isolation")
        self.assertTrue(standing.decided_at, "the record must carry its own timestamp")

    def test_unclear_is_recorded_but_never_becomes_a_label(self) -> None:
        self.log.record(self.critical, "unclear", reviewer="A")
        self.log.record(self.thin, "rejected", reviewer="A")
        chosen, labels = self.log.labels_for([self.critical, self.thin])
        self.assertEqual([item.reference for item in chosen], ["NM-2"])
        self.assertEqual(labels, [0])
        self.assertEqual(self.log.counts()["decided"], 2)
        self.assertEqual(self.log.counts()["labels"], 1)

    def test_an_unreviewed_report_is_never_guessed_at(self) -> None:
        chosen, labels = self.log.labels_for([self.critical, self.thin])
        self.assertEqual((chosen, labels), ([], []))

    def test_changing_a_decision_supersedes_without_erasing(self) -> None:
        self.log.record(self.critical, "confirmed", reviewer="A")
        self.log.record(self.critical, "rejected", reviewer="B")
        self.assertEqual(self.log.for_result(self.critical).decision, "rejected")
        self.assertEqual(len(self.log.entries), 2, "the first call stays in the trail")
        self.assertEqual(self.log.counts()["revisions"], 1)

    def test_overturning_the_engine_is_recorded_as_such(self) -> None:
        entry = self.log.record(self.critical, "rejected", reviewer="A")
        self.assertTrue(entry.overturns_engine)
        self.assertFalse(self.log.record(self.critical, "confirmed").overturns_engine)

    def test_undo_withdraws_only_the_last_entry(self) -> None:
        self.log.record(self.critical, "confirmed")
        self.log.record(self.thin, "rejected")
        withdrawn = self.log.undo()
        self.assertEqual(withdrawn.reference, "NM-2")
        self.assertIsNone(self.log.for_result(self.thin))
        self.assertIsNotNone(self.log.for_result(self.critical))
        self.log.undo()
        self.assertIsNone(self.log.undo(), "undoing an empty log is not an error")

    def test_export_carries_the_whole_trail(self) -> None:
        self.log.record(self.critical, "confirmed", reviewer="A", note="verified on site")
        self.log.record(self.critical, "rejected", reviewer="B")
        path = self.log.export_csv(os.path.join(self.folder, "trail.csv"))
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["decision"], "confirmed")
        self.assertEqual(rows[0]["note"], "verified on site")
        self.assertIn("fingerprint", rows[0])

    def test_a_read_only_location_is_reported_not_raised(self) -> None:
        """An expert must never believe a decision was saved when it was not."""
        log = DecisionLog(os.path.join(self.folder, "nope", "\0", "decisions.json"))
        log.record(self.critical, "confirmed")
        self.assertFalse(log.saved)
        self.assertEqual(len(log.entries), 1, "it is still held for this session")

    def test_a_corrupt_file_reads_as_empty(self) -> None:
        with open(self.log.path, "w", encoding="utf-8") as handle:
            handle.write("{ not json")
        self.assertEqual(DecisionLog(self.log.path).load().entries, [])

    def test_an_unknown_decision_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ReviewDecision(fingerprint="x", reference="NM-1", decision="probably")

    def test_decided_reports_leave_the_queue(self) -> None:
        results = [self.critical, self.thin]
        queue = ReviewQueue()
        self.assertEqual(len(queue.build(results)), 2)
        self.log.record(self.critical, "confirmed")
        remaining = queue.build(results, skip=self.log.decided())
        self.assertEqual([item.reference for item in remaining], ["NM-2"])
        self.assertEqual(len(queue.build(results)), 2, "the queue itself is unchanged")


class TestEnergyWithoutBarrierTrigger(unittest.TestCase):
    """P(SIF) is energy x barrier, so a missing barrier scores zero - and hides."""

    def test_high_energy_with_no_barrier_reaches_a_person(self) -> None:
        result = _result("NM-3", "Lanyard clipped to the handrail at eighteen metres",
                         high_energy=True, barrier_failed=False, confidence=0.7,
                         iogp_rule="Working at Height", energy_source="Gravity / Fall")
        trigger, reason = ReviewQueue.classify(result)
        self.assertEqual(trigger, "Energy, no barrier")
        self.assertIn("Working at Height", reason)
        self.assertIn("confirm the barrier held", reason)

    def test_a_flagged_report_is_not_queued_twice_for_it(self) -> None:
        result = _result("NM-4", "No LOTO applied to the live feeder", sif_potential=True,
                         high_energy=True, barrier_failed=True, risk_band="Critical")
        self.assertEqual(ReviewQueue.classify(result)[0], "Critical risk")

    def test_an_ordinary_report_still_reaches_nobody(self) -> None:
        result = _result("NM-5", "Loose chequered plate refitted the same morning",
                         high_energy=False, barrier_failed=False, confidence=0.8,
                         risk_band="Low")
        self.assertIsNone(ReviewQueue.classify(result)[0])


class TestSampleReports(unittest.TestCase):
    """The bundled test material must actually exercise what it claims to."""

    FOLDER = "samples"

    def test_every_sample_named_in_the_readme_exists(self) -> None:
        with open(os.path.join(self.FOLDER, "README.md"), encoding="utf-8") as handle:
            readme = handle.read()
        for name in ("near_miss_reports.csv", "shift_log.txt", "permit_observation.pdf",
                     "scanned_uauc_report.png", "multilingual_report.txt"):
            self.assertTrue(os.path.isfile(os.path.join(self.FOLDER, name)), name)
            self.assertIn(name, readme, f"{name} is not documented")

    def test_the_csv_imports_with_its_references(self) -> None:
        from main import read_csv_reports

        narratives, references = read_csv_reports(
            os.path.join(self.FOLDER, "near_miss_reports.csv"))
        self.assertEqual(len(narratives), 18)
        self.assertEqual(references[0], "NM-2601")
        self.assertGreaterEqual(sum(1 for text in narratives if len(text) > 180), 13,
                                "most narratives carry a real amount of detail")
        self.assertGreaterEqual(sum(1 for text in narratives if len(text) < 70), 2,
                                "short narratives are deliberate - they test thin evidence")

    def test_the_sample_corpus_exercises_every_offline_trigger(self) -> None:
        from main import read_csv_reports

        narratives, references = read_csv_reports(
            os.path.join(self.FOLDER, "near_miss_reports.csv"))
        pipeline = SIFPipeline(backend="hashing")
        results = [pipeline.analyze(text, reference=reference)
                   for text, reference in zip(narratives, references)]
        triggers = {item.trigger for item in ReviewQueue().build(results)}
        self.assertEqual(triggers, {"Critical risk", "Thin evidence", "Energy, no barrier"})
        self.assertGreaterEqual(sum(1 for item in results if item.sif_potential), 4)

    def test_the_scan_has_no_text_layer_so_ocr_has_to_run(self) -> None:
        with open(os.path.join(self.FOLDER, "scanned_uauc_report.png"), "rb") as handle:
            header = handle.read(8)
        self.assertEqual(header, b"\x89PNG\r\n\x1a\n")

    def test_the_multilingual_sample_covers_the_languages_claimed(self) -> None:
        with open(os.path.join(self.FOLDER, "multilingual_report.txt"),
                  encoding="utf-8") as handle:
            text = handle.read()
        for language in ("HINDI", "MARATHI", "TAMIL", "TELUGU", "KANNADA"):
            self.assertIn(language, text)
        self.assertTrue(looks_non_latin(text.split("--- TAMIL")[1][:400]))


@unittest.skipUnless(HAS_PYQT, "PyQt6 is not installed")
class TestReviewBench(unittest.TestCase):
    """Driving the review page the way a reviewer does."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        import main2

        # A modal dialog in a headless test hangs the whole run, so record the
        # calls instead of showing them - and let a test assert on them.
        self.dialogs: List[str] = []
        self._real_boxes = (main2.QMessageBox.information, main2.QMessageBox.warning)
        main2.QMessageBox.information = lambda *args, **kw: self.dialogs.append(args[-1])
        main2.QMessageBox.warning = lambda *args, **kw: self.dialogs.append(args[-1])

        self.folder = tempfile.mkdtemp(prefix="sif-bench-")
        self.window = main2.MainWindow()
        # Never touch the operator's real decision log from a test.
        self.window.decisions = DecisionLog(os.path.join(self.folder, "decisions.json"))
        self.window.rows = [
            dict(reference="R-1", raw_text="Cable left ungrounded, no LOTO applied",
                 sif_potential=True, risk_score=95.0, risk_band="Critical",
                 iogp_rule="Energy Isolation", activity="maintenance",
                 location="Pump station", barrier_failure="LOTO", energy_source="electrical",
                 high_energy=True, barrier_failed=True, confidence=0.9),
            dict(reference="R-2", raw_text="Vessel entered with no gas test and no ventilation",
                 sif_potential=True, risk_score=88.0, risk_band="Critical",
                 iogp_rule="Confined Space", activity="entry", location="ETP",
                 barrier_failure="gas test", energy_source="toxic", high_energy=True,
                 barrier_failed=True, confidence=0.8),
            dict(reference="R-3", raw_text="Observation raised", sif_potential=False,
                 risk_score=0.0, risk_band="Low", iogp_rule="Unclassified / General HSE",
                 activity="round", location="site", barrier_failure="none",
                 energy_source="none", confidence=0.05),
        ]
        self.window._refresh()

    def tearDown(self) -> None:
        import main2

        main2.QMessageBox.information, main2.QMessageBox.warning = self._real_boxes
        self.window.close()

    def _grow_corpus(self, copies: int = 4) -> None:
        """Repeat the three fixtures so the corpus is big enough to train on."""
        base = list(self.window.rows)
        for index, row in enumerate(base * copies, start=1):
            self.window.rows.append(
                dict(row, reference=f"X-{index}", raw_text=f"{row['raw_text']} {index}"))
        self.window._refresh()

    def test_the_queue_shows_what_is_outstanding(self) -> None:
        self.assertEqual(self.window.outstanding_reviews, 3)
        self.assertEqual(self.window.review_view.table.rowCount(), 3)
        self.assertIn("3", self.window.review_view.progress.text())

    def test_selecting_a_row_shows_the_whole_case(self) -> None:
        self.window.review_view.select(0)
        view = self.window.review_view
        self.assertEqual(view.reference.text(), "R-1")
        self.assertIn("Energy Isolation", view.fields["rule"]._full_value)
        self.assertIn("Rules:", view.opinions.text())
        self.assertTrue(view.buttons["confirmed"].isEnabled())

    def test_a_decision_leaves_the_queue_and_becomes_a_label(self) -> None:
        view = self.window.review_view
        view.select(0)
        view.reviewer.setText("D. Reviewer")
        view.note.setText("isolation certificate never raised")
        view._decide("confirmed")

        self.assertEqual(self.window.outstanding_reviews, 2)
        entry = self.window.decisions.for_result(self.window._result_at(1))
        self.assertEqual(entry.decision, "confirmed")
        self.assertEqual(entry.reviewer, "D. Reviewer")
        self.assertIn("isolation certificate", entry.note)
        _, labels = self.window.decisions.labels_for(self.window._as_results())
        self.assertEqual(labels, [1])
        self.assertEqual(view.note.text(), "", "the note belongs to one report only")

    def test_deciding_never_skips_the_next_report(self) -> None:
        """The queue rebuild moves the next report up; advancing again would skip it."""
        view = self.window.review_view
        view.select(0)
        seen = [view.reference.text()]
        for _ in range(2):
            view._decide("rejected")
            seen.append(view.reference.text())
        self.assertEqual(len(set(seen[:3])), 3, f"a report was skipped: {seen}")
        self.assertEqual(self.window.outstanding_reviews, 1)

    def test_decided_reports_come_back_when_asked_for(self) -> None:
        view = self.window.review_view
        view.select(0)
        view._decide("confirmed")
        self.assertEqual(view.table.rowCount(), 2)
        view.show_decided.setChecked(True)
        self.assertEqual(view.table.rowCount(), 3)
        self.assertEqual(self.window.outstanding_reviews, 2,
                         "showing them again does not make them outstanding")
        statuses = [self.window.queue_rows[index]["status"] for index in range(3)]
        self.assertIn(DECISION_SHORT["confirmed"], statuses)

    def test_the_trail_keeps_every_entry_including_the_changed_ones(self) -> None:
        view = self.window.review_view
        view.select(0)
        view._decide("confirmed")
        view.show_decided.setChecked(True)
        view.select(0)
        view._decide("rejected")
        self.assertEqual(view.trail_table.rowCount(), 2)
        self.assertEqual(self.window.decisions.counts()["decided"], 1)

    def test_undo_puts_the_report_back_in_the_queue(self) -> None:
        view = self.window.review_view
        view.select(0)
        view._decide("confirmed")
        self.assertEqual(self.window.outstanding_reviews, 2)
        self.window.undo_decision()
        self.assertEqual(self.window.outstanding_reviews, 3)
        self.assertEqual(view.trail_table.rowCount(), 0)

    def test_the_header_counts_outstanding_work_not_decided_work(self) -> None:
        tile = self.window.dashboard.tile_review
        self.window.review_view.select(0)
        self.window.review_view._decide("rejected")
        self.assertEqual(tile._value.text(), "2")
        self.assertIn("1 decided", tile._note.text())

    def test_too_small_a_corpus_is_refused_rather_than_trained(self) -> None:
        started = []
        self.window._start = lambda worker, message: started.append(message)
        self.window.train_model()
        self.assertEqual(started, [], "three reports must not reach a training run")
        self.assertTrue(any("at least four" in text for text in self.dialogs))

    def test_training_prefers_human_labels_once_there_are_enough(self) -> None:
        """Below the threshold it trains on pipeline verdicts, and says which."""
        import main2

        started = []
        self.window._start = lambda worker, message: started.append((worker, message))
        self._grow_corpus()
        self.window.train_model()
        self.assertIn("pipeline verdict", started[-1][1])
        self.assertIsNone(started[-1][0]._labels)

        for result in self.window._as_results():
            self.window.decisions.record(
                result, "confirmed" if result.sif_potential else "rejected", reviewer="D")
        self.window.train_model()
        worker, message = started[-1]
        self.assertIn("reviewed decision", message)
        self.assertGreaterEqual(len(worker._labels), main2.MainWindow.MIN_HUMAN_LABELS)
        self.assertEqual(len(worker._labels), len(worker._results))
        self.assertEqual(worker._label_source, "human review decisions")
        self.assertEqual(set(worker._labels), {0, 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
