"""Tests for the second build: languages, the local LLM, and the workflow map.

These cover what ``app2.py`` adds. The original build keeps its own suite in
``test_sif.py``; both must pass, since the two share the analysis stack.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from sif.encoders import HashingEncoder
from sif.llm import OllamaEngine, looks_non_latin
from sif.ocr import (LANGUAGE_CHOICES, LANGUAGES, UNSUPPORTED_LANGUAGES,
                     DocumentExtractor, resolve_language)
from sif.pipeline import SIFPipeline
from sif.review import ReviewQueue

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
        from ui2 import components, views, workflow

        for module in (main2, components, views, workflow):
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

    def test_original_build_is_untouched(self) -> None:
        """app.py must keep working exactly as before."""
        import app
        import main as build_one

        self.assertTrue(hasattr(app, "main"))
        self.assertTrue(hasattr(build_one, "MainWindow"))
        with open(app.__file__, encoding="utf-8") as handle:
            self.assertNotIn("main2", handle.read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
