"""Optional local LLM engine (Ollama) - a fourth opinion, never the decision.

The deterministic rules, the sentence encoder and the trained model all reason
over vocabulary the system was given. A local instruction model adds something
none of them can: it reads a narrative the way a person does, and it can read one
written in Tamil, Hindi, Marathi or Kannada and hand back English for the rest of
the pipeline.

Two jobs, both optional:

``translate``
    Non-English narrative in, English out, so the rule and prototype layers -
    which are English - can work on it at all.
``analyze``
    A second reading of the report: SIF potential, IOGP rule, activity, location,
    failed barrier and a one-line rationale, returned as JSON.

Everything here is *additive*. Ollama runs on the operator's own machine
(http://localhost:11434), nothing leaves it, and when it is not running the
console behaves exactly as it does without this module - :meth:`OllamaEngine.available`
is false, no fields are set, and no stage changes its verdict.

The client speaks the Ollama REST API over the standard library, so installing
the ``ollama`` Python package is not required.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["OllamaEngine", "LLMOpinion", "DEFAULT_HOST", "DEFAULT_MODEL",
           "looks_non_latin"]

LOGGER = logging.getLogger(__name__)

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("SIF_LLM_MODEL", "llama3.2")
CONNECT_TIMEOUT = 3.0
GENERATE_TIMEOUT = 120.0

#: Kept deliberately narrow: the model classifies and explains, it does not invent
#: policy. The energy/barrier definition is stated so its answer is comparable
#: with the rule layer's.
ANALYSIS_PROMPT = """You are an HSE analyst for an oil and gas operator.

A report has SIF potential (Serious Injury or Fatality potential) only when BOTH:
  1. a high-energy source is present (gravity/height, electrical, pressure,
     suspended load, vehicle, fire, toxic atmosphere, excavation, thermal), AND
  2. a critical barrier failed, was missing, bypassed or was not verified.
High energy with intact barriers is controlled work. A barrier lapse with no high
energy is a housekeeping issue.

Classify this field report and reply with JSON only, no prose:
{{"sif_potential": true|false, "iogp_rule": "<one of: Working at Height, Energy
Isolation, Line of Fire, Confined Space, Safe Mechanical Lifting, Hot Work,
Driving, Bypassing Safety Controls, Work Authorisation, Excavation & Ground
Disturbance, Well Control & Process Containment, Unclassified>", "activity":
"<short task>", "location": "<site or area>", "barrier_failure": "<the control
that failed, or none>", "rationale": "<one sentence>"}}

REPORT:
{report}
"""

TRANSLATION_PROMPT = """Translate this workplace safety report into English.
Keep every technical term, measurement, equipment name and abbreviation exactly as
written (for example 11 kV, LOTO, PTW, GGS-4). Reply with the translation only, no
commentary.

REPORT:
{report}
"""


def looks_non_latin(text: str, threshold: float = 0.20) -> bool:
    """True when enough of the text sits outside the Latin block to need translation.

    Field reports mix scripts freely - a Tamil narrative still says "11 kV" and
    "LOTO" - so this asks what share of the *letters* are non-Latin rather than
    looking for any single character.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    non_latin = sum(1 for ch in letters if ord(ch) > 0x02FF)
    return non_latin / len(letters) >= threshold


@dataclass
class LLMOpinion:
    """What the model said about one report."""

    sif_potential: Optional[bool] = None
    iogp_rule: str = ""
    activity: str = ""
    location: str = ""
    barrier_failure: str = ""
    rationale: str = ""
    model: str = ""
    elapsed_ms: float = 0.0
    error: str = ""
    raw: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the model returned a usable verdict."""
        return self.error == "" and self.sif_potential is not None

    def to_dict(self) -> Dict[str, object]:
        payload = dict(self.__dict__)
        payload["raw"] = dict(self.raw)
        return payload


class OllamaEngine:
    """Thin, dependency-free client for a local Ollama server.

    Example
    -------
    >>> engine = OllamaEngine(model="llama3.2")
    >>> if engine.available():                      # doctest: +SKIP
    ...     opinion = engine.analyze("No harness worn on the scaffold at 6 m.")
    ...     opinion.sif_potential
    True
    """

    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL,
                 timeout: float = GENERATE_TIMEOUT) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.last_error = ""

    # -- connectivity ------------------------------------------------------

    def _request(self, path: str, payload: Optional[dict] = None,
                 timeout: Optional[float] = None) -> dict:
        """POST (or GET) JSON to the Ollama API and return the decoded body."""
        url = f"{self.host}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data else "GET")
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def available(self) -> bool:
        """True when a server answers on the configured host."""
        try:
            self._request("/api/tags", timeout=CONNECT_TIMEOUT)
            self.last_error = ""
            return True
        except Exception as exc:  # noqa: BLE001 - any failure means "not available"
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def models(self) -> List[str]:
        """Names of the models the server has pulled."""
        try:
            body = self._request("/api/tags", timeout=CONNECT_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        return [str(item.get("name", "")) for item in body.get("models", [])
                if item.get("name")]

    def status(self) -> str:
        """One line for the interface."""
        if not self.available():
            return (f"Ollama not reachable at {self.host} - the console runs without it "
                    f"({self.last_error})")
        installed = self.models()
        if self.model not in installed:
            return (f"Ollama up at {self.host}, but '{self.model}' is not pulled. "
                    f"Run: ollama pull {self.model}"
                    + (f"  (available: {', '.join(installed[:4])})" if installed else ""))
        return f"Ollama ready at {self.host} - model '{self.model}'"

    # -- generation --------------------------------------------------------

    def _generate(self, prompt: str, json_mode: bool = False) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if json_mode:
            payload["format"] = "json"
        body = self._request("/api/generate", payload)
        return str(body.get("response", "")).strip()

    def translate(self, text: str) -> str:
        """Translate a non-English narrative to English; returns '' on failure."""
        if not text.strip():
            return ""
        try:
            return self._generate(TRANSLATION_PROMPT.format(report=text.strip()))
        except Exception as exc:  # noqa: BLE001 - translation is best-effort
            self.last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Ollama translation failed: %s", self.last_error)
            return ""

    def analyze(self, text: str) -> LLMOpinion:
        """Ask the model to classify one report. Never raises."""
        import time

        started = time.perf_counter()
        if not text.strip():
            return LLMOpinion(error="empty report", model=self.model)
        try:
            raw = self._generate(ANALYSIS_PROMPT.format(report=text.strip()),
                                 json_mode=True)
        except Exception as exc:  # noqa: BLE001 - the console must survive this
            message = f"{type(exc).__name__}: {exc}"
            self.last_error = message
            LOGGER.warning("Ollama analysis failed: %s", message)
            return LLMOpinion(error=message, model=self.model,
                              elapsed_ms=round((time.perf_counter() - started) * 1000, 1))

        opinion = self._parse(raw)
        opinion.model = self.model
        opinion.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return opinion

    @staticmethod
    def _parse(raw: str) -> LLMOpinion:
        """Turn the model's reply into an opinion, tolerating stray prose."""
        text = raw.strip()
        if not text:
            return LLMOpinion(error="empty response")
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        try:
            body = json.loads(text)
        except json.JSONDecodeError as exc:
            return LLMOpinion(error=f"unparsable response: {exc}", raw={"response": raw})
        if not isinstance(body, dict):
            return LLMOpinion(error="response was not an object", raw={"response": raw})

        flag = body.get("sif_potential")
        if isinstance(flag, str):
            flag = flag.strip().lower() in {"true", "yes", "1", "sif", "sif-potential"}
        elif not isinstance(flag, bool):
            flag = None

        def field_text(*names: str) -> str:
            for name in names:
                value = body.get(name)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        return LLMOpinion(
            sif_potential=flag,
            iogp_rule=field_text("iogp_rule", "rule"),
            activity=field_text("activity", "task"),
            location=field_text("location", "site"),
            barrier_failure=field_text("barrier_failure", "barrier"),
            rationale=field_text("rationale", "reason", "explanation"),
            raw=body,
        )
