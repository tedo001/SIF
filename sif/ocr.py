"""Document ingestion - reading reports that arrive as files, not text.

Field reports rarely arrive as clean CSV rows: they are scanned shift logs,
photographed permit forms, exported PDFs and Word-printed observation sheets.
This module turns any of those into the plain text the pipeline consumes.

Backends, tried in order and recorded on the result so the operator knows what
read the document:

``text``
    Plain ``.txt`` / ``.md`` / ``.csv`` - read directly.
``pdf-text``
    PDFs that carry a text layer - extracted without OCR, which is faster and
    exact.
``paddleocr``
    Scanned PDFs and images - `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_
    with angle classification, which handles the rotated phone photographs that
    make up most site-submitted evidence. Per-line confidence is averaged into
    :attr:`ExtractedDocument.confidence` so a bad scan can be routed to a human
    instead of silently producing garbage.

PaddleOCR is optional: when it is not installed, image input reports a clear,
actionable error and PDFs still work through their text layer.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["ExtractedDocument", "DocumentExtractor", "OCRUnavailable",
           "PaddleOCRBackend", "TEXT_SUFFIXES", "IMAGE_SUFFIXES", "LANGUAGES",
           "LANGUAGE_CHOICES", "UNSUPPORTED_LANGUAGES", "resolve_language",
           "cache_directory", "cached_models", "models_present", "prefetch",
           "verified_languages", "remember_verified"]

LOGGER = logging.getLogger(__name__)

TEXT_SUFFIXES = (".txt", ".md", ".log", ".csv", ".tsv")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
PDF_SUFFIXES = (".pdf",)

#: A PDF page with fewer characters than this is treated as scanned, not typed.
TEXT_LAYER_MIN_CHARS = 40
#: Render scale for OCR - 2.0 puts a 12 pt glyph at roughly 24 px tall.
OCR_RENDER_SCALE = 2.0


#: Languages this build of PaddleOCR can recognise, as
#: ``friendly name -> (paddle language code, script family)``.
#:
#: PaddleOCR groups most Indian languages by *script*: one Devanagari model reads
#: Hindi, Marathi, Nepali, Sanskrit, Bhojpuri, Maithili and Konkani, because they
#: share glyphs. Tamil, Telugu and Kannada have their own recognisers. The codes
#: below were read from the installed package rather than assumed - passing a code
#: it does not know makes PaddleOCR fail at model resolution, not at read time.
LANGUAGES: Dict[str, Tuple[str, str]] = {
    "English": ("en", "Latin"),
    "Hindi / हिन्दी": ("hi", "Devanagari"),
    "Marathi / मराठी": ("mr", "Devanagari"),
    "Nepali / नेपाली": ("ne", "Devanagari"),
    "Sanskrit / संस्कृत": ("sa", "Devanagari"),
    "Bhojpuri / भोजपुरी": ("bho", "Devanagari"),
    "Maithili / मैथिली": ("mai", "Devanagari"),
    "Konkani / कोंकणी": ("gom", "Devanagari"),
    "Tamil / தமிழ்": ("ta", "Tamil"),
    "Telugu / తెలుగు": ("te", "Telugu"),
    "Kannada / ಕನ್ನಡ": ("ka", "Kannada"),
    "Urdu / اردو": ("ur", "Arabic"),
}

#: Presented in the interface in this order - English first, then by usage.
LANGUAGE_CHOICES: Tuple[str, ...] = tuple(LANGUAGES)

#: Indian languages this PaddleOCR build has no recogniser for. Naming them keeps
#: the interface honest instead of silently reading them with the wrong model.
UNSUPPORTED_LANGUAGES: Tuple[str, ...] = (
    "Bengali", "Gujarati", "Punjabi / Gurmukhi", "Malayalam", "Odia", "Assamese",
)


def resolve_language(value: str) -> str:
    """Return the PaddleOCR code for a friendly name, code, or ISO-ish input.

    Accepts ``"Tamil / தமிழ்"``, ``"tamil"``, ``"ta"`` and ``"kn"`` (the ISO code
    for Kannada, which PaddleOCR spells ``ka``). Unknown values fall back to
    English rather than raising, because a wrong dropdown entry should not stop a
    batch - the result records which language was actually used.
    """
    if not value:
        return "en"
    text = str(value).strip()
    if text in LANGUAGES:
        return LANGUAGES[text][0]
    lowered = text.lower()
    aliases = {"kn": "ka", "hin": "hi", "tam": "ta", "tel": "te", "kan": "ka",
               "mar": "mr", "urd": "ur", "eng": "en", "devanagari": "hi"}
    if lowered in aliases:
        return aliases[lowered]
    for name, (code, _script) in LANGUAGES.items():
        if lowered == code or lowered == name.split(" / ")[0].lower():
            return code
    LOGGER.warning("Unknown OCR language %r - falling back to English", value)
    return "en"


class OCRUnavailable(RuntimeError):
    """Raised when a document needs OCR and no OCR backend is installed."""


# ---------------------------------------------------------------------------
# The model cache - "download once, then never again"
# ---------------------------------------------------------------------------
#
# PaddleOCR downloads its detection, orientation and recognition models the first
# time an engine is constructed, and keeps them in a cache directory on the
# machine. They are not reinstalled per run, per language or per session - but
# the console used to have no way of knowing they were there, so every start-up
# told the operator that models "download on first use" and asked them to check
# again. These helpers look at the disk instead, so a machine that has already
# fetched them says so and stays saying so.


def cache_directory() -> str:
    """The directory PaddleOCR keeps downloaded models in, on this machine.

    Mirrors PaddleX's own resolution (``PADDLE_PDX_CACHE_HOME``, else
    ``~/.paddlex``) rather than importing it, because importing paddlex costs
    the very seconds this check exists to save.
    """
    return os.environ.get("PADDLE_PDX_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".paddlex")


def _model_directories() -> List[str]:
    """Every place a downloaded model may sit, newest layout first."""
    return [
        os.path.join(cache_directory(), "official_models"),   # PaddleX 3.x
        os.path.join(os.path.expanduser("~"), ".paddleocr", "whl"),  # PaddleOCR 2.x
    ]


def cached_models() -> List[str]:
    """Names of the OCR models already downloaded to this machine."""
    found: List[str] = []
    for directory in _model_directories():
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            path = os.path.join(directory, entry)
            # A model is a directory with weights in it; a half-finished download
            # leaves an empty folder, which must not count as present.
            if not os.path.isdir(path):
                continue
            with os.scandir(path) as contents:
                if any(True for _ in contents):
                    found.append(entry)
    return found


def verified_languages() -> Dict[str, str]:
    """Language codes this machine has actually run OCR with, and when.

    Written by :meth:`DocumentExtractor.probe` after a load that really worked.
    It is a convenience for the status line, never a substitute for the disk
    check: a cleared cache makes the models absent whatever this file says.
    """
    from . import prefs

    stored = prefs.get("ocr_verified", {})
    return {str(key): str(value) for key, value in stored.items()} \
        if isinstance(stored, dict) else {}


def remember_verified(language: str) -> None:
    """Record that OCR loaded successfully for ``language`` on this machine."""
    from . import prefs

    stored = verified_languages()
    stored[language] = datetime.now().isoformat(timespec="seconds")
    prefs.set_value("ocr_verified", stored)
    LOGGER.info("Recorded OCR as verified for %r (models in %s)",
                language, cache_directory())


def models_present() -> bool:
    """True when a usable set of models is already on this machine.

    Usable means both halves of the pipeline: something that finds text and
    something that reads it. Model naming changes between PaddleOCR versions, so
    this matches on the role in the name rather than on an exact list, and never
    claims readiness from a single downloaded file.
    """
    names = [name.lower() for name in cached_models()]
    detection = any("det" in name for name in names)
    recognition = any("rec" in name for name in names)
    return bool(detection and recognition)


@dataclass
class ExtractedDocument:
    """Text recovered from one file, with the provenance of the extraction."""

    path: str
    text: str = ""
    backend: str = ""
    pages: int = 0
    confidence: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def blocks(self) -> List[str]:
        """Split the document into report-sized blocks on blank lines."""
        raw = [block.strip() for block in self.text.split("\n\n")]
        blocks = [block for block in raw if len(block) > 25]
        return blocks or ([self.text.strip()] if self.text.strip() else [])

    def to_dict(self) -> dict:
        return {"path": self.path, "backend": self.backend, "pages": self.pages,
                "confidence": self.confidence, "characters": len(self.text),
                "warnings": list(self.warnings)}


class PaddleOCRBackend:
    """Thin wrapper around PaddleOCR with lazy, thread-safe initialisation.

    The engine is built on first use because constructing it downloads the
    detection, classification and recognition models (~10 MB) and takes a few
    seconds - work that belongs on a worker thread, never in ``__init__``.
    """

    #: One backend per (language, orientation) for the life of the process. The
    #: engine costs seconds to build, and the extractor is rebuilt whenever the
    #: operator changes language - without this, switching Tamil to English and
    #: back would construct three engines and load the models three times.
    _shared: Dict[tuple, "PaddleOCRBackend"] = {}
    _shared_lock = threading.Lock()

    def __init__(self, language: str = "en", use_angle_cls: bool = True) -> None:
        self.language = language
        self.use_angle_cls = use_angle_cls
        self._engine = None
        self._lock = threading.Lock()
        #: Set once a load has failed, so a batch does not retry a download that
        #: cannot succeed - and so the Settings tab can show the real reason.
        self.failure: Optional[str] = None

    @classmethod
    def shared(cls, language: str = "en", use_angle_cls: bool = True) -> "PaddleOCRBackend":
        """The backend for this language, reused for the life of the process."""
        key = (language, bool(use_angle_cls))
        with cls._shared_lock:
            backend = cls._shared.get(key)
            if backend is None:
                backend = cls(language=language, use_angle_cls=use_angle_cls)
                cls._shared[key] = backend
            return backend

    def reset(self) -> None:
        """Forget a previous failure so the next load genuinely retries.

        A download that failed because the machine was offline must not be
        remembered as a permanent verdict once it is back on the network.
        """
        with self._lock:
            self.failure = None

    @property
    def loaded(self) -> bool:
        """True once the engine has been constructed successfully."""
        return self._engine is not None

    @staticmethod
    def installed() -> bool:
        """True when PaddleOCR *and* its runtime are present.

        ``paddleocr`` installs cleanly without ``paddlepaddle``, so checking the
        wrapper alone would report a capability that fails at first use - both
        names are resolved.

        They are resolved with :func:`importlib.util.find_spec`, which locates a
        module without executing it. Importing ``paddle`` for real costs around
        twenty seconds on Windows and prints compiler-toolchain warnings, and
        this runs while the window is opening: the Settings tab asks it for a
        status line. An install that is present but broken is caught by
        :meth:`load`, which reports it through :class:`OCRUnavailable`.
        """
        try:
            return all(importlib.util.find_spec(name) is not None
                       for name in ("paddle", "paddleocr"))
        except Exception:  # noqa: BLE001 - a broken install is also "not available"
            return False

    def load(self) -> None:
        """Instantiate the OCR engine (downloads models on first run)."""
        with self._lock:
            if self._engine is not None:
                return
            if self.failure:
                raise OCRUnavailable(self.failure)
            from paddleocr import PaddleOCR

            LOGGER.info("Initialising PaddleOCR (lang=%s, orientation=%s)",
                        self.language, self.use_angle_cls)
            # The constructor signature changed across major versions and unknown
            # keywords raise, so try the known shapes newest-first.
            candidates = (
                {"lang": self.language, "use_textline_orientation": self.use_angle_cls},
                {"lang": self.language, "use_angle_cls": self.use_angle_cls,
                 "show_log": False},
                {"lang": self.language},
            )
            errors = []
            for keywords in candidates:
                try:
                    self._engine = PaddleOCR(**keywords)
                    LOGGER.info("PaddleOCR ready (%s)", ", ".join(sorted(keywords)))
                    return
                except (TypeError, ValueError) as exc:
                    # An unknown keyword only means "wrong version" - keep trying.
                    errors.append(f"{sorted(keywords)}: {exc}")
                except Exception as exc:  # noqa: BLE001 - model download, disk, driver
                    self.failure = (
                        f"PaddleOCR could not start: {exc}. Its models download on first "
                        "use, so an offline machine needs them pre-fetched into "
                        "~/.paddlex (or set PADDLE_PDX_MODEL_SOURCE to a reachable host).")
                    LOGGER.warning("%s", self.failure)
                    raise OCRUnavailable(self.failure) from exc
            self.failure = "Could not construct PaddleOCR - " + " | ".join(errors)
            raise OCRUnavailable(self.failure)

    def read(self, image_path: str):
        """Return ``(text, mean_confidence)`` for one image."""
        if self._engine is None:
            self.load()
        raw = self._engine.ocr(image_path)
        lines: List[str] = []
        scores: List[float] = []
        for entry in self._flatten(raw):
            text, score = entry
            if text:
                lines.append(text)
                if score is not None:
                    scores.append(float(score))
        confidence = round(sum(scores) / len(scores), 3) if scores else None
        return "\n".join(lines), confidence

    @staticmethod
    def _flatten(raw) -> List[tuple]:
        """Normalise the several shapes PaddleOCR has returned across versions."""
        results: List[tuple] = []
        if raw is None:
            return results
        # 3.x: list of dicts with rec_texts / rec_scores.
        if isinstance(raw, dict):
            raw = [raw]
        for page in raw:
            if isinstance(page, dict):
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                results.extend(zip(texts, list(scores) + [None] * len(texts)))
                continue
            if not page:
                continue
            for line in page:
                # 2.x: [box, (text, score)]
                try:
                    payload = line[1]
                    if isinstance(payload, (list, tuple)):
                        results.append((str(payload[0]), payload[1] if len(payload) > 1 else None))
                    else:
                        results.append((str(payload), None))
                except (IndexError, TypeError):
                    continue
        return results


class DocumentExtractor:
    """Reads text out of report files, choosing the cheapest backend that works.

    Example
    -------
    >>> extractor = DocumentExtractor()
    >>> document = extractor.extract("shift_log.pdf")     # doctest: +SKIP
    >>> document.backend                                   # doctest: +SKIP
    'pdf-text'
    """

    def __init__(self, language: str = "en", enable_ocr: bool = True) -> None:
        self.enable_ocr = enable_ocr
        #: The friendly name as chosen, and the code actually handed to PaddleOCR.
        self.language_name = language
        self.language = resolve_language(language)
        self._ocr = PaddleOCRBackend.shared(self.language) if enable_ocr else None

    # -- capability ---------------------------------------------------------

    def ocr_available(self) -> bool:
        """True when OCR can actually be used for this extractor."""
        return bool(self.enable_ocr and PaddleOCRBackend.installed())

    def status(self) -> str:
        """One line describing the ingestion capability, for the Settings tab.

        The models are downloaded once per machine and kept on disk, so this
        looks at the disk rather than at whether an engine happens to be loaded
        in this session. A machine that has already fetched them is told so at
        every start-up, instead of being asked to check again forever.
        """
        if not self.enable_ocr:
            return "OCR disabled - PDFs read through their text layer only"
        if not PaddleOCRBackend.installed():
            return ("PaddleOCR runtime unavailable - scanned pages cannot be read "
                    "(pip install paddleocr paddlepaddle)")
        if self._ocr is not None and self._ocr.failure:
            return f"PaddleOCR installed but not usable - {self._ocr.failure}"
        script = LANGUAGES.get(self.language_name, (self.language, "Latin"))[1]
        label = f"{self.language_name} [{self.language}, {script} model]"
        if self._ocr is not None and self._ocr.loaded:
            return f"PaddleOCR ready - reading {label}"
        verified = verified_languages().get(self.language)
        if verified:
            return (f"PaddleOCR ready for {label} - models are on this machine "
                    f"({cache_directory()}), verified {verified[:10]}. No download needed.")
        if models_present():
            return (f"PaddleOCR installed for {label} - models already cached in "
                    f"{cache_directory()}; the {script} recogniser downloads once if it "
                    "is not among them")
        return (f"PaddleOCR installed for {label}; the models download once per machine "
                "on first use - run 'Check OCR availability' to fetch them now")

    # -- extraction ---------------------------------------------------------

    def probe(self) -> Tuple[bool, str]:
        """Try to bring the OCR engine up and report the outcome.

        This is what the Settings button calls: the only way to know whether a
        machine can actually OCR is to load the models, which downloads them on
        first use - so it runs on a worker thread, never on the GUI thread.
        """
        if not self.enable_ocr:
            return False, "OCR is disabled in Settings"
        if not PaddleOCRBackend.installed():
            return False, ("PaddleOCR runtime not installed "
                           "(pip install paddleocr paddlepaddle)")
        # A previous failure was probably a network that has since come back;
        # pressing the button must mean "try again", not "repeat the verdict".
        self._ocr.reset()
        try:
            self._ocr.load()
        except Exception as exc:  # noqa: BLE001 - report, never raise into the UI
            return False, str(exc)
        remember_verified(self.language)
        return True, (f"PaddleOCR ready - models are on this machine "
                      f"({cache_directory()}) and will not download again")

    def extract(self, path: str) -> ExtractedDocument:
        """Read one file and return its text with the backend that produced it."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        suffix = os.path.splitext(path)[1].lower()
        if suffix in TEXT_SUFFIXES:
            return self._read_text(path)
        if suffix in PDF_SUFFIXES:
            return self._read_pdf(path)
        if suffix in IMAGE_SUFFIXES:
            return self._read_image(path)
        raise ValueError(f"Unsupported file type '{suffix}' for {os.path.basename(path)}")

    def extract_many(self, paths: Sequence[str]) -> List[ExtractedDocument]:
        """Extract several files, recording failures as warnings rather than raising."""
        documents: List[ExtractedDocument] = []
        for path in paths:
            try:
                documents.append(self.extract(path))
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop a batch
                LOGGER.warning("Could not read %s: %s", path, exc)
                documents.append(ExtractedDocument(path=path, backend="failed",
                                                   warnings=[str(exc)]))
        return documents

    # -- backends -----------------------------------------------------------

    @staticmethod
    def _read_text(path: str) -> ExtractedDocument:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        LOGGER.info("Read %s as plain text (%d characters)", os.path.basename(path), len(text))
        return ExtractedDocument(path=path, text=text, backend="text", pages=1)

    def _read_pdf(self, path: str) -> ExtractedDocument:
        """Prefer the embedded text layer; fall back to OCR for scanned pages."""
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Reading PDFs requires pypdfium2 (pip install pypdfium2)") from exc

        document = pdfium.PdfDocument(path)
        page_texts: List[str] = []
        scanned_pages: List[int] = []
        try:
            for number, page in enumerate(document):
                text = page.get_textpage().get_text_range() or ""
                page_texts.append(text)
                if len(text.strip()) < TEXT_LAYER_MIN_CHARS:
                    scanned_pages.append(number)

            result = ExtractedDocument(
                path=path, text="\n\n".join(page_texts).strip(),
                backend="pdf-text", pages=len(page_texts))

            if scanned_pages and self.ocr_available():
                LOGGER.info("%d page(s) of %s have no text layer - running OCR",
                            len(scanned_pages), os.path.basename(path))
                try:
                    ocr_text, confidence = self._ocr_pages(document, scanned_pages)
                except OCRUnavailable as exc:
                    # A page without text is still worth returning; say what failed.
                    result.warnings.append(str(exc))
                    ocr_text, confidence = "", None
                if ocr_text:
                    result.text = (result.text + "\n\n" + ocr_text).strip()
                    result.backend = "pdf-text+paddleocr" if any(
                        len(item.strip()) >= TEXT_LAYER_MIN_CHARS for item in page_texts
                    ) else "paddleocr"
                    result.confidence = confidence
            elif scanned_pages:
                result.warnings.append(
                    f"{len(scanned_pages)} page(s) have no text layer and OCR is unavailable")
        finally:
            document.close()

        LOGGER.info("Read %s via %s (%d pages, %d characters)", os.path.basename(path),
                    result.backend, result.pages, len(result.text))
        return result

    def _ocr_pages(self, document, page_numbers: Sequence[int]):
        """Render the given PDF pages and OCR them."""
        import tempfile

        texts: List[str] = []
        confidences: List[float] = []
        with tempfile.TemporaryDirectory() as folder:
            for number in page_numbers:
                image_path = os.path.join(folder, f"page_{number}.png")
                document[number].render(scale=OCR_RENDER_SCALE).to_pil().save(image_path)
                text, confidence = self._ocr.read(image_path)
                if text:
                    texts.append(text)
                if confidence is not None:
                    confidences.append(confidence)
        mean = round(sum(confidences) / len(confidences), 3) if confidences else None
        return "\n\n".join(texts), mean

    def _read_image(self, path: str) -> ExtractedDocument:
        if not self.ocr_available():
            raise OCRUnavailable(
                "Reading an image needs PaddleOCR. Install it with "
                "'pip install paddleocr paddlepaddle', or enable it in Settings.")
        text, confidence = self._ocr.read(path)
        LOGGER.info("Read %s via PaddleOCR (%d characters, confidence %s)",
                    os.path.basename(path), len(text), confidence)
        return ExtractedDocument(path=path, text=text, backend="paddleocr", pages=1,
                                 confidence=confidence)


# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------

def prefetch(languages: Sequence[str] = ("en",)) -> List[Tuple[str, bool, str]]:
    """Download and verify the OCR models for ``languages``, once.

    This is the deliberate version of what otherwise happens by accident the
    first time somebody opens a scan: it fetches the models, proves they load,
    and records each language as verified, so from then on the console reports
    OCR as ready without touching the network again.

    For a plant machine with no route to the internet, run this on a connected
    machine and copy the whole cache directory (:func:`cache_directory`) across;
    the console reads what it finds there.

    Returns one ``(language, ok, message)`` per requested language, and never
    raises: a language that cannot be fetched is reported, and the rest continue.
    """
    outcomes: List[Tuple[str, bool, str]] = []
    for name in languages:
        code = resolve_language(name)
        extractor = DocumentExtractor(language=name)
        ok, message = extractor.probe()
        outcomes.append((code, ok, message))
        LOGGER.info("Prefetch %s: %s", code, "ready" if ok else message)
    return outcomes


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m sif.ocr [--languages en hi ta]`` - fetch the models once."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m sif.ocr",
        description="Download the PaddleOCR models this machine needs, once.")
    parser.add_argument("languages", nargs="*", default=["en"],
                        help="languages to fetch (names, codes or aliases); default: en")
    parser.add_argument("--list", action="store_true",
                        help="show what is already cached and exit")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(f"Model cache: {cache_directory()}")
    cached = cached_models()
    print(f"Already downloaded: {', '.join(cached) if cached else 'nothing yet'}")
    verified = verified_languages()
    if verified:
        print("Verified languages: "
              + ", ".join(f"{code} ({when[:10]})" for code, when in sorted(verified.items())))
    if arguments.list:
        return 0
    if not PaddleOCRBackend.installed():
        print("PaddleOCR is not installed. Run: pip install paddleocr paddlepaddle")
        return 2

    failures = 0
    for code, ok, message in prefetch(arguments.languages):
        print(f"  {code}: {'ready' if ok else 'FAILED'} - {message}")
        failures += 0 if ok else 1
    print("Done. These models stay on this machine; the console will not download "
          "them again." if not failures else
          f"{failures} language(s) could not be fetched - see the messages above.")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(_main())
