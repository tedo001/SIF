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

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = ["ExtractedDocument", "DocumentExtractor", "OCRUnavailable",
           "TEXT_SUFFIXES", "IMAGE_SUFFIXES"]

LOGGER = logging.getLogger(__name__)

TEXT_SUFFIXES = (".txt", ".md", ".log", ".csv", ".tsv")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
PDF_SUFFIXES = (".pdf",)

#: A PDF page with fewer characters than this is treated as scanned, not typed.
TEXT_LAYER_MIN_CHARS = 40
#: Render scale for OCR - 2.0 puts a 12 pt glyph at roughly 24 px tall.
OCR_RENDER_SCALE = 2.0


class OCRUnavailable(RuntimeError):
    """Raised when a document needs OCR and no OCR backend is installed."""


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

    def __init__(self, language: str = "en", use_angle_cls: bool = True) -> None:
        self.language = language
        self.use_angle_cls = use_angle_cls
        self._engine = None
        self._lock = threading.Lock()
        #: Set once a load has failed, so a batch does not retry a download that
        #: cannot succeed - and so the Settings tab can show the real reason.
        self.failure: Optional[str] = None

    @property
    def loaded(self) -> bool:
        """True once the engine has been constructed successfully."""
        return self._engine is not None

    @staticmethod
    def installed() -> bool:
        """True when PaddleOCR *and* its runtime are importable.

        ``paddleocr`` installs cleanly without ``paddlepaddle``, so checking the
        wrapper alone would report a capability that fails at first use.
        """
        try:
            import paddle  # noqa: F401
            import paddleocr  # noqa: F401
        except Exception:  # noqa: BLE001 - a broken install is also "not available"
            return False
        return True

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
        self._ocr = PaddleOCRBackend(language=language) if enable_ocr else None

    # -- capability ---------------------------------------------------------

    def ocr_available(self) -> bool:
        """True when OCR can actually be used for this extractor."""
        return bool(self.enable_ocr and PaddleOCRBackend.installed())

    def status(self) -> str:
        """One line describing the ingestion capability, for the Settings tab."""
        if not self.enable_ocr:
            return "OCR disabled - PDFs read through their text layer only"
        if not PaddleOCRBackend.installed():
            return ("PaddleOCR runtime unavailable - scanned pages cannot be read "
                    "(pip install paddleocr paddlepaddle)")
        if self._ocr is not None and self._ocr.failure:
            return f"PaddleOCR installed but not usable - {self._ocr.failure}"
        if self._ocr is not None and self._ocr.loaded:
            return "PaddleOCR ready - scanned PDFs and images can be read"
        return ("PaddleOCR installed - its models download on first use; "
                "run 'Check OCR availability' to confirm this machine can fetch them")

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
        try:
            self._ocr.load()
        except Exception as exc:  # noqa: BLE001 - report, never raise into the UI
            return False, str(exc)
        return True, "PaddleOCR ready - scanned PDFs and images can be read"

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
