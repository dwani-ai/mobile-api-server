import logging
from functools import lru_cache
from io import BytesIO

from pypdf import PdfReader

from app.config import get_settings

logger = logging.getLogger(__name__)


class OcrAdapter:
    async def extract_text(
        self, file_bytes: bytes, filename: str | None, page_number: int, language: str
    ) -> str:
        raise NotImplementedError


class StubOcrAdapter(OcrAdapter):
    async def extract_text(
        self, file_bytes: bytes, filename: str | None, page_number: int, language: str
    ) -> str:
        logger.info(
            "stub OCR page=%s lang=%s name=%s bytes=%s",
            page_number,
            language,
            filename,
            len(file_bytes),
        )
        return f"[stub OCR text for page {page_number} lang={language}]"


class PdfTextOcrAdapter(OcrAdapter):
    """Extract text from PDF pages using pypdf (not true OCR for scanned images)."""

    async def extract_text(
        self, file_bytes: bytes, filename: str | None, page_number: int, language: str
    ) -> str:
        _ = language
        reader = PdfReader(BytesIO(file_bytes))
        idx = max(0, min(page_number - 1, len(reader.pages) - 1))
        if not reader.pages:
            return ""
        page = reader.pages[idx]
        return page.extract_text() or ""


@lru_cache
def get_ocr_adapter() -> OcrAdapter:
    impl = get_settings().ocr_implementation
    if impl == "pdf_text":
        return PdfTextOcrAdapter()
    if impl == "stub":
        return StubOcrAdapter()
    return StubOcrAdapter()
