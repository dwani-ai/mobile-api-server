import logging
from io import BytesIO

from fastapi import HTTPException
from pypdf import PdfReader
from pypdf.errors import PdfStreamError

from app.clients.vllm import VllmClient
from app.config import Settings

logger = logging.getLogger(__name__)


class PdfAdapter:
    def __init__(self, vllm: VllmClient, settings: Settings) -> None:
        self._vllm = vllm
        self._settings = settings

    async def summarize_pdf(self, file_bytes: bytes, tgt_lang: str, model: str) -> str:
        try:
            reader = PdfReader(BytesIO(file_bytes))
        except PdfStreamError as e:
            logger.warning("invalid pdf: %s", e)
            raise HTTPException(
                status_code=400,
                detail="Invalid or unreadable PDF",
            ) from e
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            parts.append(t)
        text = "\n\n".join(parts)
        if not text.strip():
            text = "[empty or image-only PDF; stub summary]"
        m = model or self._settings.default_summary_model or self._settings.default_chat_model
        return await self._vllm.summarize_text(text=text, tgt_lang=tgt_lang, model=m)
