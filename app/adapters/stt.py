import logging
from functools import lru_cache

from app.clients.vllm import VllmClient, get_vllm_client
from app.config import get_settings

logger = logging.getLogger(__name__)


class SttAdapter:
    async def transcribe(
        self, audio: bytes, language: str, *, mime_type: str | None = None
    ) -> str:
        raise NotImplementedError


class StubSttAdapter(SttAdapter):
    async def transcribe(
        self, audio: bytes, language: str, *, mime_type: str | None = None
    ) -> str:
        _ = mime_type
        logger.info("stub STT language=%s bytes=%s", language, len(audio))
        return f"[stub transcription lang={language}]"


class VllmGemmaSttAdapter(SttAdapter):
    """Speech recognition using vLLM + Gemma 4 E2B ASR prompts (see `app/prompts/gemma_e2b.py`)."""

    def __init__(self, vllm: VllmClient) -> None:
        self._vllm = vllm

    async def transcribe(
        self, audio: bytes, language: str, *, mime_type: str | None = None
    ) -> str:
        return await self._vllm.transcribe_audio(audio, language, mime_type=mime_type)


@lru_cache
def get_stt_adapter() -> SttAdapter:
    impl = get_settings().stt_implementation
    if impl == "vllm_gemma":
        return VllmGemmaSttAdapter(get_vllm_client())
    return StubSttAdapter()
