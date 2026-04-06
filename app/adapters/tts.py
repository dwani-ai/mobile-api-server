import io
import logging
import wave
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


class TtsAdapter:
    async def synthesize(self, text: str, language: str) -> tuple[bytes, str]:
        """Return (audio_bytes, media_type)."""
        raise NotImplementedError


def _minimal_wav_silence(duration_s: float = 0.2, rate: int = 8000) -> bytes:
    nframes = int(rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * nframes)
    return buf.getvalue()


class StubTtsAdapter(TtsAdapter):
    async def synthesize(self, text: str, language: str) -> tuple[bytes, str]:
        logger.info("stub TTS language=%s len=%s", language, len(text))
        return _minimal_wav_silence(), "audio/wav"


@lru_cache
def get_tts_adapter() -> TtsAdapter:
    impl = get_settings().tts_implementation
    if impl == "stub":
        return StubTtsAdapter()
    return StubTtsAdapter()
