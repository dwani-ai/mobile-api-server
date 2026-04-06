from functools import lru_cache

from app.adapters.ocr import OcrAdapter, get_ocr_adapter
from app.adapters.pdf import PdfAdapter
from app.adapters.stt import SttAdapter, get_stt_adapter
from app.adapters.tts import TtsAdapter, get_tts_adapter
from app.clients.vllm import VllmClient, get_vllm_client
from app.config import Settings, get_settings


@lru_cache
def _pdf_adapter() -> PdfAdapter:
    return PdfAdapter(get_vllm_client(), get_settings())


def get_pdf_adapter() -> PdfAdapter:
    return _pdf_adapter()


def deps_settings() -> Settings:
    return get_settings()


def deps_vllm() -> VllmClient:
    return get_vllm_client()


def deps_stt() -> SttAdapter:
    return get_stt_adapter()


def deps_tts() -> TtsAdapter:
    return get_tts_adapter()


def deps_ocr() -> OcrAdapter:
    return get_ocr_adapter()
