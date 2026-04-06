from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vllm_base_url: str = Field(
        default="http://127.0.0.1:8000/v1",
        validation_alias=AliasChoices("VLLM_BASE_URL", "LITELLM_API_BASE"),
        description="OpenAI-compatible base URL (e.g. vLLM or LiteLLM); must include /v1",
    )
    vllm_api_key: str = Field(default="dummy")
    vllm_timeout_seconds: float = Field(default=120.0)
    default_chat_model: str = Field(default="default")
    default_translate_model: str | None = Field(default=None)
    default_vision_model: str | None = Field(default=None)
    default_summary_model: str | None = Field(default=None)
    default_audio_model: str | None = Field(
        default=None,
        description="Model id for Gemma/vLLM audio ASR (e.g. google/gemma-4-E2B-it)",
    )

    max_upload_mb: int = Field(default=50)
    max_tokens: int = Field(
        default=2048,
        description="Default max_tokens for indic_chat and similar chat completions",
    )
    stt_implementation: str = Field(
        default="stub",
        description="stub | vllm_gemma (Gemma 4 E2B ASR prompts + vLLM chat/completions)",
    )
    tts_implementation: str = Field(default="stub")
    ocr_implementation: str = Field(default="stub")


@lru_cache
def get_settings() -> Settings:
    return Settings()
