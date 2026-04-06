from __future__ import annotations

import base64
import logging
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.prompts.gemma_e2b import asr_prompt
from app.schemas.v1 import (
    ChatRequest,
    ChatResponse,
    ChatChoice,
    ChatChoiceMessage,
    TranslationRequest,
    TranslationResponse,
    VisualQueryResponse,
)

logger = logging.getLogger(__name__)

# Gemma 4 E2B sampling defaults (model card — Best Practices, Sampling Parameters).
_GEMMA_ASR_TEMPERATURE = 1.0
_GEMMA_ASR_TOP_P = 0.95
_GEMMA_ASR_TOP_K = 64


def _mime_to_audio_format(mime_type: str | None) -> str:
    if not mime_type:
        return "wav"
    m = mime_type.split(";")[0].strip().lower()
    if m in ("audio/wav", "audio/x-wav", "audio/wave"):
        return "wav"
    if m in ("audio/mpeg", "audio/mp3"):
        return "mp3"
    if m == "audio/flac":
        return "flac"
    if m in ("audio/webm",):
        return "webm"
    if m in ("audio/mp4", "audio/aac", "audio/m4a"):
        return "aac"
    if m in ("audio/ogg", "audio/opus"):
        return "ogg"
    return "wav"


class VllmClient:
    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(
            base_url=s.vllm_base_url.rstrip("/"),
            api_key=s.vllm_api_key,
            timeout=s.vllm_timeout_seconds,
        )
        self._settings = s

    async def chat(self, body: ChatRequest) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": body.model,
            "messages": [m.model_dump() for m in body.messages],
        }
        if body.temperature is not None:
            kwargs["temperature"] = body.temperature
        if body.max_tokens is not None:
            kwargs["max_tokens"] = body.max_tokens
        if body.top_p is not None:
            kwargs["top_p"] = body.top_p
        if body.frequency_penalty is not None:
            kwargs["frequency_penalty"] = body.frequency_penalty
        if body.presence_penalty is not None:
            kwargs["presence_penalty"] = body.presence_penalty
        if body.stop is not None:
            kwargs["stop"] = body.stop

        resp = await self._client.chat.completions.create(**kwargs)
        choices = []
        for i, c in enumerate(resp.choices):
            msg = c.message
            content = getattr(msg, "content", None)
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            choices.append(
                ChatChoice(
                    index=i,
                    message=ChatChoiceMessage(role=msg.role, content=content),
                    finish_reason=c.finish_reason,
                )
            )
        return ChatResponse(
            id=resp.id,
            created=resp.created,
            model=resp.model,
            choices=choices,
        )

    async def translate(self, req: TranslationRequest) -> TranslationResponse:
        model = req.model or self._settings.default_translate_model or self._settings.default_chat_model
        src = req.src_lang or "auto"
        user = (
            f"Translate the following text from {src} to {req.tgt_lang}. "
            "Reply with only the translated text, no quotes or explanation.\n\n"
            f"{req.text}"
        )
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user}],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        return TranslationResponse(translated_text=text)

    async def visual_query(
        self,
        *,
        image_bytes: bytes,
        mime: str,
        query: str,
        src_lang: str,
        tgt_lang: str,
        model: str | None,
    ) -> VisualQueryResponse:
        m = (
            model
            or self._settings.default_vision_model
            or self._settings.default_chat_model
        )
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Answer in {tgt_lang}. User query (context: src_lang={src_lang}): {query}"
                ),
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        resp = await self._client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": content}],
            temperature=0.4,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return VisualQueryResponse(answer=answer)

    async def summarize_text(
        self,
        *,
        text: str,
        tgt_lang: str,
        model: str,
    ) -> str:
        user = (
            f"Summarize the following document in {tgt_lang}. "
            "Use clear sections if helpful. Be concise.\n\n"
            f"{text[:120_000]}"
        )
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user}],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    async def transcribe_audio(
        self,
        audio: bytes,
        language: str,
        *,
        mime_type: str | None = None,
    ) -> str:
        """ASR via chat/completions with `input_audio`, matching the vLLM Gemma 4 recipe.

        Content order follows the official example (text, then audio):
        https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html#audio-transcription-openai-sdk
        User text uses Gemma 4 E2B ASR prompts from `app/prompts/gemma_e2b.py`.
        """
        model = self._settings.default_audio_model or self._settings.default_chat_model
        fmt = _mime_to_audio_format(mime_type)
        b64 = base64.b64encode(audio).decode("utf-8")
        text_prompt = asr_prompt(language)
        # Same shape as vLLM recipe: text first, then input_audio (base64 + format).
        content: list[dict[str, Any]] = [
            {"type": "text", "text": text_prompt},
            {
                "type": "input_audio",
                "input_audio": {"data": b64, "format": fmt},
            },
        ]
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=_GEMMA_ASR_TEMPERATURE,
            top_p=_GEMMA_ASR_TOP_P,
            extra_body={"top_k": _GEMMA_ASR_TOP_K},
        )
        raw = resp.choices[0].message.content
        if isinstance(raw, list):
            raw = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in raw
            )
        return (raw or "").strip()


@lru_cache
def get_vllm_client() -> VllmClient:
    return VllmClient()
