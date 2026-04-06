from __future__ import annotations

import base64
import json
import logging
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.dwani_languages import language_display_name
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


def _strip_json_fence(raw: str) -> str:
    s = (raw or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    out: list[str] = []
    skip_first = True
    for line in lines:
        if skip_first and line.strip().startswith("```"):
            skip_first = False
            continue
        if line.strip() == "```":
            break
        out.append(line)
    return "\n".join(out).strip()


def _parse_translation_json(content: str, *, expected_len: int) -> list[str]:
    """Parse model output into a list of strings (dwani-api-server-compatible fallbacks)."""
    cleaned = _strip_json_fence(content).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Translation response not JSON, using raw text: %s", cleaned[:200])
        if expected_len == 1:
            return [cleaned]
        raise ValueError("Invalid response format from translation model") from None

    translations: list[str] = []
    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            translations = list(data)
        elif all(isinstance(x, dict) for x in data):
            keys = ("translation", "translated", "tgt", "text", "tr")
            for item in data:
                val = None
                for k in keys:
                    if k in item and isinstance(item[k], str):
                        val = item[k]
                        break
                if val is None:
                    raise ValueError("Invalid response format from translation model")
                translations.append(val)
        else:
            raise ValueError("Invalid response format from translation model")
    elif isinstance(data, str):
        translations = [data]
    elif isinstance(data, dict) and "translations" in data and isinstance(data["translations"], list):
        inner = data["translations"]
        if not all(isinstance(x, str) for x in inner):
            raise ValueError("Invalid response format from translation model")
        translations = list(inner)
    else:
        raise ValueError("Invalid response format from translation model")

    if len(translations) != expected_len:
        raise ValueError("Invalid response format from translation model")
    return translations


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
        """Single batched LLM call returning a JSON array (see dwani-api-server /v1/translate)."""
        model = self._settings.default_translate_model or self._settings.default_chat_model
        src_name = language_display_name(req.src_lang)
        tgt_name = language_display_name(req.tgt_lang)
        n = len(req.sentences)
        if n == 0:
            return TranslationResponse(translations=[])

        system_prompt = (
            f"You are a professional translator. Translate the following list of sentences "
            f"from {src_name} to {tgt_name}. Respond ONLY with a valid JSON array of the "
            "translated sentences in the same order, without any additional text or explanations."
        )
        sentences_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(req.sentences))
        user_prompt = f"Sentences to translate:\n\n{sentences_text}"

        resp = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = (resp.choices[0].message.content or "").strip()
        translations = _parse_translation_json(raw, expected_len=n)
        return TranslationResponse(translations=translations)

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
