"""JSON contracts aligned with the Android Retrofit client; adjust fields to match Kotlin data classes if they differ."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"


# --- Chat (POST v1/indic_chat) ---


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = None


class IndicChatRequest(BaseModel):
    """Body for POST /v1/indic_chat (prompt + languages + model id)."""

    prompt: str
    src_lang: str
    tgt_lang: str
    model: str


class ChatChoiceMessage(BaseModel):
    role: str
    content: str | None = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatChoice] = Field(default_factory=list)


# --- Translation (POST v1/translate) ---


class TranslationRequest(BaseModel):
    text: str
    src_lang: str | None = None
    tgt_lang: str
    model: str | None = None


class TranslationResponse(BaseModel):
    translated_text: str


# --- Visual query (POST v1/indic_visual_query) ---


class VisualQueryResponse(BaseModel):
    answer: str


# --- Extract text (POST v1/extract-text) ---


class ExtractTextResponse(BaseModel):
    text: str
    page_number: int | None = None


# --- PDF summary (POST v1/indic-summarize-pdf-all) ---


class PdfSummaryResponse(BaseModel):
    summary: str


# --- Transcription (POST v1/transcribe/) ---


class TranscriptionResponse(BaseModel):
    text: str
