import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from openai import APIError, APITimeoutError

from app.clients.vllm import VllmClient
from app.config import get_settings
from app.dependencies import deps_vllm
from app.lang_names import get_language_name
from app.schemas.v1 import ChatMessage, ChatRequest, ChatResponse, IndicChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()

VALID_INDIC_CHAT_MODELS = ("gemma4",)


@router.post(
    "/indic_chat",
    response_model=ChatResponse,
    summary="Chat with AI",
    description=(
        "Generate a chat response from a prompt, language code, and model, "
        "with translation support and time-to-words conversion."
    ),
    tags=["Chat"],
    responses={
        200: {"description": "Chat response", "model": ChatResponse},
        400: {"description": "Invalid prompt, language code, or model"},
        504: {"description": "Chat service timeout"},
    },
)
async def indic_chat(
    chat_request: IndicChatRequest,
    vllm: VllmClient = Depends(deps_vllm),
    _x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> ChatResponse:
    _ = _x_api_key
    if not chat_request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(chat_request.prompt) > 10000:
        raise HTTPException(status_code=400, detail="Prompt cannot exceed 10000 characters")

    if chat_request.model not in VALID_INDIC_CHAT_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from {list(VALID_INDIC_CHAT_MODELS)}",
        )

    settings = get_settings()
    logger.debug(
        "indic_chat prompt=%r src_lang=%s tgt_lang=%s model=%s",
        chat_request.prompt,
        chat_request.src_lang,
        chat_request.tgt_lang,
        chat_request.model,
    )

    language_name = get_language_name(chat_request.tgt_lang)
    system_prompt = (
        "You are dwani, a helpful assistant. Answer questions considering India as base "
        "country and Karnataka as base state. Provide a concise response in one sentence "
        f"maximum. Do not explain. Return answer only in {language_name}"
    )

    body = ChatRequest(
        model=chat_request.model,
        messages=[
            ChatMessage(
                role="system",
                content=[{"type": "text", "text": system_prompt}],
            ),
            ChatMessage(
                role="user",
                content=[{"type": "text", "text": chat_request.prompt}],
            ),
        ],
        temperature=0.3,
        max_tokens=settings.max_tokens,
    )

    try:
        return await vllm.chat(body)
    except APITimeoutError:
        logger.error("Chat API request timed out")
        raise HTTPException(status_code=504, detail="Chat service timeout") from None
    except APIError as e:
        logger.error("Chat API error: %s", e)
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}") from e
    except Exception as e:
        logger.exception("Error processing indic_chat")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}") from e
