from fastapi import APIRouter, Depends, Header

from app.clients.vllm import VllmClient
from app.dependencies import deps_vllm
from app.schemas.v1 import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/indic_chat", response_model=ChatResponse)
async def indic_chat(
    chat_request: ChatRequest,
    vllm: VllmClient = Depends(deps_vllm),
    _x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> ChatResponse:
    _ = _x_api_key
    return await vllm.chat(chat_request)
