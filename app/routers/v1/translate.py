from fastapi import APIRouter, Depends, Header

from app.clients.vllm import VllmClient
from app.dependencies import deps_vllm
from app.schemas.v1 import TranslationRequest, TranslationResponse

router = APIRouter()


@router.post("/translate", response_model=TranslationResponse)
async def translate(
    translation_request: TranslationRequest,
    vllm: VllmClient = Depends(deps_vllm),
    _x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> TranslationResponse:
    _ = _x_api_key
    return await vllm.translate(translation_request)
