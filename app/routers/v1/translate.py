from fastapi import APIRouter, Depends, Header, HTTPException

from app.clients.vllm import VllmClient
from app.dependencies import deps_vllm
from app.dwani_languages import SUPPORTED_LANGUAGE_CODES
from app.schemas.v1 import TranslationRequest, TranslationResponse

router = APIRouter()


@router.post(
    "/translate",
    response_model=TranslationResponse,
    summary="Translate Text",
    description="Translate a list of sentences from a source to a target language.",
    responses={
        200: {"description": "Translation result", "model": TranslationResponse},
        400: {"description": "Invalid sentences or languages"},
        500: {"description": "Translation service error"},
        504: {"description": "Translation service timeout"},
    },
)
async def translate(
    translation_request: TranslationRequest,
    vllm: VllmClient = Depends(deps_vllm),
    _x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> TranslationResponse:
    _ = _x_api_key
    if not translation_request.sentences:
        raise HTTPException(status_code=400, detail="Sentences cannot be empty")
    if (
        translation_request.src_lang not in SUPPORTED_LANGUAGE_CODES
        or translation_request.tgt_lang not in SUPPORTED_LANGUAGE_CODES
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported language codes: src={translation_request.src_lang}, "
                f"tgt={translation_request.tgt_lang}"
            ),
        )
    try:
        return await vllm.translate(translation_request)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
