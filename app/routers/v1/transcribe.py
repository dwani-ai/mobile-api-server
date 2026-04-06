from fastapi import APIRouter, Depends, File, Header, Query, UploadFile

from app.adapters.stt import SttAdapter
from app.dependencies import deps_stt
from app.schemas.v1 import TranscriptionResponse
from app.routers.v1._upload import read_upload_limited

router = APIRouter()


@router.post("/transcribe/")
async def transcribe_audio(
    language: str = Query(..., description="Language code"),
    audio: UploadFile = File(..., description="Audio file"),
    stt: SttAdapter = Depends(deps_stt),
    _x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> TranscriptionResponse:
    _ = _x_api_key
    raw = await read_upload_limited(audio)
    text = await stt.transcribe(raw, language, mime_type=audio.content_type)
    return TranscriptionResponse(text=text)
