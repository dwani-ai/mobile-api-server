from fastapi import APIRouter, Depends, File, Header, Query, UploadFile
from fastapi.responses import Response

from app.adapters.stt import SttAdapter
from app.adapters.tts import TtsAdapter
from app.dependencies import deps_stt, deps_tts
from app.routers.v1._upload import read_upload_limited

router = APIRouter()


@router.post("/speech_to_speech")
async def speech_to_speech(
    language: str = Query(...),
    file: UploadFile = File(...),
    stt: SttAdapter = Depends(deps_stt),
    tts: TtsAdapter = Depends(deps_tts),
    _x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> Response:
    _ = _x_api_key
    raw = await read_upload_limited(file)
    text = await stt.transcribe(raw, language, mime_type=file.content_type)
    audio_bytes, media_type = await tts.synthesize(text, language)
    return Response(content=audio_bytes, media_type=media_type)
