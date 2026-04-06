from urllib.parse import unquote

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import Response

from app.adapters.tts import TtsAdapter
from app.dependencies import deps_tts

router = APIRouter()


@router.post("/audio/speech")
async def text_to_speech(
    input: str = Query(..., description="Text to speak"),
    language: str = Query(...),
    tts: TtsAdapter = Depends(deps_tts),
    _x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> Response:
    _ = _x_api_key
    text = unquote(input)
    audio_bytes, media_type = await tts.synthesize(text, language)
    return Response(content=audio_bytes, media_type=media_type)
