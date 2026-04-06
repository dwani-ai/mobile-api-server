from fastapi import APIRouter, Depends, File, Header, Query, UploadFile

from app.adapters.ocr import OcrAdapter
from app.dependencies import deps_ocr
from app.schemas.v1 import ExtractTextResponse
from app.routers.v1._upload import read_upload_limited

router = APIRouter()


@router.post("/extract-text", response_model=ExtractTextResponse)
async def extract_text(
    *,
    file: UploadFile = File(...),
    page_number: int = Query(..., ge=1),
    language: str = Query(...),
    ocr: OcrAdapter = Depends(deps_ocr),
    _x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> ExtractTextResponse:
    _ = _x_api_key
    raw = await read_upload_limited(file)
    text = await ocr.extract_text(
        raw, filename=file.filename, page_number=page_number, language=language
    )
    return ExtractTextResponse(text=text, page_number=page_number)
