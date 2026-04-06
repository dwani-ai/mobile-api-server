from fastapi import APIRouter, Depends, File, Header, UploadFile

from app.adapters.pdf import PdfAdapter
from app.dependencies import get_pdf_adapter
from app.schemas.v1 import PdfSummaryResponse
from app.routers.v1._upload import read_upload_limited

router = APIRouter()


@router.post("/indic-summarize-pdf-all", response_model=PdfSummaryResponse)
async def summarize_pdf(
    *,
    file: UploadFile = File(...),
    tgt_lang: UploadFile = File(..., alias="tgt_lang"),
    model_part: UploadFile = File(..., alias="model"),
    pdf: PdfAdapter = Depends(get_pdf_adapter),
    _x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> PdfSummaryResponse:
    _ = _x_api_key
    raw = await read_upload_limited(file)
    tgt = (await tgt_lang.read()).decode("utf-8", errors="replace").strip()
    model_name = (await model_part.read()).decode("utf-8", errors="replace").strip()
    summary = await pdf.summarize_pdf(raw, tgt_lang=tgt, model=model_name)
    return PdfSummaryResponse(summary=summary)
