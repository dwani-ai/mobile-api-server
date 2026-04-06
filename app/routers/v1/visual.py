from fastapi import APIRouter, Depends, File, Header, Query, UploadFile

from app.clients.vllm import VllmClient
from app.dependencies import deps_vllm
from app.schemas.v1 import VisualQueryResponse
from app.routers.v1._upload import read_upload_limited

router = APIRouter()


@router.post("/indic_visual_query", response_model=VisualQueryResponse)
async def indic_visual_query(
    *,
    file: UploadFile = File(...),
    query: UploadFile = File(..., description="Text query part"),
    src_lang: str = Query(...),
    tgt_lang: str = Query(...),
    vllm: VllmClient = Depends(deps_vllm),
    _x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> VisualQueryResponse:
    _ = _x_api_key
    raw = await read_upload_limited(file)
    mime = file.content_type or "application/octet-stream"
    qbytes = await query.read()
    qtext = qbytes.decode("utf-8", errors="replace")
    return await vllm.visual_query(
        image_bytes=raw,
        mime=mime,
        query=qtext,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        model=None,
    )
