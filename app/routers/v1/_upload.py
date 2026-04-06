from fastapi import HTTPException, UploadFile

from app.config import get_settings


async def read_upload_limited(upload: UploadFile, *, max_mb: int | None = None) -> bytes:
    limit = max_mb if max_mb is not None else get_settings().max_upload_mb
    data = await upload.read()
    if len(data) > limit * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {limit} MB",
        )
    return data
