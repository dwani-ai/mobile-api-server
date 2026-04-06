from fastapi import APIRouter

from . import (
    chat,
    extract,
    pdf_summary,
    speech,
    sts,
    transcribe,
    translate,
    visual,
)

router = APIRouter()
router.include_router(transcribe.router, tags=["Audio"])
router.include_router(chat.router, tags=["Chat"])
router.include_router(speech.router, tags=["Audio"])
router.include_router(translate.router, tags=["Translation"])
router.include_router(visual.router, tags=["Chat"])
router.include_router(sts.router, tags=["Audio"])
router.include_router(extract.router, tags=["PDF"])
router.include_router(pdf_summary.router, tags=["PDF"])
