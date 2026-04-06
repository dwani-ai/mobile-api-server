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
router.include_router(transcribe.router, tags=["transcribe"])
router.include_router(chat.router, tags=["chat"])
router.include_router(speech.router, tags=["speech"])
router.include_router(translate.router, tags=["translate"])
router.include_router(visual.router, tags=["visual"])
router.include_router(sts.router, tags=["speech_to_speech"])
router.include_router(extract.router, tags=["extract"])
router.include_router(pdf_summary.router, tags=["pdf"])
