import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import get_settings
from app.middleware.api_key import ApiKeyMiddleware
from app.routers.v1 import router as v1_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    yield


_settings = get_settings()
_cors = [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
_cors_allow = _cors if _cors else ["*"]

app = FastAPI(
    title="dwani.ai API",
    description="A multimodal inference API designed for privacy",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Chat", "description": "Chat-related endpoints"},
        {"name": "Audio", "description": "Audio processing and TTS endpoints"},
        {"name": "Translation", "description": "Text translation endpoints"},
        {"name": "Utility", "description": "General utility endpoints"},
        {"name": "PDF", "description": "PDF extraction and summarization"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ApiKeyMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "code": "validation_error"},
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s in %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/", summary="Redirect to Docs", tags=["Utility"])
async def root_redirect():
    return RedirectResponse(url="/docs")


@app.get("/healthz", tags=["Utility"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/v1/health",
    summary="Check API Health",
    description="Returns the health status of the API and the current default model name.",
    tags=["Utility"],
)
async def v1_health() -> dict[str, str]:
    s = get_settings()
    return {"status": "healthy", "model": s.default_chat_model}


app.include_router(v1_router, prefix="/v1")
