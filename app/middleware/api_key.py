from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


def _extract_api_key(request: Request) -> str:
    """First non-empty credential from X-API-Key, Api-Key, or Authorization: Bearer."""
    h = request.headers
    for name in ("x-api-key", "api-key"):
        v = (h.get(name) or "").strip()
        if v:
            return v
    auth = (h.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if token:
            return token
    return ""


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in ("/healthz", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)
        if not _extract_api_key(request):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key", "code": "unauthorized"},
            )
        return await call_next(request)
