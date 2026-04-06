from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings


class ApiKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in ("/healthz", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)
        key = request.headers.get("X-API-Key")
        if not key or key not in self._settings.api_key_set:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key", "code": "unauthorized"},
            )
        return await call_next(request)
