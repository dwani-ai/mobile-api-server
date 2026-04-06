import os

# Must be set before importing app modules that cache settings.
os.environ.setdefault("API_KEYS", "test-key")
os.environ.setdefault("VLLM_BASE_URL", "http://vllm.test/v1")

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def api_headers():
    return {"X-API-Key": "test-key"}
