from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from pypdf import PdfWriter

from app.adapters.pdf import PdfAdapter
from app.clients.vllm import VllmClient
from app.config import get_settings
from app.dependencies import deps_vllm, get_pdf_adapter
from app.main import app
from app.schemas.v1 import (
    ChatChoice,
    ChatChoiceMessage,
    ChatResponse,
    TranslationResponse,
    VisualQueryResponse,
)


@pytest.fixture
def mock_vllm() -> MagicMock:
    m = MagicMock(spec=VllmClient)
    m.chat = AsyncMock(
        return_value=ChatResponse(
            id="1",
            created=1,
            model="m",
            choices=[
                ChatChoice(
                    message=ChatChoiceMessage(role="assistant", content="hello"),
                )
            ],
        )
    )
    m.translate = AsyncMock(return_value=TranslationResponse(translated_text="hola"))
    m.visual_query = AsyncMock(return_value=VisualQueryResponse(answer="cat"))
    m.summarize_text = AsyncMock(return_value="short summary")
    return m


@pytest.fixture
def client_with_vllm(mock_vllm):
    app.dependency_overrides[deps_vllm] = lambda: mock_vllm

    def pdf_dep():
        return PdfAdapter(mock_vllm, get_settings())

    app.dependency_overrides[get_pdf_adapter] = pdf_dep
    transport = ASGITransport(app=app)
    yield AsyncClient(transport=transport, base_url="http://test")
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_healthz_no_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_v1_requires_key():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/v1/translate", json={"text": "a", "tgt_lang": "es"})
    assert r.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {"X-API-Key": "X-API-Key"},
        {"Api-Key": "opaque-secret"},
        {"Authorization": "Bearer some-token"},
    ],
)
async def test_v1_accepts_key_from_headers(headers, client_with_vllm, mock_vllm):
    ac = client_with_vllm
    r = await ac.post(
        "/v1/translate",
        headers=headers,
        json={"text": "hello", "tgt_lang": "es"},
    )
    assert r.status_code == 200
    mock_vllm.translate.assert_awaited_once()


@pytest.mark.asyncio
async def test_indic_chat(client_with_vllm, mock_vllm, api_headers):
    ac = client_with_vllm
    r = await ac.post(
        "/v1/indic_chat",
        headers=api_headers,
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hello"
    mock_vllm.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_translate(client_with_vllm, mock_vllm, api_headers):
    ac = client_with_vllm
    r = await ac.post(
        "/v1/translate",
        headers=api_headers,
        json={"text": "hello", "tgt_lang": "es"},
    )
    assert r.status_code == 200
    assert r.json()["translated_text"] == "hola"
    mock_vllm.translate.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcribe(async_client, api_headers):
    files = {"audio": ("a.wav", b"fake", "audio/wav")}
    r = await async_client.post(
        "/v1/transcribe/",
        headers=api_headers,
        params={"language": "en"},
        files=files,
    )
    assert r.status_code == 200
    assert "text" in r.json()
    assert "stub" in r.json()["text"].lower()


@pytest.mark.asyncio
async def test_text_to_speech(async_client, api_headers):
    r = await async_client.post(
        "/v1/audio/speech",
        headers=api_headers,
        params={"input": "hi", "language": "en"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")
    assert len(r.content) > 0


@pytest.mark.asyncio
async def test_visual_query(client_with_vllm, mock_vllm, api_headers):
    ac = client_with_vllm
    files = {
        "file": ("x.png", b"\x89PNG\r\n\x1a\n", "image/png"),
        "query": ("q", b"what?", "text/plain"),
    }
    r = await ac.post(
        "/v1/indic_visual_query",
        headers=api_headers,
        params={"src_lang": "en", "tgt_lang": "en"},
        files=files,
    )
    assert r.status_code == 200
    assert r.json()["answer"] == "cat"
    mock_vllm.visual_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_speech_to_speech(async_client, api_headers):
    files = {"file": ("a.wav", b"x", "audio/wav")}
    r = await async_client.post(
        "/v1/speech_to_speech",
        headers=api_headers,
        params={"language": "en"},
        files=files,
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")


@pytest.mark.asyncio
async def test_extract_text(async_client, api_headers):
    files = {"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}
    r = await async_client.post(
        "/v1/extract-text",
        headers=api_headers,
        params={"page_number": 1, "language": "en"},
        files=files,
    )
    assert r.status_code == 200
    assert "text" in r.json()


@pytest.mark.asyncio
async def test_summarize_pdf(client_with_vllm, mock_vllm, api_headers):
    ac = client_with_vllm
    buf = BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=72, height=72)
    w.write(buf)
    pdf_bytes = buf.getvalue()
    files = {
        "file": ("t.pdf", pdf_bytes, "application/pdf"),
        "tgt_lang": ("tgt_lang.txt", b"en", "text/plain"),
        "model": ("model.txt", b"mymodel", "text/plain"),
    }
    r = await ac.post("/v1/indic-summarize-pdf-all", headers=api_headers, files=files)
    assert r.status_code == 200
    assert r.json()["summary"] == "short summary"
    mock_vllm.summarize_text.assert_awaited()
