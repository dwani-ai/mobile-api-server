# Mobile API Server

FastAPI backend for the mobile client: JSON and multipart endpoints for chat, translation, vision, speech (STT/TTS/STS), PDF summarization, and text extraction. LLM work is delegated to an **OpenAI-compatible** upstream (**vLLM** or **LiteLLM**) using the official Python `openai` client.

## Architecture (ASCII)

```
                          +---------------------------+
                          |        Clients          |
                          |  (Android / services)   |
                          +-------------+-----------+
                                        |
                          X-API-Key     |  HTTP(S)
                          (header)      |
                                        v
+--------------------------------------------------------------------------------+
|                                   nginx                                       |
|  - reverse proxy to the API container                                         |
|  - client_max_body_size (large uploads)                                       |
|  - elevated proxy read/send timeouts (LLM + uploads)                          |
|  - optional: TLS termination in front                                         |
+----------------------------------------+--------------------------------------+
                                         |
                                         v
+--------------------------------------------------------------------------------+
|                         FastAPI  (uvicorn :8000)                              |
|  +---------------------------------------------------------------------------+ |
|  | Global: /healthz (no API key)  |  /docs, /openapi.json (no API key)       | |
|  +---------------------------------------------------------------------------+ |
|  | Middleware: ApiKeyMiddleware  ->  X-API-Key must match API_KEYS          | |
|  +---------------------------------------------------------------------------+ |
|  | Routers: /v1/transcribe, /v1/indic_chat, /v1/audio/speech, ...            | |
|  +------------------+----------------+----------------+------------------------+ |
|                    |                |                |                        |
|                    v                v                v                        | |
|            +---------------+ +-------------+ +---------------+              | |
|            | STT / TTS     | | VllmClient  | | PDF + OCR     |              | |
|            | adapters      | | (AsyncOpenAI| | adapters      |              | |
|            | (stub or      | |  chat API)  | | (pypdf, stub, |              | |
|            | vllm_gemma)   | |             | | pdf_text)     |              | |
|            +-------+-------+ +------+------+ +--------+------+              | |
|                    |                |                |                        |
+--------------------------------------------------------------------------------+
                     |                |                |
                     |                |                |
                     |                v                |
                     |       OpenAI-compatible         |
                     |       HTTP (JSON multimodal)    |
                     |                |                |
                     |                v                |
                     |       +-------------------+       |
                     |       |  vLLM or LiteLLM |       |
                     |       |  (Gemma / other)  |       |
                     |       +-------------------+       |
                     |                                 |
                     +---------------------------------+
```

**Data flow summary**

- **Authenticated API traffic** enters nginx, is forwarded to FastAPI, and is checked against `API_KEYS` via `X-API-Key`.
- **Chat, translate, vision, PDF summarize (after text extraction)** call `VllmClient`, which uses `POST /chat/completions` on the configured base URL (`VLLM_BASE_URL` or `LITELLM_API_BASE`, must include `/v1`).
- **Transcribe / speech-to-speech (STT leg)** can use stubs or `STT_IMPLEMENTATION=vllm_gemma` (Gemma ASR prompts + `input_audio` per the [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html#audio-transcription-openai-sdk)).
- **TTS** returns binary audio from a stub or future engine adapter.
- **extract-text** uses OCR adapters (stub or PDF text extraction).

## Project layout

```
app/
  main.py              # App factory, middleware, /healthz
  config.py            # pydantic-settings
  clients/vllm.py      # OpenAI-compatible client + transcribe/vision/chat helpers
  adapters/            # stt, tts, ocr, pdf
  prompts/gemma_e2b.py # Gemma 4 E2B ASR prompt text
  routers/v1/          # One module per route group
  schemas/v1.py        # Request/response models
docker/
  Dockerfile
  nginx.conf
docker-compose.yml
tests/
```

## Quick start

**1. Configure environment**

```bash
cp .env.example .env
# Set API_KEYS, VLLM_BASE_URL (or LITELLM_API_BASE), and model defaults.
```

**2. Run locally**

```bash
uv sync --all-extras
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**3. Run with Docker Compose (API + nginx)**

```bash
docker compose --env-file .env up --build
```

Nginx listens on `HTTP_PORT` (default **8080**) and proxies to the API service.

## Environment

See [`.env.example`](.env.example) for all variables. Important ones:

| Variable | Role |
|----------|------|
| `API_KEYS` | Comma-separated keys; must match `X-API-Key` |
| `VLLM_BASE_URL` or `LITELLM_API_BASE` | OpenAI-compatible base URL ending in `/v1` |
| `DEFAULT_*_MODEL` | Fallback model IDs when the client omits them |
| `STT_IMPLEMENTATION` | `stub` or `vllm_gemma` |
| `MAX_UPLOAD_MB` | Upload size guard |

## API surface (v1)

| Method | Path | Notes |
|--------|------|--------|
| `POST` | `/v1/transcribe/` | multipart audio + `language` |
| `POST` | `/v1/indic_chat` | JSON chat |
| `POST` | `/v1/audio/speech` | TTS; query `input`, `language` |
| `POST` | `/v1/translate` | JSON |
| `POST` | `/v1/indic_visual_query` | multipart file + `query` + langs |
| `POST` | `/v1/speech_to_speech` | multipart audio |
| `POST` | `/v1/extract-text` | multipart + `page_number`, `language` |
| `POST` | `/v1/indic-summarize-pdf-all` | multipart PDF + `tgt_lang`, `model` parts |

OpenAPI: `/docs` or `/openapi.json` (unauthenticated in the default middleware).

## Tests

```bash
uv run pytest
```

## References

- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [Gemma 4 — audio transcription (OpenAI SDK)](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html#audio-transcription-openai-sdk)
- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) (prompt wording for ASR)
