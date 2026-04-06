"""Language codes aligned with dwani-api-server (FLORES-style *_Script)."""

from __future__ import annotations

# Same set as dwani-api-server main.py SUPPORTED_LANGUAGES
SUPPORTED_LANGUAGE_CODES: frozenset[str] = frozenset(
    {
        "eng_Latn",
        "hin_Deva",
        "kan_Knda",
        "tam_Taml",
        "mal_Mlym",
        "tel_Telu",
        "asm_Beng",
        "kas_Arab",
        "pan_Guru",
        "ben_Beng",
        "kas_Deva",
        "san_Deva",
        "brx_Deva",
        "mai_Deva",
        "sat_Olck",
        "doi_Deva",
        "snd_Arab",
        "mar_Deva",
        "snd_Deva",
        "gom_Deva",
        "mni_Beng",
        "guj_Gujr",
        "mni_Mtei",
        "npi_Deva",
        "urd_Arab",
        "ory_Orya",
        "deu_Latn",
        "fra_Latn",
        "nld_Latn",
        "spa_Latn",
        "ita_Latn",
        "por_Latn",
        "rus_Cyrl",
        "pol_Latn",
    }
)

# Display names for prompts (subset + fallback to code)
LANGUAGE_CODE_TO_NAME: dict[str, str] = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "tam_Taml": "Tamil",
    "mal_Mlym": "Malayalam",
    "tel_Telu": "Telugu",
    "asm_Beng": "Assamese",
    "ben_Beng": "Bengali",
    "guj_Gujr": "Gujarati",
    "mar_Deva": "Marathi",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "urd_Arab": "Urdu",
    "deu_Latn": "German",
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "nld_Latn": "Dutch",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "pol_Latn": "Polish",
}


def language_display_name(code: str) -> str:
    c = (code or "").strip()
    if not c:
        return "English"
    return LANGUAGE_CODE_TO_NAME.get(c, c)
