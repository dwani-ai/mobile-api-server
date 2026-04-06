"""Map language codes to English names for prompts (fallback: the code itself)."""

# Common codes for India-focused use; extend as needed.
_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "bn": "Bengali",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "kok": "Konkani",
    "sat": "Santali",
    "mni": "Manipuri",
    "brx": "Bodo",
    "doi": "Dogri",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ne": "Nepali",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ja": "Japanese",
    "zh": "Chinese",
}


def get_language_name(code: str) -> str:
    c = (code or "").strip().lower()
    if not c:
        return "English"
    return _LANG_NAMES.get(c, c)
