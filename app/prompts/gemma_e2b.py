"""
ASR prompt strings from the Gemma 4 E2B instruction-tuned model card (Audio / ASR).

See: https://huggingface.co/google/gemma-4-E2B-it — Best Practices, section 6 (Audio),
and the cookbook audio snippet (original-language transcription).

For **vLLM** OpenAI-style calls, follow the Gemma 4 recipe: **text**, then **`input_audio`**
(https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html#audio-transcription-openai-sdk).
The Hugging Face model card sometimes recommends other modality orders for raw Transformers; this server matches vLLM.
"""

# Template from "Audio Speech Recognition (ASR)" in Best Practices (section 6).
GEMMA_ASR_PROMPT_TEMPLATE = """Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."""

# Cookbook-style prompt when the speech should stay in its original language (no fixed locale).
GEMMA_ASR_PROMPT_ORIGINAL_LANGUAGE = """Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."""


def asr_prompt(language: str) -> str:
    """Return the Gemma E2B ASR user text for the given `language` query param.

    Use the original-language variant when `language` is empty or clearly \"auto\".
    Otherwise substitute `{LANGUAGE}` with the client-provided code or name (e.g. `en`, `hi`).
    """
    lang = (language or "").strip()
    if not lang or lang.lower() in ("auto", "detect", "unknown"):
        return GEMMA_ASR_PROMPT_ORIGINAL_LANGUAGE
    return GEMMA_ASR_PROMPT_TEMPLATE.format(LANGUAGE=lang)
