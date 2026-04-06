from app.prompts.gemma_e2b import (
    GEMMA_ASR_PROMPT_ORIGINAL_LANGUAGE,
    GEMMA_ASR_PROMPT_TEMPLATE,
    asr_prompt,
)


def test_asr_prompt_uses_language_template():
    p = asr_prompt("en")
    assert "en" in p
    assert "Transcribe the following speech segment" in p
    assert GEMMA_ASR_PROMPT_TEMPLATE.format(LANGUAGE="en") == p


def test_asr_prompt_auto_uses_original_language():
    p = asr_prompt("auto")
    assert p == GEMMA_ASR_PROMPT_ORIGINAL_LANGUAGE
    assert "original language" in p.lower()
