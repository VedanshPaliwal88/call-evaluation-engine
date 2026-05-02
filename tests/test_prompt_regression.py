from __future__ import annotations

from pathlib import Path

from call_evaluation.models.analysis import (
    ComplianceViolation,
    ContextLabel,
    PrivacyVerificationStatus,
    SentimentLabel,
    SeverityLabel,
)
from call_evaluation.services.llm_client import LLMClient


def test_profanity_prompt_file_exists_and_is_not_hardcoded() -> None:
    prompt_path = Path("src/call_evaluation/detectors/llm/prompts/profanity_v1.txt")
    assert prompt_path.exists()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "{{TRANSCRIPT}}" in prompt_text
    assert "{{TARGET_SPEAKER_ROLE}}" in prompt_text
    assert "noisy transcribed speech" in prompt_text.lower()


def test_profanity_prompt_contains_expected_enums() -> None:
    prompt_text = Path("src/call_evaluation/detectors/llm/prompts/profanity_v1.txt").read_text(encoding="utf-8")
    for value in SeverityLabel:
        assert value.value in prompt_text
    for value in ContextLabel:
        assert value.value in prompt_text
    for value in SentimentLabel:
        assert value.value in prompt_text


def test_llm_client_uses_zero_temperature() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            raise RuntimeError("stop after capture")

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    llm_client = LLMClient()
    llm_client._client = FakeClient()
    llm_client._import_error = None
    try:
        llm_client.classify_json("{}", dict)  # type: ignore[arg-type]
    except RuntimeError:
        pass
    assert llm_client._client.chat.completions.kwargs["temperature"] == 0.0


def test_compliance_prompt_contains_expected_enums_and_edge_cases() -> None:
    prompt_text = Path("src/call_evaluation/detectors/llm/prompts/compliance_v1.txt").read_text(encoding="utf-8")
    for value in PrivacyVerificationStatus:
        assert value.value in prompt_text
    for value in ComplianceViolation:
        assert value.value in prompt_text
    for phrase in [
        "voicemail",
        "wrong-person",
        "implied verification",
        "partial verification",
        "disclosure before any verification attempt begins",
    ]:
        assert phrase in prompt_text.lower()
