from __future__ import annotations

from pydantic import BaseModel

from call_evaluation.models.analysis import (
    ComplianceAnalysisResult,
    ComplianceViolation,
    ContextLabel,
    EvidenceSpan,
    PrivacyVerificationStatus,
    ProfanityAnalysisResult,
    SentimentLabel,
    SeverityLabel,
)
from call_evaluation.detectors.llm.compliance import LLMComplianceDetector
from call_evaluation.ingestion import IngestionService
from call_evaluation.services.llm_client import LLMClient


class DummyClient:
    def __init__(self, response: BaseModel) -> None:
        self.response = response

    def render_prompt(self, prompt_name: str, replacements: dict[str, str]) -> str:
        return f"{prompt_name}:{sorted(replacements)}"

    def classify_json(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        assert prompt
        assert issubclass(response_model, BaseModel)
        return self.response


def test_profanity_result_schema_is_valid() -> None:
    result = ProfanityAnalysisResult(
        call_id="x",
        flag=True,
        speaker_role="CUSTOMER",
        severity=SeverityLabel.SEVERE,
        sentiment=SentimentLabel.NEGATIVE,
        context=ContextLabel.DIRECTED_AT_AGENT,
        evidence=[EvidenceSpan(speaker_role="CUSTOMER", text="fckn useless", stime=0, etime=2, reason="matched")],
        notes="ok",
    )
    assert result.flag is True


def test_compliance_result_schema_is_valid() -> None:
    result = ComplianceAnalysisResult(
        call_id="x",
        violation=ComplianceViolation.YES,
        verification_status=PrivacyVerificationStatus.UNVERIFIED,
        violation_type="ACCOUNT_DETAILS_BEFORE_VERIFICATION",
        evidence=[],
        notes="ok",
    )
    assert result.verification_status.value == "UNVERIFIED"


def test_llm_client_sanitizes_bad_enum_values_to_safe_defaults() -> None:
    client = LLMClient()
    client._import_error = None

    class FakeChoice:
        message = type("M", (), {"content": '{"flag":true,"speaker_role":"CUSTOMER","severity":"VERY_SEVERE","sentiment":"HAPPY","context":"SHOUTING","evidence":[],"notes":"test"}'})()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        chat = FakeChat()

    client._client = FakeOpenAI()
    result = client.classify_json("prompt", ProfanityAnalysisResult)
    assert result.severity.value == "NONE"
    assert result.sentiment.value == "NEUTRAL"
    assert result.context.value == "AMBIENT"


def test_llm_client_timeout_returns_safe_fallback() -> None:
    client = LLMClient()
    client._import_error = None

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("Connection timed out after 30 seconds")

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        chat = FakeChat()

    client._client = FakeOpenAI()
    result = client.classify_json("prompt", ProfanityAnalysisResult)
    assert result.flag is False
    assert "timed out" in result.notes.lower()


def test_llm_client_is_timeout_detects_string_variants() -> None:
    assert LLMClient._is_timeout(RuntimeError("Connection timed out")) is True
    assert LLMClient._is_timeout(RuntimeError("request timeout exceeded")) is True
    assert LLMClient._is_timeout(RuntimeError("some other error")) is False


def test_llm_compliance_detector_routes_through_prompt_file() -> None:
    transcript = IngestionService().load_named_bytes(
        "case.json",
        b'[{"speaker":"Agent","text":"You owe $450 on this account.","stime":0,"etime":3}]',
    )[0]
    response = ComplianceAnalysisResult(
        call_id="placeholder",
        violation=ComplianceViolation.YES,
        verification_status=PrivacyVerificationStatus.UNVERIFIED,
        violation_type="ACCOUNT_DETAILS_BEFORE_VERIFICATION",
        evidence=[],
        notes="ok",
    )
    detector = LLMComplianceDetector(llm_client=DummyClient(response))
    result = detector.analyze(transcript)
    assert result.call_id == "case"
    assert result.violation.value == "YES"
