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
        violation_type="DISCLOSURE_BEFORE_VERIFICATION",
        evidence=[],
        notes="ok",
    )
    assert result.verification_status.value == "UNVERIFIED"


def test_llm_compliance_detector_routes_through_prompt_file() -> None:
    transcript = IngestionService().load_named_bytes(
        "case.json",
        b'[{"speaker":"Agent","text":"You owe $450 on this account.","stime":0,"etime":3}]',
    )[0]
    response = ComplianceAnalysisResult(
        call_id="placeholder",
        violation=ComplianceViolation.YES,
        verification_status=PrivacyVerificationStatus.UNVERIFIED,
        violation_type="DISCLOSURE_BEFORE_VERIFICATION",
        evidence=[],
        notes="ok",
    )
    detector = LLMComplianceDetector(llm_client=DummyClient(response))
    result = detector.analyze(transcript)
    assert result.call_id == "case"
    assert result.violation.value == "YES"
