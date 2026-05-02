from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AnalysisApproach(str, Enum):
    REGEX = "regex"
    LLM = "llm"


class SeverityLabel(str, Enum):
    NONE = "NONE"
    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class ContextLabel(str, Enum):
    DIRECTED_AT_AGENT = "DIRECTED_AT_AGENT"
    DIRECTED_AT_CUSTOMER = "DIRECTED_AT_CUSTOMER"
    SELF_EXPRESSION = "SELF_EXPRESSION"
    NARRATIVE_QUOTE = "NARRATIVE_QUOTE"
    AMBIENT = "AMBIENT"


class PrivacyVerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    UNVERIFIED = "UNVERIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ComplianceViolation(str, Enum):
    YES = "YES"
    NO = "NO"


class SpecialCallType(str, Enum):
    EMPTY = "EMPTY"
    VOICEMAIL = "VOICEMAIL"
    WRONG_PERSON = "WRONG_PERSON"
    SINGLE_SPEAKER = "SINGLE_SPEAKER"
    STANDARD = "STANDARD"


class EvidenceSpan(BaseModel):
    speaker_role: str
    text: str
    stime: float
    etime: float
    reason: str


class ProfanityAnalysisResult(BaseModel):
    call_id: str = ""
    flag: bool
    speaker_role: str
    severity: SeverityLabel
    sentiment: SentimentLabel
    context: ContextLabel
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    notes: str = ""


class ComplianceAnalysisResult(BaseModel):
    call_id: str = ""
    violation: ComplianceViolation
    verification_status: PrivacyVerificationStatus
    violation_type: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    notes: str = ""


class MetricResult(BaseModel):
    call_id: str
    total_duration: float = 0.0
    agent_talk_pct: float = 0.0
    customer_talk_pct: float = 0.0
    silence_pct: float = 0.0
    overtalk_pct: float = 0.0
    special_case: str = ""


class BatchProcessingReport(BaseModel):
    processed_calls: int
    successful_calls: int
    failed_calls: int
    errors: dict[str, list[str]] = Field(default_factory=dict)
