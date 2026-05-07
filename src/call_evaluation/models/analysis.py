"""Shared enums and Pydantic result models used across detectors and services."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AnalysisApproach(str, Enum):
    """Which detection backend to use for a batch run."""

    REGEX = "regex"
    LLM = "llm"


class SeverityLabel(str, Enum):
    """Ordered profanity severity tiers, from clean to most severe."""

    NONE = "NONE"
    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


class SentimentLabel(str, Enum):
    """Overall emotional tone of a speaker in the call."""

    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class ContextLabel(str, Enum):
    """How profanity is used — direction and framing within the utterance."""

    DIRECTED_AT_AGENT = "DIRECTED_AT_AGENT"
    DIRECTED_AT_CUSTOMER = "DIRECTED_AT_CUSTOMER"
    SELF_EXPRESSION = "SELF_EXPRESSION"
    NARRATIVE_QUOTE = "NARRATIVE_QUOTE"
    AMBIENT = "AMBIENT"


class PrivacyVerificationStatus(str, Enum):
    """How many identity factors the agent confirmed before disclosing account data."""

    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    UNVERIFIED = "UNVERIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ComplianceViolation(str, Enum):
    """Whether the call contains a privacy or compliance violation."""

    YES = "YES"
    NO = "NO"


class SpecialCallType(str, Enum):
    """Structural category of a call that may affect how analysis is applied."""

    EMPTY = "EMPTY"
    VOICEMAIL = "VOICEMAIL"
    WRONG_PERSON = "WRONG_PERSON"
    SINGLE_SPEAKER = "SINGLE_SPEAKER"
    STANDARD = "STANDARD"


class EvidenceSpan(BaseModel):
    """A timestamped excerpt from a transcript that supports a detection finding."""

    speaker_role: str
    text: str
    stime: float
    etime: float
    reason: str


class ProfanityAnalysisResult(BaseModel):
    """Complete profanity analysis result for a single speaker in one call."""

    call_id: str = ""
    flag: bool
    speaker_role: str
    severity: SeverityLabel
    sentiment: SentimentLabel
    context: ContextLabel
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    notes: str = ""


class ComplianceAnalysisResult(BaseModel):
    """Complete compliance analysis result for a single call."""

    call_id: str = ""
    violation: ComplianceViolation
    verification_status: PrivacyVerificationStatus
    violation_type: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    notes: str = ""


class MetricResult(BaseModel):
    """Timing and talk-time metrics computed from a single call's timestamps."""

    call_id: str
    total_duration: float = 0.0
    agent_talk_time: float = 0.0
    customer_talk_time: float = 0.0
    agent_talk_pct: float = 0.0
    customer_talk_pct: float = 0.0
    silence_pct: float = 0.0
    overtalk_pct: float = 0.0
    special_case: str = ""


class BatchProcessingReport(BaseModel):
    """Summary counts and per-call errors from a batch analysis run."""

    processed_calls: int
    successful_calls: int
    failed_calls: int
    errors: dict[str, list[str]] = Field(default_factory=dict)
