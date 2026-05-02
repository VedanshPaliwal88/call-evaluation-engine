from __future__ import annotations

import re

from call_evaluation.models.analysis import (
    ComplianceAnalysisResult,
    ComplianceViolation,
    EvidenceSpan,
    PrivacyVerificationStatus,
    SpecialCallType,
)
from call_evaluation.models.transcript import SpeakerRole, TranscriptFilePayload
from call_evaluation.utils.text import normalize_for_matching, normalize_for_verification


DISCLOSURE_PATTERNS = [
    re.compile(r"\byour balance\b"),
    re.compile(r"\boutstanding balance\b"),
    re.compile(r"\boverdue balance\b"),
    re.compile(r"\bbalance of\b"),
    re.compile(r"\byour account details?\b"),
    re.compile(r"\baccount details?\b"),
    re.compile(r"\byou owe\b"),
    re.compile(r"\blast payment was\b"),
    re.compile(r"\bi see your last payment\b"),
]
VERIFICATION_PATTERNS = [
    re.compile(r"\bdate of birth\b"),
    re.compile(r"\bdob\b"),
    re.compile(r"\baddress\b"),
    re.compile(r"\bsocial security\b"),
    re.compile(r"\bssn\b"),
]

MONTH_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
)
ORDINAL_PATTERN = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty first|twenty-first|twenty second|twenty-second|twenty third|twenty-third|twenty fourth|twenty-fourth|twenty fifth|twenty-fifth|twenty sixth|twenty-sixth|twenty seventh|twenty-seventh|twenty eighth|twenty-eighth|twenty ninth|twenty-ninth|thirtieth|thirty first|thirty-first)"
YEAR_WORD_PATTERN = r"(?:nineteen|twenty)(?:\s+(?:oh|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)){1,2}"
ADDRESS_WORD_NUMBER_PATTERN = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"

SSN_DIGIT_PATTERN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")
SSN_SPELLED_PATTERN = re.compile(
    r"\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)(?:[\s-]+(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)){8}\b"
)
DOB_SLASH_PATTERN = re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b")
DOB_MONTH_TEXT_PATTERN = re.compile(rf"\b{MONTH_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+(?:19|20)\d{{2}}\b")
DOB_OF_MONTH_PATTERN = re.compile(rf"\b(?:the\s+)?{ORDINAL_PATTERN}\s+of\s+{MONTH_PATTERN}\b")
DOB_SPELLED_PATTERN = re.compile(rf"\b{MONTH_PATTERN}\s+{ORDINAL_PATTERN}\s+{YEAR_WORD_PATTERN}\b")
ADDRESS_DIGIT_PATTERN = re.compile(
    r"\b\d{1,6}\s+[a-z]+(?:\s+[a-z]+){0,4}\s+(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|court|ct|boulevard|blvd|way)\b"
)
ADDRESS_SPELLED_PATTERN = re.compile(
    rf"\b(?:{ADDRESS_WORD_NUMBER_PATTERN})(?:[\s-]+(?:{ADDRESS_WORD_NUMBER_PATTERN})){{0,3}}\s+[a-z]+(?:\s+[a-z]+){{0,4}}\s+(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|court|ct|boulevard|blvd|way)\b"
)


class RegexComplianceDetector:
    def analyze(self, transcript: TranscriptFilePayload) -> ComplianceAnalysisResult:
        tags = set(transcript.special_tags)

        if SpecialCallType.VOICEMAIL.value in tags:
            voicemail_evidence = transcript.turns[:1]
            return ComplianceAnalysisResult(
                call_id=transcript.call_id,
                violation=ComplianceViolation.YES if self._contains_disclosure(transcript) else ComplianceViolation.NO,
                verification_status=PrivacyVerificationStatus.NOT_APPLICABLE,
                violation_type="VOICEMAIL_DISCLOSURE" if self._contains_disclosure(transcript) else "VOICEMAIL_NO_DISCLOSURE",
                evidence=[
                    EvidenceSpan(
                        speaker_role=turn.speaker.value,
                        text=turn.text,
                        stime=turn.stime,
                        etime=turn.etime,
                        reason="Voicemail-style interaction prevents live identity verification.",
                    )
                    for turn in voicemail_evidence
                ],
                notes="Voicemail detected; verification cannot occur in a live two-way flow.",
            )

        if SpecialCallType.WRONG_PERSON.value in tags:
            return ComplianceAnalysisResult(
                call_id=transcript.call_id,
                violation=ComplianceViolation.NO,
                verification_status=PrivacyVerificationStatus.NOT_APPLICABLE,
                violation_type="WRONG_PERSON",
                evidence=[],
                notes="Wrong-person call detected; no valid borrower verification flow applies.",
            )

        verified = False
        pending_verification = False
        evidence: list[EvidenceSpan] = []

        for turn in transcript.turns:
            normalized = normalize_for_matching(turn.text)
            verification_text = normalize_for_verification(turn.text)
            if turn.speaker == SpeakerRole.AGENT and any(pattern.search(normalized) for pattern in VERIFICATION_PATTERNS):
                pending_verification = True
            elif turn.speaker == SpeakerRole.CUSTOMER and pending_verification and self._looks_like_customer_confirmation(verification_text):
                verified = True
                pending_verification = False

            if turn.speaker == SpeakerRole.AGENT and self._contains_disclosure_text(normalized):
                evidence.append(
                    EvidenceSpan(
                        speaker_role=turn.speaker.value,
                        text=turn.text,
                        stime=turn.stime,
                        etime=turn.etime,
                        reason="Agent disclosed potentially sensitive account information.",
                    )
                )
                if not verified:
                    return ComplianceAnalysisResult(
                        call_id=transcript.call_id,
                        violation=ComplianceViolation.YES,
                        verification_status=PrivacyVerificationStatus.UNVERIFIED,
                        violation_type="DISCLOSURE_BEFORE_VERIFICATION",
                        evidence=evidence,
                        notes="Regex path found sensitive disclosure before full verification.",
                    )

        return ComplianceAnalysisResult(
            call_id=transcript.call_id,
            violation=ComplianceViolation.NO,
            verification_status=PrivacyVerificationStatus.VERIFIED if verified else PrivacyVerificationStatus.UNVERIFIED,
            violation_type="NO_VIOLATION",
            evidence=evidence,
            notes="No pre-verification disclosure was detected by regex rules.",
        )

    def _contains_disclosure(self, transcript: TranscriptFilePayload) -> bool:
        return any(self._contains_disclosure_text(normalize_for_matching(turn.text)) for turn in transcript.turns if turn.speaker == SpeakerRole.AGENT)

    @staticmethod
    def _contains_disclosure_text(normalized_text: str) -> bool:
        if re.search(r"\b(a|the)\s+balance\b", normalized_text) and not re.search(r"\b(your|outstanding|overdue|owe|account|payment)\b", normalized_text):
            return False
        return any(pattern.search(normalized_text) for pattern in DISCLOSURE_PATTERNS)

    @staticmethod
    def _looks_like_customer_confirmation(normalized_text: str) -> bool:
        normalized_text = normalized_text.strip()
        verification_patterns = [
            SSN_DIGIT_PATTERN,
            SSN_SPELLED_PATTERN,
            DOB_SLASH_PATTERN,
            DOB_MONTH_TEXT_PATTERN,
            DOB_OF_MONTH_PATTERN,
            DOB_SPELLED_PATTERN,
            ADDRESS_DIGIT_PATTERN,
            ADDRESS_SPELLED_PATTERN,
        ]
        return any(pattern.search(normalized_text) for pattern in verification_patterns)
