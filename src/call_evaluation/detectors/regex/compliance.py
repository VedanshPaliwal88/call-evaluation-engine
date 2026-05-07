"""Regex-based compliance detector for two-factor identity verification and disclosure timing."""
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
from call_evaluation.utils.text import normalize_for_verification


DISCLOSURE_PATTERNS = [
    re.compile(r"\bbalance of\b"),
    re.compile(r"\bbalance (?:is|was)\b"),
    re.compile(r"\bcurrent balance\b"),
    re.compile(r"\byour overdue balance is\b"),
    re.compile(r"\boverdue balance is\b"),
    re.compile(r"\byou owe\b"),
    re.compile(r"\blast payment was\b"),
    re.compile(r"\bi see your last payment\b"),
]
# Agent-side patterns: which factor is being requested
DOB_AGENT_PATTERNS = [
    re.compile(r"\bdate of birth\b"),
    re.compile(r"\bdob\b"),
]
ADDRESS_AGENT_PATTERNS = [
    re.compile(r"\baddress\b"),
]
SSN_AGENT_PATTERNS = [
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
    """Detect pre-verification account disclosures using a two-factor identity check."""

    def analyze(self, transcript: TranscriptFilePayload) -> ComplianceAnalysisResult:
        """Evaluate a transcript for compliance with identity verification requirements.

        Implements a per-turn state machine tracking three independent identity
        factors (date of birth, address, SSN). A disclosure before two factors are
        confirmed constitutes a violation. Special call types (voicemail, wrong-person)
        are handled before the main loop and return early.

        Args:
            transcript: Validated payload including special_tags and normalized turns.

        Returns:
            ComplianceAnalysisResult with violation, verification_status, and evidence.
        """
        tags = set(transcript.special_tags)

        if SpecialCallType.VOICEMAIL.value in tags:
            voicemail_evidence = transcript.turns[:1]
            voicemail_has_specific_disclosure = self._contains_specific_disclosure(transcript)
            return ComplianceAnalysisResult(
                call_id=transcript.call_id,
                violation=ComplianceViolation.YES if voicemail_has_specific_disclosure else ComplianceViolation.NO,
                verification_status=PrivacyVerificationStatus.NOT_APPLICABLE,
                violation_type="NO_VERIFICATION" if voicemail_has_specific_disclosure else "NOT_APPLICABLE",
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
                violation_type="NOT_APPLICABLE",
                evidence=[],
                notes="Wrong-person call detected; no valid borrower verification flow applies.",
            )

        # Two-factor state machine: each factor has a "pending" flag (agent asked)
        # and a "confirmed" flag (customer supplied a matching response).
        # Verification requires at least 2 of 3 factors confirmed before disclosure.
        dob_pending     = False
        address_pending = False
        ssn_pending     = False

        dob_confirmed     = False
        address_confirmed = False
        ssn_confirmed     = False

        pre_verification_disclosure: list[EvidenceSpan] = []
        evidence: list[EvidenceSpan] = []

        for turn in transcript.turns:
            normalized = normalize_for_verification(turn.text)

            if turn.speaker == SpeakerRole.AGENT:
                # Agent requests arm the pending flags; customer responses will
                # confirm them in the next branch of this loop.
                if any(p.search(normalized) for p in DOB_AGENT_PATTERNS):
                    dob_pending = True
                if any(p.search(normalized) for p in ADDRESS_AGENT_PATTERNS):
                    address_pending = True
                if any(p.search(normalized) for p in SSN_AGENT_PATTERNS):
                    ssn_pending = True

            elif turn.speaker == SpeakerRole.CUSTOMER:
                # A factor is confirmed only when the agent already requested it
                # (pending=True) and the customer's reply matches the pattern.
                if dob_pending and self._looks_like_dob_confirmation(normalized):
                    dob_confirmed = True
                    dob_pending = False
                if address_pending and self._looks_like_address_confirmation(normalized):
                    address_confirmed = True
                    address_pending = False
                if ssn_pending and self._looks_like_ssn_confirmation(normalized):
                    ssn_confirmed = True
                    ssn_pending = False

            if turn.speaker == SpeakerRole.AGENT and self._contains_disclosure_text(normalized):
                span = EvidenceSpan(
                    speaker_role=turn.speaker.value,
                    text=turn.text,
                    stime=turn.stime,
                    etime=turn.etime,
                    reason="Agent disclosed potentially sensitive account information.",
                )
                evidence.append(span)
                # Record disclosure as pre-verification if fewer than 2 factors
                # have been confirmed at the moment this turn is spoken
                factors_so_far = sum([dob_confirmed, address_confirmed, ssn_confirmed])
                if factors_so_far < 2:
                    pre_verification_disclosure.append(span)

        factors_confirmed = sum([dob_confirmed, address_confirmed, ssn_confirmed])

        if pre_verification_disclosure:
            if factors_confirmed >= 2:
                return ComplianceAnalysisResult(
                    call_id=transcript.call_id,
                    violation=ComplianceViolation.YES,
                    verification_status=PrivacyVerificationStatus.VERIFIED,
                    violation_type="ACCOUNT_DETAILS_BEFORE_VERIFICATION",
                    evidence=pre_verification_disclosure,
                    notes="Regex path found disclosure before two-factor verification was fully completed.",
                )
            return ComplianceAnalysisResult(
                call_id=transcript.call_id,
                violation=ComplianceViolation.YES,
                verification_status=PrivacyVerificationStatus.PARTIAL if factors_confirmed == 1 else PrivacyVerificationStatus.UNVERIFIED,
                violation_type="NO_VERIFICATION",
                evidence=pre_verification_disclosure,
                notes="Regex path found sensitive disclosure before two-factor verification.",
            )

        if factors_confirmed >= 2:
            final_status = PrivacyVerificationStatus.VERIFIED
        elif factors_confirmed == 1:
            final_status = PrivacyVerificationStatus.PARTIAL
        else:
            final_status = PrivacyVerificationStatus.NOT_APPLICABLE

        return ComplianceAnalysisResult(
            call_id=transcript.call_id,
            violation=ComplianceViolation.NO,
            verification_status=final_status,
            violation_type="NOT_APPLICABLE",
            evidence=evidence,
            notes="No pre-verification disclosure was detected by regex rules.",
        )

    def _contains_specific_disclosure(self, transcript: TranscriptFilePayload) -> bool:
        """Return True if any agent turn in the transcript contains a disclosure phrase.

        Args:
            transcript: Full call payload to scan.

        Returns:
            True if at least one agent turn matches a DISCLOSURE_PATTERNS entry.
        """
        return any(self._contains_disclosure_text(normalize_for_verification(turn.text)) for turn in transcript.turns if turn.speaker == SpeakerRole.AGENT)

    @staticmethod
    def _looks_like_dob_confirmation(normalized_text: str) -> bool:
        """Return True if the customer's utterance looks like a date-of-birth response.

        Args:
            normalized_text: Lowercase, punctuation-stripped turn text.

        Returns:
            True if any DOB pattern (slash, month-text, ordinal, or spelled-out) matches.
        """
        return any(p.search(normalized_text) for p in [
            DOB_SLASH_PATTERN,
            DOB_MONTH_TEXT_PATTERN,
            DOB_OF_MONTH_PATTERN,
            DOB_SPELLED_PATTERN,
        ])

    @staticmethod
    def _looks_like_address_confirmation(normalized_text: str) -> bool:
        """Return True if the customer's utterance looks like a street address response.

        Args:
            normalized_text: Lowercase, punctuation-stripped turn text.

        Returns:
            True if a digit-based or spelled-out address pattern matches.
        """
        return any(p.search(normalized_text) for p in [
            ADDRESS_DIGIT_PATTERN,
            ADDRESS_SPELLED_PATTERN,
        ])

    @staticmethod
    def _looks_like_ssn_confirmation(normalized_text: str) -> bool:
        """Return True if the customer's utterance looks like a Social Security Number.

        Args:
            normalized_text: Lowercase, punctuation-stripped turn text.

        Returns:
            True if a digit SSN or a nine-digit spelled-out pattern matches.
        """
        return any(p.search(normalized_text) for p in [
            SSN_DIGIT_PATTERN,
            SSN_SPELLED_PATTERN,
        ])

    @staticmethod
    def _contains_disclosure_text(normalized_text: str) -> bool:
        """Return True if the text contains a specific account disclosure phrase.

        Excludes generic balance references that lack personal-account context
        (e.g. "a balance" without "your", "owe", or "payment") to avoid false positives.

        Args:
            normalized_text: Lowercase, punctuation-stripped agent turn text.

        Returns:
            True if a DISCLOSURE_PATTERNS entry matches and the generic-balance filter
            does not suppress it.
        """
        if re.search(r"\b(a|the)\s+balance\b", normalized_text) and not re.search(r"\b(your|owe|payment)\b", normalized_text):
            return False
        return any(pattern.search(normalized_text) for pattern in DISCLOSURE_PATTERNS)

