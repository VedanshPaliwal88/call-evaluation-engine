from __future__ import annotations

import re

from call_evaluation.models.analysis import (
    ContextLabel,
    EvidenceSpan,
    ProfanityAnalysisResult,
    SentimentLabel,
    SeverityLabel,
)
from call_evaluation.models.transcript import SpeakerRole, TranscriptFilePayload
from call_evaluation.utils.text import normalize_for_matching


PROFANITY_PATTERNS = {
    "fuck": re.compile(r"\b(fuck|fucking|fucker)\b"),
    "shit": re.compile(r"\b(shit|shitty)\b"),
    "damn": re.compile(r"\b(damn|dammit)\b"),
    "hell": re.compile(r"\bhell\b"),
    "crap": re.compile(r"\bcrap\b"),
    "bullshit": re.compile(r"\b(bull shit|bullshit)\b"),
    "bastard": re.compile(r"\b(bastard)\b"),
    "bitch": re.compile(r"\b(bitch)\b"),
    "asshole": re.compile(r"\b(asshole)\b"),
    "idiot": re.compile(r"\b(idiot)\b"),
    "stupid": re.compile(r"\b(stupid|dumb|moron|useless|pissed off)\b"),
    "screw": re.compile(r"\bscrew you\b"),
}

NARRATIVE_PATTERNS = [
    re.compile(r"\bhe said\b"),
    re.compile(r"\bshe said\b"),
    re.compile(r"\bthey called me\b"),
    re.compile(r"\byesterday\b"),
]


def _severity_from_hits(hits: list[str], context: ContextLabel) -> SeverityLabel:
    if not hits:
        return SeverityLabel.NONE
    if context == ContextLabel.AMBIENT and hits == ["hell"]:
        return SeverityLabel.MILD
    if context == ContextLabel.SELF_EXPRESSION and "damn" in hits and len(hits) == 1:
        return SeverityLabel.MILD
    if context in {ContextLabel.DIRECTED_AT_AGENT, ContextLabel.DIRECTED_AT_CUSTOMER}:
        if any(hit in {"fuck", "asshole", "bastard", "bullshit", "bitch", "idiot"} for hit in hits) or len(hits) >= 2:
            return SeverityLabel.SEVERE
        return SeverityLabel.MODERATE
    if any(hit in {"fuck", "shit"} for hit in hits):
        return SeverityLabel.MODERATE
    return SeverityLabel.MILD


def _context_from_text(text: str, speaker_role: SpeakerRole) -> ContextLabel:
    if re.search(r"\b(i|i am|i'm|me|my)\b", text):
        return ContextLabel.SELF_EXPRESSION
    if any(pattern.search(text) for pattern in NARRATIVE_PATTERNS):
        return ContextLabel.NARRATIVE_QUOTE
    if re.search(r"\b(go to hell|hell no)\b", text):
        return ContextLabel.AMBIENT
    if speaker_role == SpeakerRole.CUSTOMER:
        return ContextLabel.DIRECTED_AT_AGENT
    if speaker_role == SpeakerRole.AGENT:
        return ContextLabel.DIRECTED_AT_CUSTOMER
    return ContextLabel.AMBIENT


class RegexProfanityDetector:
    def analyze(self, transcript: TranscriptFilePayload, target_role: SpeakerRole) -> ProfanityAnalysisResult:
        evidence: list[EvidenceSpan] = []
        hits: list[str] = []
        context = ContextLabel.AMBIENT

        for turn in transcript.turns:
            if turn.speaker != target_role:
                continue
            normalized = normalize_for_matching(turn.text)
            matched_terms = [label for label, pattern in PROFANITY_PATTERNS.items() if pattern.search(normalized)]
            if not matched_terms:
                continue
            context = _context_from_text(normalized, target_role)
            hits.extend(matched_terms)
            evidence.append(
                EvidenceSpan(
                    speaker_role=turn.speaker.value,
                    text=turn.text,
                    stime=turn.stime,
                    etime=turn.etime,
                    reason=f"Matched normalized profanity terms: {', '.join(matched_terms)}",
                )
            )

        severity = _severity_from_hits(hits, context)
        flag = bool(hits)
        return ProfanityAnalysisResult(
            call_id=transcript.call_id,
            flag=flag,
            speaker_role=target_role.value,
            severity=severity,
            sentiment=SentimentLabel.NEGATIVE if flag else SentimentLabel.NEUTRAL,
            context=context if flag else ContextLabel.AMBIENT,
            evidence=evidence,
            notes="Regex-based context-aware profanity detection.",
        )
