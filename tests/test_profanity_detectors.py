from __future__ import annotations

from pathlib import Path

from call_evaluation.detectors.regex.profanity import RegexProfanityDetector
from call_evaluation.ingestion import IngestionService
from call_evaluation.models.transcript import SpeakerRole


FIXTURES = Path(__file__).parent / "fixtures"


def test_profanity_detector_handles_noisy_text_and_context() -> None:
    transcript = IngestionService().load_named_bytes(
        "noisy.json",
        b"""
        [
          {"speaker":"Customer","text":"u r so dum and fckn useless","stime":0,"etime":4},
          {"speaker":"Agent","text":"Please do not use that language.","stime":4.5,"etime":7}
        ]
        """,
    )[0]
    result = RegexProfanityDetector().analyze(transcript, SpeakerRole.CUSTOMER)
    assert result.flag is True
    assert result.context.value == "DIRECTED_AT_AGENT"
    assert result.severity.value in {"MODERATE", "SEVERE"}


def test_profanity_detector_ignores_clean_conversation() -> None:
    transcript = IngestionService().load_path(FIXTURES / "valid_transcript.yaml")[0]
    result = RegexProfanityDetector().analyze(transcript, SpeakerRole.AGENT)
    assert result.flag is False
    assert result.severity.value == "NONE"


def test_profanity_detector_does_not_use_generic_you_for_direction() -> None:
    transcript = IngestionService().load_named_bytes(
        "polite.json",
        b"""
        [
          {"speaker":"Agent","text":"Thank you for your time. I will help you with your account today.","stime":0,"etime":5},
          {"speaker":"Customer","text":"Thank you.","stime":5.5,"etime":6.5}
        ]
        """,
    )[0]
    result = RegexProfanityDetector().analyze(transcript, SpeakerRole.AGENT)
    assert result.flag is False


def test_profanity_false_positive_traps() -> None:
    detector = RegexProfanityDetector()
    cases = [
        ("thank you for your help today", False, "NONE", "AMBIENT"),
        ("what the heck is going on", False, "NONE", "AMBIENT"),
        ("she said he was a bastard to her", True, "MILD", "NARRATIVE_QUOTE"),
        ("this situation is really crappy weather-wise", False, "NONE", "AMBIENT"),
        ("i'm damn sure i paid that", True, "MILD", "SELF_EXPRESSION"),
        ("go to hell", True, "MILD", "AMBIENT"),
        ("my account number is 4 5 6 7 8 9", False, "NONE", "AMBIENT"),
    ]
    for text, expected_flag, expected_severity, expected_context in cases:
        transcript = IngestionService().load_named_bytes(
            "case.json",
            f'[{{"speaker":"Customer","text":"{text}","stime":0,"etime":3}}]'.encode(),
        )[0]
        result = detector.analyze(transcript, SpeakerRole.CUSTOMER)
        assert result.flag is expected_flag
        assert result.severity.value == expected_severity
        assert result.context.value == expected_context


def test_profanity_false_negative_traps() -> None:
    detector = RegexProfanityDetector()
    cases = [
        "this is absolute f*** nonsense",
        "you're such an a**hole",
        "what the s*** is wrong with you",
        "this is complete bullship",
        "b**** please",
    ]
    for text in cases:
        transcript = IngestionService().load_named_bytes(
            "case.json",
            f'[{{"speaker":"Customer","text":"{text}","stime":0,"etime":3}}]'.encode(),
        )[0]
        result = detector.analyze(transcript, SpeakerRole.CUSTOMER)
        assert result.flag is True


def test_profanity_context_classification_cases() -> None:
    detector = RegexProfanityDetector()
    cases = [
        ("Agent", "you're an idiot", "DIRECTED_AT_CUSTOMER", "SEVERE"),
        ("Customer", "you're useless", "DIRECTED_AT_AGENT", "MODERATE"),
        ("Customer", "i was so pissed off yesterday", "SELF_EXPRESSION", "MILD"),
        ("Agent", "he said she was a bitch", "NARRATIVE_QUOTE", "MILD"),
    ]
    for speaker, text, expected_context, expected_severity in cases:
        transcript = IngestionService().load_named_bytes(
            "case.json",
            f'[{{"speaker":"{speaker}","text":"{text}","stime":0,"etime":3}}]'.encode(),
        )[0]
        role = SpeakerRole.AGENT if speaker == "Agent" else SpeakerRole.CUSTOMER
        result = detector.analyze(transcript, role)
        assert result.flag is True
        assert result.context.value == expected_context
        assert result.severity.value == expected_severity
