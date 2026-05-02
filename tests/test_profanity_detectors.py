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
