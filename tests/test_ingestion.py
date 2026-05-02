from __future__ import annotations

import io
import zipfile
from pathlib import Path

from call_evaluation.ingestion import IngestionService


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_yaml_transcript() -> None:
    service = IngestionService()
    payload = service.load_path(FIXTURES / "valid_transcript.yaml")[0]
    assert payload.validation.is_valid is True
    assert payload.call_id == "valid_transcript"
    assert payload.turns[0].speaker.value == "AGENT"


def test_empty_transcript_returns_validation_error() -> None:
    service = IngestionService()
    payload = service.load_path(FIXTURES / "empty_transcript.json")[0]
    assert payload.validation.is_valid is False
    assert "Transcript file is empty." in payload.validation.errors


def test_zip_batch_supports_multiple_transcripts() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("one.json", (FIXTURES / "voicemail_transcript.json").read_text(encoding="utf-8"))
        archive.writestr("two.yaml", (FIXTURES / "valid_transcript.yaml").read_text(encoding="utf-8"))
    service = IngestionService()
    payloads = service.load_named_bytes("batch.zip", buffer.getvalue())
    assert len(payloads) == 2
    assert {payload.call_id for payload in payloads} == {"one", "two"}


def test_special_tags_detect_wrong_person_and_single_speaker() -> None:
    service = IngestionService()
    wrong_person = service.load_path(FIXTURES / "wrong_person_transcript.json")[0]
    single_speaker = service.load_path(FIXTURES / "single_speaker_transcript.json")[0]
    assert "WRONG_PERSON" in wrong_person.special_tags
    assert "SINGLE_SPEAKER" in single_speaker.special_tags
