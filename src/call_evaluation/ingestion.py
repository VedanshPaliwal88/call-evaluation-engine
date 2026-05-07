"""Transcript ingestion — parse JSON/YAML files and ZIP archives into validated payloads."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from call_evaluation.models.analysis import SpecialCallType
from call_evaluation.models.transcript import (
    NormalizedTurn,
    SpeakerRole,
    TranscriptFilePayload,
    TranscriptValidationResult,
)

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency until installed
    yaml = None


def _normalize_speaker(value: str | None) -> SpeakerRole:
    cleaned = (value or "").strip().lower()
    if cleaned == "agent":
        return SpeakerRole.AGENT
    if cleaned in {"customer", "borrower"}:
        return SpeakerRole.CUSTOMER
    return SpeakerRole.UNKNOWN


def _detect_special_tags(turns: list[NormalizedTurn]) -> list[str]:
    if not turns:
        return [SpecialCallType.EMPTY.value]

    speakers = {turn.speaker for turn in turns if turn.speaker != SpeakerRole.UNKNOWN}
    normalized_text = " ".join(turn.text.lower() for turn in turns)
    tags: list[str] = []

    if len(speakers) <= 1:
        tags.append(SpecialCallType.SINGLE_SPEAKER.value)

    voicemail_markers = ["voicemail", "leave a message", "call us back", "return my call", "please call back"]
    wrong_person_markers = ["wrong person", "wrong number", "remove this number", "do not know"]
    if any(marker in normalized_text for marker in voicemail_markers) and SpeakerRole.CUSTOMER not in speakers:
        tags.append(SpecialCallType.VOICEMAIL.value)
    if any(marker in normalized_text for marker in wrong_person_markers):
        tags.append(SpecialCallType.WRONG_PERSON.value)
    if not tags:
        tags.append(SpecialCallType.STANDARD.value)
    return tags


def _to_payload(call_id: str, source_name: str, raw_items: list[dict]) -> TranscriptFilePayload:
    errors: list[str] = []
    turns: list[NormalizedTurn] = []

    if not raw_items:
        errors.append("Transcript file is empty.")

    for index, item in enumerate(raw_items):
        try:
            turns.append(
                NormalizedTurn(
                    speaker=_normalize_speaker(item.get("speaker")),
                    text=str(item.get("text", "")),
                    stime=float(item.get("stime", 0)),
                    etime=float(item.get("etime", 0)),
                )
            )
        except (TypeError, ValueError, ValidationError) as exc:
            errors.append(f"Turn {index} is invalid: {exc}")

    validation = TranscriptValidationResult(is_valid=not errors, errors=errors, warnings=[])
    raw_text = "\n".join(f"{turn.speaker.value}: {turn.text}" for turn in turns)
    return TranscriptFilePayload(
        call_id=call_id,
        source_name=source_name,
        turns=turns,
        validation=validation,
        special_tags=_detect_special_tags(turns),
        raw_text=raw_text,
    )


class IngestionService:
    """Load transcript files from JSON, YAML, or ZIP containers."""

    SUPPORTED_SUFFIXES = {".json", ".yaml", ".yml"}

    def load_path(self, path: Path) -> list[TranscriptFilePayload]:
        """Load transcripts from a file system path.

        Args:
            path: Path to a JSON, YAML, or ZIP file.

        Returns:
            List of payloads; a ZIP expands to multiple; a single file yields one.
        """
        return self.load_named_bytes(path.name, path.read_bytes())

    def load_named_bytes(self, name: str, raw_bytes: bytes) -> list[TranscriptFilePayload]:
        """Parse raw bytes with the given filename into transcript payloads.

        Args:
            name: Original filename used to infer format and derive call_id.
            raw_bytes: File contents.

        Returns:
            One payload for JSON/YAML; multiple for ZIP; an invalid payload on errors.
        """
        suffix = Path(name).suffix.lower()
        if suffix == ".zip":
            return self._load_zip(raw_bytes)
        if suffix not in self.SUPPORTED_SUFFIXES:
            return [self._invalid_payload(name, f"Unsupported file type: {suffix or 'unknown'}")]
        return [self._load_single(name, raw_bytes)]

    def load_batch(self, items: Iterable[tuple[str, bytes]]) -> list[TranscriptFilePayload]:
        """Load multiple files from an iterable of (name, raw_bytes) pairs.

        Args:
            items: Iterable of (filename, file bytes) tuples.

        Returns:
            Flat list of all payloads from all files, including validation failures.
        """
        payloads: list[TranscriptFilePayload] = []
        for name, raw_bytes in items:
            payloads.extend(self.load_named_bytes(name, raw_bytes))
        return payloads

    def _load_zip(self, raw_bytes: bytes) -> list[TranscriptFilePayload]:
        payloads: list[TranscriptFilePayload] = []
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_name = Path(member.filename).name
                suffix = Path(member_name).suffix.lower()
                if suffix not in self.SUPPORTED_SUFFIXES:
                    continue
                payloads.append(self._load_single(member_name, archive.read(member)))
        if not payloads:
            payloads.append(self._invalid_payload("archive.zip", "ZIP file contains no supported transcript files."))
        return payloads

    def _load_single(self, name: str, raw_bytes: bytes) -> TranscriptFilePayload:
        call_id = Path(name).stem
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return self._invalid_payload(name, "File is not valid UTF-8.")

        suffix = Path(name).suffix.lower()
        try:
            if suffix == ".json":
                data = json.loads(text)
            else:
                if yaml is None:
                    return self._invalid_payload(name, "PyYAML is required for YAML support but is not installed.")
                data = yaml.safe_load(text)
        except Exception as exc:
            return self._invalid_payload(name, f"Could not parse file: {exc}")

        if not isinstance(data, list):
            return self._invalid_payload(name, "Transcript root must be a list of utterances.")
        return _to_payload(call_id=call_id, source_name=name, raw_items=data)

    def _invalid_payload(self, name: str, error: str) -> TranscriptFilePayload:
        return TranscriptFilePayload(
            call_id=Path(name).stem,
            source_name=name,
            turns=[],
            validation=TranscriptValidationResult(is_valid=False, errors=[error], warnings=[]),
            special_tags=[SpecialCallType.EMPTY.value],
            raw_text="",
        )

