from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SpeakerRole(str, Enum):
    AGENT = "AGENT"
    CUSTOMER = "CUSTOMER"
    UNKNOWN = "UNKNOWN"


class NormalizedTurn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    speaker: SpeakerRole
    text: str
    stime: float = Field(ge=0)
    etime: float = Field(ge=0)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_time_order(self) -> "NormalizedTurn":
        if self.etime < self.stime:
            raise ValueError("etime must be greater than or equal to stime")
        return self


class TranscriptValidationResult(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TranscriptFilePayload(BaseModel):
    call_id: str
    source_name: str
    turns: list[NormalizedTurn]
    validation: TranscriptValidationResult
    special_tags: list[str] = Field(default_factory=list)
    raw_text: str = ""

