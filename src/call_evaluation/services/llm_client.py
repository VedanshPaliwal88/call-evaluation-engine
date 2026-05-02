from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from call_evaluation.config import RuntimeState, get_settings
from call_evaluation.services.exceptions import LLMUnavailableError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_ENUM_SAFE_DEFAULTS: dict[str, tuple[set[str], str]] = {
    "violation": ({"YES", "NO"}, "NO"),
    "verification_status": ({"VERIFIED", "PARTIAL", "UNVERIFIED", "NOT_APPLICABLE"}, "UNVERIFIED"),
    "violation_type": ({"NO_VIOLATION", "NO_VERIFICATION", "ACCOUNT_DETAILS_BEFORE_VERIFICATION", "NOT_APPLICABLE"}, "NOT_APPLICABLE"),
    "severity": ({"NONE", "MILD", "MODERATE", "SEVERE"}, "NONE"),
    "context": ({"DIRECTED_AT_AGENT", "DIRECTED_AT_CUSTOMER", "SELF_EXPRESSION", "NARRATIVE_QUOTE", "AMBIENT"}, "AMBIENT"),
    "sentiment": ({"POSITIVE", "NEUTRAL", "NEGATIVE"}, "NEUTRAL"),
}


def _sanitize_enum_fields(parsed: dict) -> dict:
    """Coerce any out-of-range enum values to safe defaults before Pydantic sees them."""
    for field, (allowed, default) in _ENUM_SAFE_DEFAULTS.items():
        if field not in parsed:
            continue
        value = parsed[field]
        if isinstance(value, str) and value.upper() in allowed:
            parsed[field] = value.upper()
        elif value not in allowed:
            logger.warning(
                "LLM returned unexpected value %r for field %r — falling back to %r",
                value,
                field,
                default,
            )
            parsed[field] = default
    return parsed


class LLMClient:
    """Centralized wrapper for structured LLM calls."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._import_error: str | None = None

        if not self.settings.openai_api_key:
            self._import_error = "Missing OPENAI_API_KEY. LLM features are unavailable."
            return

        try:  # pragma: no cover - depends on environment
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on environment
            self._import_error = f"OpenAI SDK is not installed: {exc}"
            return

        self._client = OpenAI(api_key=self.settings.openai_api_key)

    def get_runtime_state(self) -> RuntimeState:
        return RuntimeState(
            api_available=self._client is not None,
            model_name=self.settings.llm_model,
            message=self._import_error,
        )

    def require_available(self) -> None:
        state = self.get_runtime_state()
        if not state.api_available:
            raise LLMUnavailableError(state.message or "LLM support is unavailable.")

    def load_prompt_template(self, prompt_name: str) -> str:
        prompt_path = Path(self.settings.prompt_dir) / prompt_name
        return prompt_path.read_text(encoding="utf-8")

    def render_prompt(self, prompt_name: str, replacements: dict[str, str]) -> str:
        prompt = self.load_prompt_template(prompt_name)
        for key, value in replacements.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", value)
        return prompt

    def classify_json(self, prompt: str, response_model: type[ModelT]) -> ModelT:
        self.require_available()
        assert self._client is not None

        try:
            response = self._client.chat.completions.create(
                model=self.settings.llm_model,
                temperature=0.0,
                timeout=30.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "timeout" in exc_str or "timed out" in exc_str:
                logger.warning("LLM request timed out — returning safe fallback result.")
                return self._safe_fallback(response_model, "LLM request timed out")
            raise LLMUnavailableError(f"LLM request failed: {exc}") from exc

        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMUnavailableError(f"LLM response was not valid JSON: {exc}") from exc

        parsed = _sanitize_enum_fields(parsed)

        try:
            return response_model.model_validate(parsed)
        except ValidationError as exc:
            logger.warning("LLM response failed schema validation after sanitization: %s", exc)
            return self._safe_fallback(response_model, "LLM response failed schema validation")

    def _safe_fallback(self, response_model: type[ModelT], reason: str) -> ModelT:
        """Build a minimal valid instance of response_model using safe defaults."""
        defaults: dict[str, object] = {
            "flag": False,
            "speaker_role": "UNKNOWN",
            "severity": "NONE",
            "sentiment": "NEUTRAL",
            "context": "AMBIENT",
            "violation": "NO",
            "verification_status": "UNVERIFIED",
            "violation_type": "NOT_APPLICABLE",
            "evidence": [],
            "notes": reason,
        }
        return response_model.model_validate(defaults)
