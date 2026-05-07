"""OpenAI client wrapper with structured JSON output, enum sanitization, and timeout fallback."""
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

_LLM_TIMEOUT_SECONDS = 30.0

try:  # openai may not be installed in all environments
    from openai import APITimeoutError as _OpenAITimeoutError
except ImportError:  # pragma: no cover
    _OpenAITimeoutError = None  # type: ignore[assignment,misc]

_ENUM_SAFE_DEFAULTS: dict[str, tuple[set[str], str]] = {
    "violation": ({"YES", "NO"}, "NO"),
    "verification_status": ({"VERIFIED", "PARTIAL", "UNVERIFIED", "NOT_APPLICABLE"}, "UNVERIFIED"),
    "violation_type": ({"NO_VERIFICATION", "ACCOUNT_DETAILS_BEFORE_VERIFICATION", "NOT_APPLICABLE"}, "NOT_APPLICABLE"),
    "severity": ({"NONE", "MILD", "MODERATE", "SEVERE"}, "NONE"),
    "context": ({"DIRECTED_AT_AGENT", "DIRECTED_AT_CUSTOMER", "SELF_EXPRESSION", "NARRATIVE_QUOTE", "AMBIENT"}, "AMBIENT"),
    "sentiment": ({"POSITIVE", "NEUTRAL", "NEGATIVE"}, "NEUTRAL"),
}


def _sanitize_enum_fields(parsed: dict) -> dict:
    """Coerce any out-of-range enum values to safe defaults before Pydantic sees them.

    The LLM occasionally returns values that differ in case or are entirely
    out-of-vocabulary (e.g. a made-up violation_type). Pydantic would raise a
    ValidationError on those, which would bypass the structured-output contract.
    This function fixes them in-place before validation so the result model is
    always constructable.

    Args:
        parsed: Raw dict from json.loads of the LLM response.

    Returns:
        The same dict with enum fields normalized to uppercase valid values or
        replaced with their designated safe default.
    """
    for field, (allowed, default) in _ENUM_SAFE_DEFAULTS.items():
        if field not in parsed:
            continue
        value = parsed[field]
        # Attempt case-insensitive match first — LLM may return "yes" instead of "YES"
        if isinstance(value, str) and value.upper() in allowed:
            parsed[field] = value.upper()
        elif value not in allowed:
            # Value is genuinely out-of-vocabulary (None, int, unexpected string)
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
        """Return a snapshot indicating whether the LLM client is ready to make calls.

        Returns:
            RuntimeState with api_available flag, model name, and optional error message.
        """
        return RuntimeState(
            api_available=self._client is not None,
            model_name=self.settings.llm_model,
            message=self._import_error,
        )

    def require_available(self) -> None:
        """Raise LLMUnavailableError if the client was not successfully initialized.

        Raises:
            LLMUnavailableError: If OPENAI_API_KEY is missing or the SDK is not installed.
        """
        state = self.get_runtime_state()
        if not state.api_available:
            raise LLMUnavailableError(state.message or "LLM support is unavailable.")

    def load_prompt_template(self, prompt_name: str) -> str:
        """Read a prompt template file from the configured prompt directory.

        Args:
            prompt_name: Filename (e.g. "profanity_v1.txt") relative to prompt_dir.

        Returns:
            Raw template string with {{PLACEHOLDER}} tokens.
        """
        prompt_path = Path(self.settings.prompt_dir) / prompt_name
        return prompt_path.read_text(encoding="utf-8")

    def render_prompt(self, prompt_name: str, replacements: dict[str, str]) -> str:
        """Load a prompt template and substitute all {{KEY}} placeholders.

        Args:
            prompt_name: Template filename to load via load_prompt_template.
            replacements: Mapping of placeholder names to their substitution values.

        Returns:
            Fully rendered prompt string ready to send to the LLM.
        """
        prompt = self.load_prompt_template(prompt_name)
        for key, value in replacements.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", value)
        return prompt

    def classify_json(self, prompt: str, response_model: type[ModelT]) -> ModelT:
        """Send a prompt to the LLM and parse the JSON response into response_model.

        Applies enum sanitization before Pydantic validation and falls back to a
        safe default instance on timeout or schema validation failure.

        Args:
            prompt: Fully rendered prompt string.
            response_model: Pydantic model class to parse the LLM JSON output into.

        Returns:
            A validated instance of response_model.

        Raises:
            LLMUnavailableError: If the client is unavailable or the API call fails
                for a non-timeout reason.
        """
        self.require_available()
        if self._client is None:
            raise LLMUnavailableError("LLM client is not initialized.")

        try:
            response = self._client.chat.completions.create(
                model=self.settings.llm_model,
                temperature=0.0,
                timeout=_LLM_TIMEOUT_SECONDS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            if self._is_timeout(exc):
                logger.warning("LLM request timed out after %ss — returning safe fallback result.", _LLM_TIMEOUT_SECONDS)
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

    @staticmethod
    def _is_timeout(exc: Exception) -> bool:
        """Return True if the exception represents a request timeout.

        Checks for the OpenAI SDK's specific timeout type first, then falls back
        to string matching for environments where the SDK import failed.

        Args:
            exc: Exception raised by the OpenAI API call.

        Returns:
            True if the exception is a timeout, False otherwise.
        """
        if _OpenAITimeoutError is not None and isinstance(exc, _OpenAITimeoutError):
            return True
        return "timeout" in str(exc).lower() or "timed out" in str(exc).lower()

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
