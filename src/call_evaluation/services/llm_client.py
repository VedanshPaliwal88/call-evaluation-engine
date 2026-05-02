from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from call_evaluation.config import RuntimeState, get_settings
from call_evaluation.services.exceptions import LLMUnavailableError

ModelT = TypeVar("ModelT", bound=BaseModel)


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

        response = self._client.chat.completions.create(
            model=self.settings.llm_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
            return response_model.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMUnavailableError(f"LLM response failed schema validation: {exc}") from exc
