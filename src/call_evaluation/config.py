from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at import time
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    openai_api_key: str | None = Field(default=None)
    llm_model: str = Field(default="gpt-4o")
    prompt_dir: Path = ROOT_DIR / "src" / "call_evaluation" / "detectors" / "llm" / "prompts"
    data_dir: Path = ROOT_DIR / "data"


class RuntimeState(BaseModel):
    api_available: bool
    model_name: str
    message: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
    )
