"""LLM-backed compliance detector using the compliance_v1.txt prompt template."""
from __future__ import annotations

from call_evaluation.models.analysis import ComplianceAnalysisResult
from call_evaluation.models.transcript import TranscriptFilePayload
from call_evaluation.services.llm_client import LLMClient


class LLMComplianceDetector:
    """Detect compliance violations via a structured LLM call using the compliance_v1 prompt."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def analyze(self, transcript: TranscriptFilePayload) -> ComplianceAnalysisResult:
        """Analyze a transcript for privacy and compliance violations.

        Args:
            transcript: Validated transcript payload with raw_text for the prompt.

        Returns:
            ComplianceAnalysisResult with violation, verification_status, and evidence spans.
        """
        prompt = self.llm_client.render_prompt(
            "compliance_v1.txt",
            {
                "TRANSCRIPT": transcript.raw_text,
            },
        )
        result = self.llm_client.classify_json(prompt, ComplianceAnalysisResult)
        return result.model_copy(update={"call_id": transcript.call_id})
