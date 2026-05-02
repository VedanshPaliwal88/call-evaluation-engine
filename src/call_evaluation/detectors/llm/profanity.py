from __future__ import annotations

from call_evaluation.models.analysis import ProfanityAnalysisResult
from call_evaluation.models.transcript import SpeakerRole, TranscriptFilePayload
from call_evaluation.services.llm_client import LLMClient


class LLMProfanityDetector:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def analyze(self, transcript: TranscriptFilePayload, target_role: SpeakerRole) -> ProfanityAnalysisResult:
        prompt = self.llm_client.render_prompt(
            "profanity_v1.txt",
            {
                "TRANSCRIPT": transcript.raw_text,
                "TARGET_SPEAKER_ROLE": target_role.value,
            },
        )
        result = self.llm_client.classify_json(prompt, ProfanityAnalysisResult)
        return result.model_copy(update={"call_id": transcript.call_id})
