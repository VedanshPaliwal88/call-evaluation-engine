"""High-level orchestration layer for batch profanity, compliance, and metrics analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from call_evaluation.detectors.llm.compliance import LLMComplianceDetector
from call_evaluation.detectors.llm.profanity import LLMProfanityDetector
from call_evaluation.detectors.regex.compliance import RegexComplianceDetector
from call_evaluation.detectors.regex.profanity import RegexProfanityDetector
from call_evaluation.ingestion import IngestionService
from call_evaluation.metrics.call_metrics import MetricsService
from call_evaluation.models.analysis import (
    AnalysisApproach,
    BatchProcessingReport,
    ComplianceAnalysisResult,
    ContextLabel,
    ProfanityAnalysisResult,
    SeverityLabel,
)
from call_evaluation.models.transcript import SpeakerRole, TranscriptFilePayload


@dataclass
class ProfanityRowDetail:
    """Per-call profanity results for both speakers, plus source metadata."""

    call_id: str
    agent: ProfanityAnalysisResult
    customer: ProfanityAnalysisResult
    source_name: str
    special_tags: list[str]


@dataclass
class ComplianceRowDetail:
    """Per-call compliance result plus source metadata."""

    call_id: str
    result: ComplianceAnalysisResult
    source_name: str
    special_tags: list[str]


@dataclass
class ProfanityBatchResult:
    """Aggregated profanity results for a batch run: summary rows, report, and drill-down details."""

    rows: list[dict]
    report: BatchProcessingReport
    details: dict[str, ProfanityRowDetail] = field(default_factory=dict)


@dataclass
class ComplianceBatchResult:
    """Aggregated compliance results for a batch run: summary rows, report, and drill-down details."""

    rows: list[dict]
    report: BatchProcessingReport
    details: dict[str, ComplianceRowDetail] = field(default_factory=dict)


@dataclass
class MetricsBatchResult:
    """Aggregated metrics results for a batch run: summary rows and report."""

    rows: list[dict]
    report: BatchProcessingReport


class AnalysisService:
    """Facade that wires together ingestion, detection, and metrics for the UI and CLI tools."""

    def __init__(self) -> None:
        self.ingestion = IngestionService()
        self.regex_profanity = RegexProfanityDetector()
        self.regex_compliance = RegexComplianceDetector()
        self.llm_profanity = LLMProfanityDetector()
        self.llm_compliance = LLMComplianceDetector()
        self.metrics = MetricsService()

    def get_llm_runtime_state(self):
        """Return the current LLM availability state for display in the UI sidebar.

        Returns:
            RuntimeState indicating whether the OpenAI client is ready.
        """
        return self.llm_profanity.llm_client.get_runtime_state()

    def load_inputs(self, items: list[tuple[str, bytes]]) -> list[TranscriptFilePayload]:
        """Parse a list of (filename, raw_bytes) pairs into transcript payloads.

        Args:
            items: File names paired with their raw byte contents.

        Returns:
            List of TranscriptFilePayload, one per parsed file (including failures).
        """
        return self.ingestion.load_batch(items)

    def load_dataset_dir(self, directory: Path) -> list[TranscriptFilePayload]:
        """Load all transcript files from a directory path.

        Args:
            directory: Path to a folder containing JSON or YAML transcript files.

        Returns:
            List of TranscriptFilePayload sorted by filename.
        """
        items = [(path.name, path.read_bytes()) for path in sorted(directory.glob("*")) if path.is_file()]
        return self.load_inputs(items)

    def analyze_profanity(self, transcripts: list[TranscriptFilePayload], approach: AnalysisApproach) -> ProfanityBatchResult:
        """Run profanity detection over a batch of transcripts for both speakers.

        Args:
            transcripts: Loaded payloads from load_inputs or load_dataset_dir.
            approach: REGEX for pattern matching or LLM for model-based detection.

        Returns:
            ProfanityBatchResult with per-call summary rows, a processing report,
            and per-call detail objects keyed by call_id.
        """
        rows: list[dict] = []
        errors: dict[str, list[str]] = {}
        details: dict[str, ProfanityRowDetail] = {}
        success = 0
        detector = self.regex_profanity if approach == AnalysisApproach.REGEX else self.llm_profanity

        for transcript in transcripts:
            if not transcript.validation.is_valid:
                errors[transcript.call_id] = transcript.validation.errors
                continue

            agent = detector.analyze(transcript, SpeakerRole.AGENT)
            customer = detector.analyze(transcript, SpeakerRole.CUSTOMER)
            overall = self._summarize_profanity(agent, customer)
            rows.append(
                {
                    "call_id": transcript.call_id,
                    "agent_profanity": agent.flag,
                    "customer_profanity": customer.flag,
                    "severity": overall["severity"],
                    "context": overall["context"],
                    "special_tags": ", ".join(transcript.special_tags),
                }
            )
            details[transcript.call_id] = ProfanityRowDetail(
                call_id=transcript.call_id,
                agent=agent,
                customer=customer,
                source_name=transcript.source_name,
                special_tags=transcript.special_tags,
            )
            success += 1

        report = BatchProcessingReport(
            processed_calls=len(transcripts),
            successful_calls=success,
            failed_calls=len(transcripts) - success,
            errors=errors,
        )
        return ProfanityBatchResult(rows=rows, report=report, details=details)

    def analyze_compliance(self, transcripts: list[TranscriptFilePayload], approach: AnalysisApproach) -> ComplianceBatchResult:
        """Run compliance detection over a batch of transcripts.

        Args:
            transcripts: Loaded payloads from load_inputs or load_dataset_dir.
            approach: REGEX for pattern matching or LLM for model-based detection.

        Returns:
            ComplianceBatchResult with per-call summary rows, a processing report,
            and per-call detail objects keyed by call_id.
        """
        rows: list[dict] = []
        errors: dict[str, list[str]] = {}
        details: dict[str, ComplianceRowDetail] = {}
        success = 0
        detector = self.regex_compliance if approach == AnalysisApproach.REGEX else self.llm_compliance

        for transcript in transcripts:
            if not transcript.validation.is_valid:
                errors[transcript.call_id] = transcript.validation.errors
                continue

            result = detector.analyze(transcript)
            rows.append(
                {
                    "call_id": transcript.call_id,
                    "violation": result.violation.value,
                    "verification_status": result.verification_status.value,
                    "violation_type": result.violation_type,
                    "special_tags": ", ".join(transcript.special_tags),
                }
            )
            details[transcript.call_id] = ComplianceRowDetail(
                call_id=transcript.call_id,
                result=result,
                source_name=transcript.source_name,
                special_tags=transcript.special_tags,
            )
            success += 1

        report = BatchProcessingReport(
            processed_calls=len(transcripts),
            successful_calls=success,
            failed_calls=len(transcripts) - success,
            errors=errors,
        )
        return ComplianceBatchResult(rows=rows, report=report, details=details)

    def analyze_metrics(self, transcripts: list[TranscriptFilePayload]) -> MetricsBatchResult:
        """Compute timing metrics for a batch of transcripts.

        Args:
            transcripts: Loaded payloads from load_inputs or load_dataset_dir.

        Returns:
            MetricsBatchResult with per-call metric rows and a processing report.
        """
        rows: list[dict] = []
        errors: dict[str, list[str]] = {}
        success = 0

        for transcript in transcripts:
            if not transcript.validation.is_valid:
                errors[transcript.call_id] = transcript.validation.errors
                continue

            result = self.metrics.analyze(transcript)
            rows.append(result.model_dump())
            success += 1

        report = BatchProcessingReport(
            processed_calls=len(transcripts),
            successful_calls=success,
            failed_calls=len(transcripts) - success,
            errors=errors,
        )
        return MetricsBatchResult(rows=rows, report=report)

    @staticmethod
    def _summarize_profanity(agent: ProfanityAnalysisResult, customer: ProfanityAnalysisResult) -> dict[str, str]:
        """Derive a single call-level severity and context from both speaker results.

        Args:
            agent: Profanity result for the agent speaker.
            customer: Profanity result for the customer speaker.

        Returns:
            Dict with "severity" and "context" from the higher-severity speaker,
            or NONE/AMBIENT if neither speaker was flagged.
        """
        ranked = sorted(
            [agent, customer],
            key=lambda result: (
                AnalysisService._severity_rank(result.severity),
                len(result.evidence),
            ),
            reverse=True,
        )
        top = ranked[0]
        if not top.flag:
            return {"severity": SeverityLabel.NONE.value, "context": ContextLabel.AMBIENT.value}
        return {"severity": top.severity.value, "context": top.context.value}

    @staticmethod
    def _severity_rank(severity: SeverityLabel) -> int:
        ranks = {
            SeverityLabel.NONE: 0,
            SeverityLabel.MILD: 1,
            SeverityLabel.MODERATE: 2,
            SeverityLabel.SEVERE: 3,
        }
        return ranks[severity]
