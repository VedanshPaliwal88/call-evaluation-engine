from __future__ import annotations

from pathlib import Path

from call_evaluation.ingestion import IngestionService
from call_evaluation.models.analysis import SpecialCallType
from call_evaluation.models.transcript import (
    NormalizedTurn,
    SpeakerRole,
    TranscriptFilePayload,
    TranscriptValidationResult,
)
from call_evaluation.metrics.call_metrics import MetricsService
from call_evaluation.visualization import (
    create_distribution_histograms,
    create_metrics_box_plot,
    create_metrics_scatter_plot,
    create_top_n_figure,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _build_payload(call_id: str, turns: list[NormalizedTurn], special_tags: list[str] | None = None) -> TranscriptFilePayload:
    return TranscriptFilePayload(
        call_id=call_id,
        source_name=f"{call_id}.json",
        turns=turns,
        validation=TranscriptValidationResult(is_valid=True, errors=[], warnings=[]),
        special_tags=special_tags or [SpecialCallType.STANDARD.value],
        raw_text="\n".join(f"{turn.speaker.value}: {turn.text}" for turn in turns),
    )


def test_metrics_handles_single_speaker_without_crashing() -> None:
    transcript = IngestionService().load_path(FIXTURES / "single_speaker_transcript.json")[0]
    result = MetricsService().analyze(transcript)
    assert result.special_case == "SINGLE_SPEAKER"
    assert result.overtalk_pct == 0.0


def test_metrics_handles_zero_overtalk() -> None:
    transcript = IngestionService().load_path(FIXTURES / "valid_transcript.yaml")[0]
    result = MetricsService().analyze(transcript)
    assert result.overtalk_pct == 0.0
    assert result.total_duration > 0


def test_metrics_returns_zeroes_for_empty_transcript() -> None:
    transcript = IngestionService().load_path(FIXTURES / "empty_transcript.json")[0]
    result = MetricsService().analyze(transcript)
    assert result.special_case == "EMPTY_TRANSCRIPT"
    assert result.total_duration == 0.0
    assert result.agent_talk_time == 0.0
    assert result.customer_talk_time == 0.0
    assert result.silence_pct == 0.0
    assert result.overtalk_pct == 0.0


def test_metrics_marks_voicemail_without_crashing() -> None:
    transcript = IngestionService().load_path(FIXTURES / "voicemail_transcript.json")[0]
    result = MetricsService().analyze(transcript)
    assert result.special_case == "VOICEMAIL"
    assert result.customer_talk_time == 0.0
    assert result.overtalk_pct == 0.0


def test_metrics_computes_overlap_and_silence_from_intervals() -> None:
    transcript = _build_payload(
        "overlap-call",
        [
            NormalizedTurn(speaker=SpeakerRole.AGENT, text="Hello", stime=0, etime=5),
            NormalizedTurn(speaker=SpeakerRole.CUSTOMER, text="Hi", stime=3, etime=7),
            NormalizedTurn(speaker=SpeakerRole.AGENT, text="One more thing", stime=9, etime=10),
        ],
    )
    result = MetricsService().analyze(transcript)
    assert result.total_duration == 10.0
    assert result.agent_talk_time == 6.0
    assert result.customer_talk_time == 4.0
    assert result.overtalk_pct == 20.0
    assert result.silence_pct == 20.0


def test_metrics_handles_zero_duration_without_divide_by_zero() -> None:
    transcript = _build_payload(
        "zero-duration",
        [NormalizedTurn(speaker=SpeakerRole.AGENT, text="Ping", stime=2, etime=2)],
        special_tags=[SpecialCallType.SINGLE_SPEAKER.value],
    )
    result = MetricsService().analyze(transcript)
    assert result.special_case == "ZERO_DURATION"
    assert result.agent_talk_pct == 0.0
    assert result.customer_talk_pct == 0.0


def test_visualizations_return_expected_plotly_objects() -> None:
    rows = [
        {
            "call_id": "call-a",
            "agent_talk_pct": 40.0,
            "customer_talk_pct": 35.0,
            "silence_pct": 20.0,
            "overtalk_pct": 5.0,
        },
        {
            "call_id": "call-b",
            "agent_talk_pct": 55.0,
            "customer_talk_pct": 25.0,
            "silence_pct": 15.0,
            "overtalk_pct": 5.0,
        },
    ]
    box_plot = create_metrics_box_plot(rows)
    scatter_plot = create_metrics_scatter_plot(rows)
    distributions = create_distribution_histograms(rows)
    top_silence = create_top_n_figure(rows, "silence_pct", top_n=1)
    top_overtalk = create_top_n_figure(rows, "overtalk_pct", top_n=1)

    assert box_plot is not None
    assert len(box_plot.data) == 2
    assert scatter_plot is not None
    assert len(scatter_plot.data) == 1
    assert set(distributions.keys()) == {"silence_pct", "overtalk_pct"}
    assert top_silence is not None
    assert top_overtalk is not None
