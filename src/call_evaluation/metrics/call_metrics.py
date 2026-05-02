from __future__ import annotations

from call_evaluation.models.analysis import MetricResult
from call_evaluation.models.transcript import SpeakerRole, TranscriptFilePayload


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _sum_intervals(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _intersection_duration(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    left = _merge_intervals(left)
    right = _merge_intervals(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            total += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


class MetricsService:
    def analyze(self, transcript: TranscriptFilePayload) -> MetricResult:
        if not transcript.turns:
            return MetricResult(call_id=transcript.call_id, special_case="EMPTY_TRANSCRIPT")

        start = min(turn.stime for turn in transcript.turns)
        end = max(turn.etime for turn in transcript.turns)
        total_duration = max(end - start, 0.0)
        if total_duration == 0:
            return MetricResult(call_id=transcript.call_id, special_case="ZERO_DURATION")

        agent_intervals = [(turn.stime, turn.etime) for turn in transcript.turns if turn.speaker == SpeakerRole.AGENT]
        customer_intervals = [(turn.stime, turn.etime) for turn in transcript.turns if turn.speaker == SpeakerRole.CUSTOMER]
        all_intervals = agent_intervals + customer_intervals

        agent_talk = _sum_intervals(agent_intervals)
        customer_talk = _sum_intervals(customer_intervals)
        overtalk = _intersection_duration(agent_intervals, customer_intervals)
        combined_talk = _sum_intervals(all_intervals)
        silence = max(total_duration - combined_talk, 0.0)

        special_case = ""
        if "VOICEMAIL" in transcript.special_tags:
            special_case = "VOICEMAIL"
        elif not agent_intervals or not customer_intervals:
            special_case = "SINGLE_SPEAKER"
        elif overtalk == 0:
            special_case = "ZERO_OVERTALK"

        return MetricResult(
            call_id=transcript.call_id,
            total_duration=round(total_duration, 4),
            agent_talk_time=round(agent_talk, 4),
            customer_talk_time=round(customer_talk, 4),
            agent_talk_pct=round((agent_talk / total_duration) * 100, 2),
            customer_talk_pct=round((customer_talk / total_duration) * 100, 2),
            silence_pct=round((silence / total_duration) * 100, 2),
            overtalk_pct=round((overtalk / total_duration) * 100, 2),
            special_case=special_case,
        )
