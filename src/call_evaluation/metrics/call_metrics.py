"""Call quality metrics: silence, overtalk, and talk-time percentages from timestamps."""
from __future__ import annotations

from call_evaluation.models.analysis import MetricResult
from call_evaluation.models.transcript import SpeakerRole, TranscriptFilePayload


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping or adjacent time intervals into a minimal sorted list.

    Args:
        intervals: Unsorted list of (start, end) float pairs.

    Returns:
        Sorted list of non-overlapping (start, end) pairs.
    """
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping or touching: extend the last interval's end if needed
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _sum_intervals(intervals: list[tuple[float, float]]) -> float:
    """Return the total duration covered by a list of possibly-overlapping intervals.

    Args:
        intervals: List of (start, end) float pairs.

    Returns:
        Sum of merged interval lengths in seconds.
    """
    return sum(end - start for start, end in _merge_intervals(intervals))


def _intersection_duration(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    """Return the total overlapping duration between two sets of time intervals.

    Used to compute overtalk — the seconds where agent and customer spoke
    simultaneously. Both lists are merged before comparison so duplicate spans
    are not double-counted.

    Args:
        left: Agent speaking intervals (start, end).
        right: Customer speaking intervals (start, end).

    Returns:
        Total seconds of simultaneous speech.
    """
    left = _merge_intervals(left)
    right = _merge_intervals(right)
    i = j = 0
    total = 0.0
    # Two-pointer sweep: advance the pointer whose interval ends first.
    # This is O(n+m) and avoids the O(n*m) naive pairwise comparison.
    while i < len(left) and j < len(right):
        # Overlap region of the two current intervals
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            total += end - start
        # Advance whichever interval ends first; the other may still overlap the next
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


class MetricsService:
    """Compute silence, overtalk, and talk-time metrics from transcript timestamps."""

    def analyze(self, transcript: TranscriptFilePayload) -> MetricResult:
        """Compute timing metrics for a single transcript.

        Args:
            transcript: Validated payload with normalized turns and timestamps.

        Returns:
            MetricResult with silence_pct, overtalk_pct, talk-time percentages,
            and a special_case label for degenerate inputs (empty, voicemail, etc.).
        """
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
