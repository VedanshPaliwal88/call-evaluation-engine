from __future__ import annotations

from typing import Any


def _load_plotly_modules() -> tuple[Any, Any] | tuple[None, None]:
    try:  # pragma: no cover - visualization import depends on runtime
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        return None, None
    return px, go


def create_per_call_metrics_figure(rows: list[dict[str, Any]]) -> Any:
    px, _ = _load_plotly_modules()
    if px is None or not rows:
        return None
    return px.bar(
        rows,
        x="call_id",
        y=["silence_pct", "overtalk_pct", "agent_talk_pct", "customer_talk_pct"],
        barmode="group",
        title="Per-Call Conversation Metrics",
    )


def create_distribution_histograms(rows: list[dict[str, Any]]) -> dict[str, Any]:
    px, _ = _load_plotly_modules()
    if px is None or not rows:
        return {}
    return {
        "silence_pct": px.histogram(
            rows,
            x="silence_pct",
            nbins=min(max(len(rows), 5), 20),
            title="Silence Percentage Distribution",
        ),
        "overtalk_pct": px.histogram(
            rows,
            x="overtalk_pct",
            nbins=min(max(len(rows), 5), 20),
            title="Overtalk Percentage Distribution",
        ),
    }


def create_top_n_figure(rows: list[dict[str, Any]], metric_key: str, top_n: int = 10) -> Any:
    px, _ = _load_plotly_modules()
    if px is None or not rows:
        return None
    ranked_rows = sorted(rows, key=lambda row: float(row.get(metric_key, 0.0)), reverse=True)[: max(top_n, 1)]
    return px.bar(
        ranked_rows,
        x="call_id",
        y=metric_key,
        title=f"Top {len(ranked_rows)} Calls by {metric_key.replace('_', ' ').title()}",
    )


def create_metrics_figure(rows: list[dict[str, Any]]) -> Any:
    return create_per_call_metrics_figure(rows)
    if not rows:
        return None
    return px.bar(
        rows,
        x="call_id",
        y=["silence_pct", "overtalk_pct"],
        barmode="group",
        title="Silence and Overtalk Percentage by Call",
    )
