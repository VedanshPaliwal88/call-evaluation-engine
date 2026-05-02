from __future__ import annotations

from typing import Any


def _load_plotly_modules() -> tuple[Any, Any] | tuple[None, None]:
    try:  # pragma: no cover - visualization import depends on runtime
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        return None, None
    return px, go


def create_metrics_box_plot(rows: list[dict[str, Any]]) -> Any:
    px, _ = _load_plotly_modules()
    if px is None or not rows:
        return None
    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        plot_rows.append({"metric": "Silence %", "value": row.get("silence_pct", 0.0), "call_id": row.get("call_id", "")})
        plot_rows.append({"metric": "Overtalk %", "value": row.get("overtalk_pct", 0.0), "call_id": row.get("call_id", "")})
    return px.box(
        plot_rows,
        x="metric",
        y="value",
        color="metric",
        points="outliers",
        hover_data=["call_id"],
        title="Silence and Overtalk Distribution",
    )


def create_metrics_scatter_plot(rows: list[dict[str, Any]]) -> Any:
    px, _ = _load_plotly_modules()
    if px is None or not rows:
        return None
    hover_data = {
        "call_id": False,
        "agent_talk_pct": any("agent_talk_pct" in row for row in rows),
        "customer_talk_pct": any("customer_talk_pct" in row for row in rows),
    }
    if any("special_case" in row for row in rows):
        hover_data["special_case"] = True
    return px.scatter(
        rows,
        x="silence_pct",
        y="overtalk_pct",
        hover_name="call_id",
        hover_data=hover_data,
        title="Silence vs Overtalk by Call",
        labels={"silence_pct": "Silence %", "overtalk_pct": "Overtalk %"},
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
    return create_metrics_box_plot(rows)
