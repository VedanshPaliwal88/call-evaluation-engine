"""Plotly chart builders for Q3 metrics with optional profanity/compliance overlays."""
from __future__ import annotations

from typing import Any


_SEVERITY_COLORS: dict[str, str] = {
    "NONE":     "#888888",
    "MILD":     "#FFD700",
    "MODERATE": "#FF8C00",
    "SEVERE":   "#DC143C",
}
_SEVERITY_ORDER = ["NONE", "MILD", "MODERATE", "SEVERE"]

_VIOLATION_COLORS: dict[str, str] = {
    "NO":  "#888888",
    "YES": "#DC143C",
}
_VIOLATION_ORDER = ["NO", "YES"]


def _load_plotly_modules() -> tuple[Any, Any] | tuple[None, None]:
    try:  # pragma: no cover - visualization import depends on runtime
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        return None, None
    return px, go


def create_metrics_box_plot(rows: list[dict[str, Any]]) -> Any:
    """Create a box plot comparing silence and overtalk percentage distributions.

    Args:
        rows: List of metric row dicts containing silence_pct and overtalk_pct.

    Returns:
        Plotly figure, or None if Plotly is unavailable or rows is empty.
    """
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


def create_metrics_scatter_plot(
    rows: list[dict[str, Any]],
    profanity_rows: list[dict[str, Any]] | None = None,
    compliance_rows: list[dict[str, Any]] | None = None,
) -> Any:
    """Create a scatter plot of silence vs overtalk, optionally colored by profanity or compliance.

    Profanity severity coloring takes priority: if profanity_rows is provided the
    points are colored by severity tier (grey/yellow/orange/red). If only
    compliance_rows is provided the points are colored by violation (grey/red).
    With neither, a plain single-color scatter is returned.

    Args:
        rows: Metric row dicts containing silence_pct, overtalk_pct, and call_id.
        profanity_rows: Optional profanity summary rows; when present, drives severity coloring.
        compliance_rows: Optional compliance summary rows; used only when profanity_rows is None.

    Returns:
        Plotly figure, or None if Plotly is unavailable or rows is empty.
    """
    px, _ = _load_plotly_modules()
    if px is None or not rows:
        return None

    hover_data: dict[str, Any] = {
        "call_id": False,
        "agent_talk_pct": any("agent_talk_pct" in row for row in rows),
        "customer_talk_pct": any("customer_talk_pct" in row for row in rows),
    }
    if any("special_case" in row for row in rows):
        hover_data["special_case"] = True

    if profanity_rows is not None:
        sev_map = {r["call_id"]: r.get("severity", "NONE") for r in profanity_rows}
        enriched = [{**r, "severity": sev_map.get(r["call_id"], "NONE")} for r in rows]
        hover_data["severity"] = True
        return px.scatter(
            enriched,
            x="silence_pct",
            y="overtalk_pct",
            color="severity",
            color_discrete_map=_SEVERITY_COLORS,
            category_orders={"severity": _SEVERITY_ORDER},
            hover_name="call_id",
            hover_data=hover_data,
            title="Silence vs Overtalk by Call (coloured by profanity severity)",
            labels={"silence_pct": "Silence %", "overtalk_pct": "Overtalk %"},
        )

    if compliance_rows is not None:
        viol_map = {r["call_id"]: r.get("violation", "NO") for r in compliance_rows}
        enriched = [{**r, "violation": viol_map.get(r["call_id"], "NO")} for r in rows]
        hover_data["violation"] = True
        return px.scatter(
            enriched,
            x="silence_pct",
            y="overtalk_pct",
            color="violation",
            color_discrete_map=_VIOLATION_COLORS,
            category_orders={"violation": _VIOLATION_ORDER},
            hover_name="call_id",
            hover_data=hover_data,
            title="Silence vs Overtalk by Call (coloured by compliance violation)",
            labels={"silence_pct": "Silence %", "overtalk_pct": "Overtalk %"},
        )

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
    """Create histograms for silence and overtalk percentage distributions.

    Args:
        rows: List of metric row dicts containing silence_pct and overtalk_pct.

    Returns:
        Dict with keys "silence_pct" and "overtalk_pct" mapping to Plotly figures,
        or an empty dict if Plotly is unavailable or rows is empty.
    """
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


def create_top_n_figure(
    rows: list[dict[str, Any]],
    metric_key: str,
    top_n: int = 10,
    profanity_rows: list[dict[str, Any]] | None = None,
    compliance_rows: list[dict[str, Any]] | None = None,
) -> Any:
    """Create a bar chart of the top N calls by a given metric, with P/C badge overlays.

    Bars representing calls with profanity are annotated with a red "P" badge,
    compliance violations with an orange "C", and both with a dark-red "P C".

    Args:
        rows: Metric row dicts containing call_id and the target metric field.
        metric_key: Column name to rank and plot (e.g. "silence_pct").
        top_n: Maximum number of calls to show, ranked descending by metric_key.
        profanity_rows: Optional profanity summary rows for badge annotation.
        compliance_rows: Optional compliance summary rows for badge annotation.

    Returns:
        Plotly figure with optional badge annotations, or None if unavailable.
    """
    px, _ = _load_plotly_modules()
    if px is None or not rows:
        return None

    ranked_rows = sorted(rows, key=lambda row: float(row.get(metric_key, 0.0)), reverse=True)[: max(top_n, 1)]
    fig = px.bar(
        ranked_rows,
        x="call_id",
        y=metric_key,
        title=f"Top {len(ranked_rows)} Calls by {metric_key.replace('_', ' ').title()}",
    )

    if profanity_rows is not None or compliance_rows is not None:
        prof_map: dict[str, bool] = {}
        if profanity_rows:
            for r in profanity_rows:
                prof_map[r["call_id"]] = bool(r.get("agent_profanity")) or bool(r.get("customer_profanity"))

        comp_map: dict[str, bool] = {}
        if compliance_rows:
            for r in compliance_rows:
                comp_map[r["call_id"]] = r.get("violation", "NO") == "YES"

        max_val = max((float(r.get(metric_key, 0.0)) for r in ranked_rows), default=1.0)
        offset = max_val * 0.04

        for row in ranked_rows:
            cid = row["call_id"]
            bar_top = float(row.get(metric_key, 0.0))
            has_p = prof_map.get(cid, False)
            has_c = comp_map.get(cid, False)

            if not has_p and not has_c:
                continue

            if has_p and has_c:
                label, color = "P C", "#8B0000"
            elif has_p:
                label, color = "P", "#DC143C"
            else:
                label, color = "C", "#FF8C00"

            fig.add_annotation(
                x=cid,
                y=bar_top + offset,
                text=label,
                showarrow=False,
                font=dict(size=10, color=color, family="monospace"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
            )

    return fig
