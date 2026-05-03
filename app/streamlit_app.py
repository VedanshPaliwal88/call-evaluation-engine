from __future__ import annotations

from typing import Any

try:  # pragma: no cover - UI imports depend on runtime
    import pandas as pd
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Streamlit and pandas must be installed to run the UI.") from exc

from call_evaluation.config import get_settings
from call_evaluation.models.analysis import AnalysisApproach, BatchProcessingReport
from call_evaluation.services.analysis_service import (
    AnalysisService,
    ComplianceBatchResult,
    ComplianceRowDetail,
    MetricsBatchResult,
    ProfanityBatchResult,
)
from call_evaluation.services.exceptions import LLMUnavailableError
from call_evaluation.visualization import (
    create_distribution_histograms,
    create_metrics_box_plot,
    create_metrics_scatter_plot,
    create_top_n_figure,
)


APPROACH_OPTIONS = {
    "Pattern Matching": AnalysisApproach.REGEX,
    "LLM": AnalysisApproach.LLM,
}
ENTITY_OPTIONS = (
    "Profanity Detection",
    "Privacy and Compliance Violation",
)
STATE_KEY = "analysis_session"


def _init_page() -> None:
    st.set_page_config(page_title="Call Evaluation Project", layout="wide")
    st.title("Call Evaluation Project")
    st.caption("Batch transcript analysis for profanity, compliance, and call quality metrics.")


def _get_session_state() -> dict[str, Any]:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = {
            "transcripts": None,
            "entity_label": None,
            "approach_label": None,
            "selected_result": None,
            "metrics_result": None,
            "chart_selected_call_id": None,
        }
    return st.session_state[STATE_KEY]


def _render_sidebar(service: AnalysisService) -> tuple[str, str]:
    settings = get_settings()
    llm_state = service.get_llm_runtime_state()

    with st.sidebar:
        st.header("Configuration")
        st.write(f"Configured model: `{settings.llm_model}`")
        if llm_state.api_available:
            st.success("LLM features are available.")
            approach_labels = list(APPROACH_OPTIONS.keys())
        else:
            st.warning(llm_state.message or "LLM features are unavailable.")
            st.caption("LLM is disabled until `OPENAI_API_KEY` is configured in the project `.env` file.")
            approach_labels = ["Pattern Matching"]

        entity_label = st.selectbox("Entity", ENTITY_OPTIONS)
        approach_label = st.selectbox("Approach", approach_labels)

    return entity_label, approach_label


def _render_uploads() -> list[Any]:
    st.subheader("Upload Batch")
    st.write("Upload multiple JSON/YAML files or a single ZIP archive containing transcript files.")
    return st.file_uploader(
        "Transcript inputs",
        type=["json", "yaml", "yml", "zip"],
        accept_multiple_files=True,
        help="You can mix JSON and YAML files, or upload one ZIP archive.",
        key="transcript-uploads",
    )


def _render_report_summary(report: BatchProcessingReport) -> None:
    summary_columns = st.columns(3)
    summary_columns[0].metric("Processed", report.processed_calls)
    summary_columns[1].metric("Successful", report.successful_calls)
    summary_columns[2].metric("Failed", report.failed_calls)


def _render_errors(report: BatchProcessingReport) -> None:
    if not report.errors:
        return
    st.subheader("File Issues")
    for call_id, errors in report.errors.items():
        st.error(f"{call_id}: {' | '.join(errors)}")


def _render_profanity_results(result: ProfanityBatchResult) -> None:
    st.subheader("Q1 Results")
    _render_report_summary(result.report)
    if not result.rows:
        st.info("No valid calls were available for profanity analysis.")
        _render_errors(result.report)
        return

    dataframe = pd.DataFrame(result.rows)
    event = st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="profanity-table",
    )
    _render_errors(result.report)

    selected_index = _selected_row_index(event)
    chart_cid = _get_session_state().get("chart_selected_call_id")

    if selected_index is not None:
        # Row click takes priority
        selected_call_id = str(dataframe.iloc[selected_index]["call_id"])
        detail = result.details[selected_call_id]
        st.markdown(f"**Evidence for {selected_call_id}**")
        st.write(f"Source file: `{detail.source_name}`")
        st.write(f"Special tags: `{', '.join(detail.special_tags)}`")
        _render_profanity_detail("Agent", detail.agent)
        _render_profanity_detail("Customer", detail.customer)
    elif chart_cid:
        if chart_cid in result.details:
            st.info(f"Chart selection — showing details for `{chart_cid}`")
            detail = result.details[chart_cid]
            st.write(f"Source file: `{detail.source_name}`")
            st.write(f"Special tags: `{', '.join(detail.special_tags)}`")
            _render_profanity_detail("Agent", detail.agent)
            _render_profanity_detail("Customer", detail.customer)
        else:
            st.caption(f"Chart-selected call `{chart_cid}` is not in the current profanity results.")
    else:
        st.caption("Click a row to inspect evidence details.")


def _render_profanity_detail(label: str, result: Any) -> None:
    st.markdown(
        f"**{label}**  \n"
        f"Flag: `{result.flag}`  \n"
        f"Severity: `{result.severity.value}`  \n"
        f"Context: `{result.context.value}`  \n"
        f"Sentiment: `{result.sentiment.value}`"
    )
    if result.evidence:
        evidence_rows = [item.model_dump() for item in result.evidence]
        st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)
    else:
        st.caption(f"No {label.lower()} evidence spans were flagged.")


def _render_compliance_results(result: ComplianceBatchResult) -> None:
    st.subheader("Q2 Results")
    _render_report_summary(result.report)
    if not result.rows:
        st.info("No valid calls were available for compliance analysis.")
        _render_errors(result.report)
        return

    dataframe = pd.DataFrame(result.rows)
    event = st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="compliance-table",
    )
    _render_errors(result.report)

    selected_index = _selected_row_index(event)
    chart_cid = _get_session_state().get("chart_selected_call_id")

    if selected_index is not None:
        selected_call_id = str(dataframe.iloc[selected_index]["call_id"])
        detail = result.details[selected_call_id]
        st.markdown(f"**Evidence for {selected_call_id}**")
        _render_compliance_detail(detail)
    elif chart_cid:
        if chart_cid in result.details:
            st.info(f"Chart selection — showing details for `{chart_cid}`")
            _render_compliance_detail(result.details[chart_cid])
        else:
            st.caption(f"Chart-selected call `{chart_cid}` is not in the current compliance results.")
    else:
        st.caption("Click a row to inspect evidence details.")


def _render_compliance_detail(detail: ComplianceRowDetail) -> None:
    st.write(f"Source file: `{detail.source_name}`")
    st.write(f"Special tags: `{', '.join(detail.special_tags)}`")
    st.markdown(
        f"Violation: `{detail.result.violation.value}`  \n"
        f"Verification status: `{detail.result.verification_status.value}`  \n"
        f"Violation type: `{detail.result.violation_type}`  \n"
        f"Notes: {detail.result.notes or 'None'}"
    )
    if detail.result.evidence:
        evidence_rows = [item.model_dump() for item in detail.result.evidence]
        st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No evidence spans were returned for this call.")


def _render_metrics_section(
    result: MetricsBatchResult,
    selected_result: Any = None,
    entity_label: str = "",
) -> None:
    st.subheader("Q3 Metrics")
    _render_report_summary(result.report)
    if not result.rows:
        st.info("No valid calls were available for metrics.")
        _render_errors(result.report)
        return

    metrics_frame = pd.DataFrame(result.rows)
    st.dataframe(metrics_frame, use_container_width=True, hide_index=True)
    _render_errors(result.report)

    # Derive enrichment from whichever detector was run
    profanity_rows: list[Any] | None = None
    compliance_rows: list[Any] | None = None
    if isinstance(selected_result, ProfanityBatchResult):
        profanity_rows = selected_result.rows
    elif isinstance(selected_result, ComplianceBatchResult):
        compliance_rows = selected_result.rows

    # Correlation summary for top-10 overtalk calls
    if profanity_rows is not None or compliance_rows is not None:
        top_ids = {
            r["call_id"]
            for r in sorted(result.rows, key=lambda r: float(r.get("overtalk_pct", 0.0)), reverse=True)[:10]
        }
        summary_parts: list[str] = []
        if profanity_rows is not None:
            prof_map = {r["call_id"]: bool(r.get("agent_profanity")) or bool(r.get("customer_profanity")) for r in profanity_rows}
            p_count = sum(1 for cid in top_ids if prof_map.get(cid, False))
            summary_parts.append(f"**{p_count}** of the top 10 overtalk calls also had flagged profanity.")
        if compliance_rows is not None:
            comp_map = {r["call_id"]: r.get("violation", "NO") == "YES" for r in compliance_rows}
            c_count = sum(1 for cid in top_ids if comp_map.get(cid, False))
            summary_parts.append(f"**{c_count}** had compliance violations.")
        if summary_parts:
            st.info("  ".join(summary_parts))

    box_plot = create_metrics_box_plot(result.rows)
    scatter_plot = create_metrics_scatter_plot(
        result.rows,
        profanity_rows=profanity_rows,
        compliance_rows=compliance_rows,
    )
    histograms = create_distribution_histograms(result.rows)
    top_silence = create_top_n_figure(
        result.rows, "silence_pct", top_n=10,
        profanity_rows=profanity_rows, compliance_rows=compliance_rows,
    )
    top_overtalk = create_top_n_figure(
        result.rows, "overtalk_pct", top_n=10,
        profanity_rows=profanity_rows, compliance_rows=compliance_rows,
    )

    visual_columns = st.columns(2)
    if box_plot is not None:
        visual_columns[0].plotly_chart(box_plot, use_container_width=True)
    if scatter_plot is not None:
        visual_columns[1].plotly_chart(scatter_plot, use_container_width=True)

    histogram_columns = st.columns(2)
    if histograms.get("silence_pct") is not None:
        histogram_columns[0].plotly_chart(histograms["silence_pct"], use_container_width=True)
    if histograms.get("overtalk_pct") is not None:
        histogram_columns[1].plotly_chart(histograms["overtalk_pct"], use_container_width=True)

    # Top-N bar charts with click-to-navigate support
    top_columns = st.columns(2)
    silence_event = None
    overtalk_event = None
    if top_silence is not None:
        silence_event = top_columns[0].plotly_chart(
            top_silence, use_container_width=True, on_select="rerun", key="top-silence-chart"
        )
    if top_overtalk is not None:
        overtalk_event = top_columns[1].plotly_chart(
            top_overtalk, use_container_width=True, on_select="rerun", key="top-overtalk-chart"
        )

    # Detect bar click and store in session for Q1/Q2 evidence navigation
    selected_cid: str | None = None
    for evt in (overtalk_event, silence_event):
        if not evt:
            continue
        pts = getattr(getattr(evt, "selection", None), "points", None) or []
        if pts:
            selected_cid = str(pts[0].get("x", "")) or None
            break

    session = _get_session_state()
    if selected_cid and selected_cid != session.get("chart_selected_call_id"):
        session["chart_selected_call_id"] = selected_cid
        st.rerun()


def _selected_row_index(event: Any) -> int | None:
    if not event:
        return None
    selection = getattr(event, "selection", None)
    if isinstance(selection, dict):
        rows = selection.get("rows", [])
        return rows[0] if rows else None
    rows = getattr(selection, "rows", None)
    if rows:
        return rows[0]
    return None


def _run_analysis(
    service: AnalysisService,
    uploads: list[Any],
    entity_label: str,
    approach_label: str,
) -> None:
    if not uploads:
        st.info("Upload at least one transcript file or ZIP archive to continue.")
        return

    payloads = [(upload.name, upload.getvalue()) for upload in uploads]
    transcripts = service.load_inputs(payloads)
    metrics_result = service.analyze_metrics(transcripts)

    selected_approach = APPROACH_OPTIONS[approach_label]
    if entity_label == "Profanity Detection":
        selected_result = service.analyze_profanity(transcripts, selected_approach)
    else:
        selected_result = service.analyze_compliance(transcripts, selected_approach)

    session = _get_session_state()
    session["transcripts"] = transcripts
    session["entity_label"] = entity_label
    session["approach_label"] = approach_label
    session["selected_result"] = selected_result
    session["metrics_result"] = metrics_result
    session["chart_selected_call_id"] = None


def _render_saved_results(entity_label: str, approach_label: str) -> None:
    session = _get_session_state()
    selected_result = session.get("selected_result")
    metrics_result = session.get("metrics_result")

    if selected_result is None or metrics_result is None:
        return

    # Clear button — lets user start fresh without a browser refresh
    if st.button("Clear and Upload New Batch", type="secondary"):
        st.session_state.pop(STATE_KEY, None)
        st.session_state.pop("transcript-uploads", None)
        st.rerun()

    if session.get("entity_label") != entity_label or session.get("approach_label") != approach_label:
        st.info("Current selectors differ from the last batch run. Click `Run Batch Analysis` to refresh results.")
        return

    if entity_label == "Profanity Detection":
        _render_profanity_results(selected_result)
    else:
        _render_compliance_results(selected_result)

    st.divider()
    _render_metrics_section(
        metrics_result,
        selected_result=selected_result,
        entity_label=entity_label,
    )


def main() -> None:
    _init_page()
    service = AnalysisService()
    _get_session_state()
    entity_label, approach_label = _render_sidebar(service)
    uploads = _render_uploads()

    if st.button("Run Batch Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Analysing transcripts..."):
                _run_analysis(service, uploads, entity_label, approach_label)
        except LLMUnavailableError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

    _render_saved_results(entity_label, approach_label)


if __name__ == "__main__":
    main()
