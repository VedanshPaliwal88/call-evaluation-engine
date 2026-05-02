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
    ProfanityBatchResult,
    ProfanityRowDetail,
)
from call_evaluation.services.exceptions import LLMUnavailableError
from call_evaluation.visualization import (
    create_distribution_histograms,
    create_per_call_metrics_figure,
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


def _init_page() -> None:
    st.set_page_config(page_title="Call Evaluation Project", layout="wide")
    st.title("Call Evaluation Project")
    st.caption("Batch transcript analysis for profanity, compliance, and call quality metrics.")


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
    if selected_index is None:
        st.caption("Click a row to inspect evidence details.")
        return

    selected_call_id = str(dataframe.iloc[selected_index]["call_id"])
    detail = result.details[selected_call_id]
    with st.expander(f"Evidence for {selected_call_id}", expanded=True):
        st.write(f"Source file: `{detail.source_name}`")
        st.write(f"Special tags: `{', '.join(detail.special_tags)}`")
        _render_profanity_detail("Agent", detail.agent)
        _render_profanity_detail("Customer", detail.customer)


def _render_profanity_detail(label: str, result) -> None:
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
    if selected_index is None:
        st.caption("Click a row to inspect evidence details.")
        return

    selected_call_id = str(dataframe.iloc[selected_index]["call_id"])
    detail = result.details[selected_call_id]
    with st.expander(f"Evidence for {selected_call_id}", expanded=True):
        _render_compliance_detail(detail)


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


def _render_metrics_section(rows: list[dict], report: BatchProcessingReport) -> None:
    st.subheader("Q3 Metrics")
    _render_report_summary(report)
    if not rows:
        st.info("No valid calls were available for metrics.")
        _render_errors(report)
        return

    metrics_frame = pd.DataFrame(rows)
    st.dataframe(metrics_frame, use_container_width=True, hide_index=True)
    _render_errors(report)

    per_call_figure = create_per_call_metrics_figure(rows)
    histograms = create_distribution_histograms(rows)
    top_silence = create_top_n_figure(rows, "silence_pct", top_n=10)
    top_overtalk = create_top_n_figure(rows, "overtalk_pct", top_n=10)

    if per_call_figure is not None:
        st.plotly_chart(per_call_figure, use_container_width=True)

    histogram_columns = st.columns(2)
    if histograms.get("silence_pct") is not None:
        histogram_columns[0].plotly_chart(histograms["silence_pct"], use_container_width=True)
    if histograms.get("overtalk_pct") is not None:
        histogram_columns[1].plotly_chart(histograms["overtalk_pct"], use_container_width=True)

    top_columns = st.columns(2)
    if top_silence is not None:
        top_columns[0].plotly_chart(top_silence, use_container_width=True)
    if top_overtalk is not None:
        top_columns[1].plotly_chart(top_overtalk, use_container_width=True)


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


def main() -> None:
    _init_page()
    service = AnalysisService()
    entity_label, approach_label = _render_sidebar(service)
    uploads = _render_uploads()

    if not st.button("Run Batch Analysis", type="primary", use_container_width=True):
        return

    if not uploads:
        st.info("Upload at least one transcript file or ZIP archive to continue.")
        return

    try:
        payloads = [(upload.name, upload.getvalue()) for upload in uploads]
        transcripts = service.load_inputs(payloads)
        metrics_result = service.analyze_metrics(transcripts)

        selected_approach = APPROACH_OPTIONS[approach_label]
        if entity_label == "Profanity Detection":
            selected_result = service.analyze_profanity(transcripts, selected_approach)
            _render_profanity_results(selected_result)
        else:
            selected_result = service.analyze_compliance(transcripts, selected_approach)
            _render_compliance_results(selected_result)

        st.divider()
        _render_metrics_section(metrics_result.rows, metrics_result.report)
    except LLMUnavailableError as exc:
        st.error(str(exc))
    except Exception:
        st.error("The uploaded files could not be processed. Check file formats and transcript contents, then try again.")


if __name__ == "__main__":
    main()
