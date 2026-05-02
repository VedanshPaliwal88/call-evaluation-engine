"""Standalone annotation tool for labeling profanity and compliance in call transcripts.

Run with:
    streamlit run tools/annotator.py
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the tool works from any cwd
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CONVERSATIONS_DIR = ROOT / "All_Conversations"
PROFANITY_CSV = ROOT / "data" / "labeled" / "annotations_profanity.csv"
COMPLIANCE_CSV = ROOT / "data" / "labeled" / "annotations_compliance.csv"
SKIPPED_CSV = ROOT / "data" / "labeled" / "annotation_skips.csv"

# ---------------------------------------------------------------------------
# CSV column schemas — must stay in sync with the existing CSV files
# ---------------------------------------------------------------------------

PROFANITY_FIELDS = ["call_id", "agent_profanity", "customer_profanity", "severity", "context", "notes"]
COMPLIANCE_FIELDS = ["call_id", "violation", "verification_status", "violation_type", "notes"]
SKIP_FIELDS = ["call_id"]

# ---------------------------------------------------------------------------
# Enum options — must match Pydantic models in analysis.py
# ---------------------------------------------------------------------------

SEVERITY_OPTIONS = ["NONE", "MILD", "MODERATE", "SEVERE"]
CONTEXT_OPTIONS = ["DIRECTED_AT_AGENT", "DIRECTED_AT_CUSTOMER", "SELF_EXPRESSION", "NARRATIVE_QUOTE", "AMBIENT"]
VERIFICATION_OPTIONS = ["VERIFIED", "PARTIAL", "UNVERIFIED", "NOT_APPLICABLE"]
# NOTE: NO_VIOLATION is included here for human annotators but is not a valid value in the
# compliance prompt or in _ENUM_SAFE_DEFAULTS. If an annotation uses it, the LLM sanitizer
# will fall back to NOT_APPLICABLE when that annotation is used for evaluation comparisons.
VIOLATION_TYPE_OPTIONS = ["NO_VIOLATION", "NO_VERIFICATION", "ACCOUNT_DETAILS_BEFORE_VERIFICATION", "NOT_APPLICABLE"]

# ---------------------------------------------------------------------------
# Low-level CSV helpers
# ---------------------------------------------------------------------------


def _read_csv_ids(path: Path) -> set[str]:
    """Return the set of call_id values in a CSV, empty set if file missing."""
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["call_id"] for row in csv.DictReader(fh) if row.get("call_id")}


def _read_csv_row(path: Path, call_id: str) -> dict[str, str] | None:
    """Return the first row matching call_id, or None."""
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("call_id") == call_id:
                return dict(row)
    return None


def _append_csv_row(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    """Append a row, writing the header first if the file does not yet exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------


def _load_transcript(call_id: str) -> list[dict[str, Any]]:
    path = CONVERSATIONS_DIR / call_id
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Index: which files exist and what has been labeled
# ---------------------------------------------------------------------------


def _all_conversation_files() -> list[str]:
    """Sorted list of regular (non-checkpoint) JSON filenames in All_Conversations/."""
    return sorted(
        p.name
        for p in CONVERSATIONS_DIR.iterdir()
        if p.suffix == ".json" and "checkpoint" not in p.name
    )


def _build_state() -> dict[str, Any]:
    """Rebuild the mutable annotation index from disk. Called fresh each rerun."""
    all_files = _all_conversation_files()
    profanity_ids = _read_csv_ids(PROFANITY_CSV)
    compliance_ids = _read_csv_ids(COMPLIANCE_CSV)
    skipped_ids = _read_csv_ids(SKIPPED_CSV)

    fully_annotated = {f for f in all_files if f in profanity_ids and f in compliance_ids}
    profanity_only = {f for f in all_files if f in profanity_ids and f not in compliance_ids}
    compliance_only = {f for f in all_files if f not in profanity_ids and f in compliance_ids}
    unannotated = {f for f in all_files if f not in profanity_ids and f not in compliance_ids}

    pending = [
        f for f in all_files
        if f not in fully_annotated and f not in skipped_ids
    ]

    return {
        "all_files": all_files,
        "profanity_ids": profanity_ids,
        "compliance_ids": compliance_ids,
        "skipped_ids": skipped_ids,
        "fully_annotated": fully_annotated,
        "profanity_only": profanity_only,
        "compliance_only": compliance_only,
        "unannotated": unannotated,
        "pending": pending,
    }


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

_SESSION_KEY = "annotator"


def _session() -> dict[str, Any]:
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = {
            "current": None,
            "show_form": False,
            "flash": "",
        }
    return st.session_state[_SESSION_KEY]


def _pick_next(pending: list[str]) -> str | None:
    return random.choice(pending) if pending else None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar(index: dict[str, Any]) -> None:
    with st.sidebar:
        st.header("Stats")
        total = len(index["all_files"])
        done = len(index["fully_annotated"])
        st.metric("Total files", total)
        st.metric("Fully annotated", f"{done} / {total}")
        st.progress(done / total if total else 0)
        st.divider()
        st.write(f"Profanity only: **{len(index['profanity_only'])}**")
        st.write(f"Compliance only: **{len(index['compliance_only'])}**")
        st.write(f"Unannotated: **{len(index['unannotated'])}**")
        st.write(f"Skipped (persistent): **{len(index['skipped_ids'])}**")
        st.write(f"Pending (remaining): **{len(index['pending'])}**")


# ---------------------------------------------------------------------------
# Transcript display
# ---------------------------------------------------------------------------


def _render_transcript(call_id: str) -> None:
    turns = _load_transcript(call_id)
    if not turns:
        st.warning("Could not load transcript for this file.")
        return

    st.markdown("#### Transcript")
    for turn in turns:
        speaker = str(turn.get("speaker", "Unknown"))
        stime = float(turn.get("stime", 0))
        etime = float(turn.get("etime", 0))
        text = str(turn.get("text", "")).strip()
        label_color = "#1f77b4" if speaker.lower() == "agent" else "#ff7f0e"
        st.markdown(
            f'<span style="color:{label_color}; font-weight:bold">{speaker}</span> '
            f'<span style="color:gray; font-size:0.85em">[{stime:.1f}s – {etime:.1f}s]</span><br>'
            f'{text}',
            unsafe_allow_html=True,
        )
        st.write("")


# ---------------------------------------------------------------------------
# Read-only existing annotation display
# ---------------------------------------------------------------------------


def _render_readonly_profanity(row: dict[str, str]) -> None:
    col1, col2 = st.columns(2)
    col1.markdown(f"Agent profanity: `{row.get('agent_profanity', '—')}`")
    col1.markdown(f"Severity: `{row.get('severity', '—')}`")
    col2.markdown(f"Customer profanity: `{row.get('customer_profanity', '—')}`")
    col2.markdown(f"Context: `{row.get('context', '—')}`")
    if row.get("notes"):
        st.caption(f"Notes: {row['notes']}")


def _render_readonly_compliance(row: dict[str, str]) -> None:
    col1, col2 = st.columns(2)
    col1.markdown(f"Violation: `{row.get('violation', '—')}`")
    col1.markdown(f"Verification status: `{row.get('verification_status', '—')}`")
    col2.markdown(f"Violation type: `{row.get('violation_type', '—')}`")
    if row.get("notes"):
        st.caption(f"Notes: {row['notes']}")


# ---------------------------------------------------------------------------
# Annotation form
# ---------------------------------------------------------------------------


def _render_annotation_form(call_id: str, needs_profanity: bool, needs_compliance: bool, index: dict[str, Any]) -> None:
    st.markdown("### Annotation")

    # Show existing labels as read-only context above the form
    if not needs_profanity:
        existing = _read_csv_row(PROFANITY_CSV, call_id)
        if existing:
            with st.expander("Profanity *(already labeled — read only)*", expanded=False):
                _render_readonly_profanity(existing)

    if not needs_compliance:
        existing = _read_csv_row(COMPLIANCE_CSV, call_id)
        if existing:
            with st.expander("Compliance *(already labeled — read only)*", expanded=False):
                _render_readonly_compliance(existing)

    with st.form(key=f"form-{call_id}"):
        profanity_values: dict[str, str] | None = None
        compliance_values: dict[str, str] | None = None

        # Lay out profanity and compliance side by side when both are needed
        if needs_profanity and needs_compliance:
            left, right = st.columns(2)
        else:
            left = right = None

        # Profanity section
        if needs_profanity:
            container = left if left is not None else st.container()
            with container:
                with st.expander("**Profanity**", expanded=True):
                    agent = st.radio(
                        "Agent profanity", ["no", "yes"], horizontal=True, key=f"{call_id}-agent"
                    )
                    customer = st.radio(
                        "Customer profanity", ["no", "yes"], horizontal=True, key=f"{call_id}-customer"
                    )
                    severity = st.selectbox("Severity", SEVERITY_OPTIONS, key=f"{call_id}-severity")
                    context = st.selectbox("Context", CONTEXT_OPTIONS, key=f"{call_id}-context")
                    p_notes = st.text_input("Notes", key=f"{call_id}-p-notes", placeholder="Optional")
                    profanity_values = {
                        "call_id": call_id,
                        "agent_profanity": agent,
                        "customer_profanity": customer,
                        "severity": severity,
                        "context": context,
                        "notes": p_notes,
                    }

        # Compliance section
        if needs_compliance:
            container = right if right is not None else st.container()
            with container:
                with st.expander("**Compliance**", expanded=True):
                    violation = st.radio(
                        "Violation", ["no", "yes"], horizontal=True, key=f"{call_id}-violation"
                    )
                    verification = st.selectbox(
                        "Verification status", VERIFICATION_OPTIONS, key=f"{call_id}-verification"
                    )
                    vtype = st.selectbox(
                        "Violation type", VIOLATION_TYPE_OPTIONS, key=f"{call_id}-vtype"
                    )
                    c_notes = st.text_input("Notes", key=f"{call_id}-c-notes", placeholder="Optional")
                    compliance_values = {
                        "call_id": call_id,
                        "violation": violation,
                        "verification_status": verification,
                        "violation_type": vtype,
                        "notes": c_notes,
                    }

        submitted = st.form_submit_button("💾 Submit", type="primary", use_container_width=True)

    if submitted:
        if profanity_values and needs_profanity:
            _append_csv_row(PROFANITY_CSV, PROFANITY_FIELDS, profanity_values)
        if compliance_values and needs_compliance:
            _append_csv_row(COMPLIANCE_CSV, COMPLIANCE_FIELDS, compliance_values)

        # Rebuild index after writing so the next pick sees the new rows
        fresh = _build_state()
        sess = _session()
        sess["current"] = _pick_next(fresh["pending"])
        sess["show_form"] = False
        sess["flash"] = f"Saved `{call_id}`. Loading next file…"
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Annotation Tool", layout="wide")
    st.title("Annotation Tool")
    st.caption("Label profanity and compliance for debt collection call transcripts.")

    sess = _session()
    index = _build_state()

    _render_sidebar(index)

    # Flash message from previous submit
    if sess["flash"]:
        st.success(sess["flash"])
        sess["flash"] = ""

    # Progress counter
    total = len(index["all_files"])
    done = len(index["fully_annotated"])
    st.subheader(f"{done} / {total} files fully annotated")
    st.progress(done / total if total else 0)

    # Ensure we have a current file
    if sess["current"] is None or sess["current"] not in index["pending"]:
        sess["current"] = _pick_next(index["pending"])
        sess["show_form"] = False

    current = sess["current"]

    if current is None:
        st.balloons()
        st.success("All files are fully annotated (or intentionally skipped). Nothing left to label!")
        return

    st.divider()

    # File header and status
    needs_profanity = current not in index["profanity_ids"]
    needs_compliance = current not in index["compliance_ids"]
    status_parts = []
    if needs_profanity:
        status_parts.append("⚠️ profanity needed")
    if needs_compliance:
        status_parts.append("⚠️ compliance needed")

    col_title, _ = st.columns([3, 1])
    col_title.markdown(f"### `{current}`")
    col_title.caption("  ·  ".join(status_parts) if status_parts else "✅ fully annotated")

    # Navigation buttons
    btn1, btn2, btn3 = st.columns(3)
    if btn1.button("⏭ Skip", use_container_width=True, help="Load a different random file"):
        sess["show_form"] = False
        sess["current"] = _pick_next([f for f in index["pending"] if f != current])
        st.rerun()

    if btn2.button("✏️ Annotate", type="primary", use_container_width=True):
        sess["show_form"] = True
        st.rerun()

    if btn3.button("✅ Already Done", use_container_width=True, help="Mark as intentionally skipped and persist across sessions"):
        _append_csv_row(SKIPPED_CSV, SKIP_FIELDS, {"call_id": current})
        fresh = _build_state()
        sess["current"] = _pick_next(fresh["pending"])
        sess["show_form"] = False
        sess["flash"] = f"`{current}` marked as done — loading next file."
        st.rerun()

    st.divider()

    # Transcript
    _render_transcript(current)

    # Annotation form (only shown after Annotate button)
    if sess["show_form"]:
        st.divider()
        _render_annotation_form(current, needs_profanity, needs_compliance, index)


if __name__ == "__main__":
    main()
