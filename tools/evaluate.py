"""
Standalone evaluation script for the call evaluation system.

Run with:
    python tools/evaluate.py

Sections:
  1. Regex validity check     — all 252 transcript files, no API cost
  2. LLM accuracy check       — annotated files only
  3. Reproducibility check    — first 10 annotated files, 2 LLM runs each
  4. Hallucination check      — scans notes from section-2 outputs, no extra cost
  5. Blind spot check         — 15 random unannotated files, plausibility + cross-check

Output: console + reports/evaluation_report.txt
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Make src/ importable when run as a standalone script from any cwd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from call_evaluation.detectors.llm.compliance import LLMComplianceDetector
from call_evaluation.detectors.llm.profanity import LLMProfanityDetector
from call_evaluation.detectors.regex.compliance import (
    DISCLOSURE_PATTERNS,
    RegexComplianceDetector,
    VERIFICATION_PATTERNS,
)
from call_evaluation.detectors.regex.profanity import PROFANITY_PATTERNS, RegexProfanityDetector
from call_evaluation.ingestion import IngestionService
from call_evaluation.models.analysis import ContextLabel, SentimentLabel, SeverityLabel
from call_evaluation.models.transcript import SpeakerRole
from call_evaluation.services.llm_client import LLMClient
from call_evaluation.utils.text import normalize_for_matching, normalize_for_verification

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONVERSATIONS_DIR = ROOT / "All_Conversations"
PROFANITY_CSV     = ROOT / "data" / "labeled" / "annotations_profanity.csv"
COMPLIANCE_CSV    = ROOT / "data" / "labeled" / "annotations_compliance.csv"
REPORTS_DIR       = ROOT / "reports"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COST_PER_CALL_CENTS = 0.056
MAX_LLM_CALLS       = 200   # raised to accommodate section 5 (adds ~45 calls)
REPRO_COUNT         = 10
REPRO_GAP_SECONDS   = 5
BLIND_SPOT_COUNT    = 15

VALID_SEVERITY     = {e.value for e in SeverityLabel}
VALID_CONTEXT      = {e.value for e in ContextLabel}
VALID_SENTIMENT    = {e.value for e in SentimentLabel}
VALID_VIOLATION    = {"YES", "NO"}
VALID_VERIFICATION = {"VERIFIED", "PARTIAL", "UNVERIFIED", "NOT_APPLICABLE"}

SEVERITY_RANK: dict[str, int] = {"NONE": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}

# ---------------------------------------------------------------------------
# Dual-output report writer
# ---------------------------------------------------------------------------

class Report:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def line(self, text: str = "") -> None:
        print(text)
        self._fh.write(text + "\n")

    def close(self) -> None:
        self._fh.close()

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_conversations() -> list[str]:
    """Sorted list of regular (non-checkpoint) JSON filenames."""
    return sorted(
        p.name for p in CONVERSATIONS_DIR.iterdir()
        if p.suffix == ".json" and "checkpoint" not in p.name
    )


def _load_annotations(path: Path) -> dict[str, dict[str, str]]:
    """Return {call_id: row_dict} for every row in a CSV."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["call_id"]: row for row in csv.DictReader(fh) if row.get("call_id")}


def _load_payload(ingestion: IngestionService, call_id: str):
    return ingestion.load_path(CONVERSATIONS_DIR / call_id)[0]


def _raw_transcript_text(call_id: str) -> str:
    """Return the concatenated turn texts from the raw JSON (no normalization)."""
    try:
        data = json.loads((CONVERSATIONS_DIR / call_id).read_text(encoding="utf-8"))
        if isinstance(data, list):
            return " ".join(str(turn.get("text", "")) for turn in data)
    except Exception:
        pass
    return ""

# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _top_profanity_severity(agent_sev: str, customer_sev: str) -> str:
    """Return the higher of the two severity values."""
    return agent_sev if SEVERITY_RANK.get(agent_sev, 0) >= SEVERITY_RANK.get(customer_sev, 0) else customer_sev


def _top_profanity_context(agent_flag: bool, agent_ctx: str, customer_flag: bool, customer_ctx: str) -> str:
    if agent_flag and not customer_flag:
        return agent_ctx
    if customer_flag and not agent_flag:
        return customer_ctx
    return agent_ctx  # both or neither: use agent


def _pct(n: int, d: int) -> str:
    if d == 0:
        return "N/A"
    return f"{n}/{d} ({100*n/d:.1f}%)"

# ---------------------------------------------------------------------------
# Cost guard
# ---------------------------------------------------------------------------

def _calculate_projected_calls(prof_count: int, comp_count: int) -> int:
    repro       = min(REPRO_COUNT, prof_count) * 2 * 2  # agent+customer × 2 runs
    blind_spot  = BLIND_SPOT_COUNT * 3                  # 2 profanity + 1 compliance per file
    return prof_count * 2 + comp_count + repro + blind_spot


def cost_guard(prof_count: int, comp_count: int) -> int:
    total = _calculate_projected_calls(prof_count, comp_count)
    if total > MAX_LLM_CALLS:
        blind = BLIND_SPOT_COUNT * 3
        print(
            f"\nABORTED — projected LLM calls ({total}) exceeds hard limit ({MAX_LLM_CALLS}).\n"
            f"  Profanity accuracy:    {prof_count} files × 2 speakers = {prof_count*2}\n"
            f"  Compliance accuracy:   {comp_count} files = {comp_count}\n"
            f"  Reproducibility:       {min(REPRO_COUNT,prof_count)} files × 2 speakers × 2 runs = {min(REPRO_COUNT,prof_count)*2*2}\n"
            f"  Blind spot:            {BLIND_SPOT_COUNT} files × 3 calls = {blind}\n"
            f"Reduce annotation set or raise MAX_LLM_CALLS to proceed."
        )
        sys.exit(1)
    return total

# ---------------------------------------------------------------------------
# Section 1 — Regex validity check
# ---------------------------------------------------------------------------

def section1_regex_validity(report: Report, ingestion: IngestionService, all_files: list[str]) -> None:
    report.line()
    report.line("=" * 62)
    report.line("SECTION 1 — REGEX VALIDITY CHECK")
    report.line("=" * 62)

    prof_detector  = RegexProfanityDetector()
    comp_detector  = RegexComplianceDetector()
    failures: list[str] = []
    crashes: list[str]  = []
    passed = 0

    for call_id in all_files:
        try:
            payload = _load_payload(ingestion, call_id)
            if not payload.validation.is_valid:
                continue

            agent_result    = prof_detector.analyze(payload, SpeakerRole.AGENT)
            customer_result = prof_detector.analyze(payload, SpeakerRole.CUSTOMER)
            comp_result     = comp_detector.analyze(payload)

            file_ok = True

            # Profanity — agent
            if agent_result.severity.value not in VALID_SEVERITY:
                failures.append(f"  {call_id}: profanity[agent].severity = {agent_result.severity.value!r}")
                file_ok = False
            if agent_result.context.value not in VALID_CONTEXT:
                failures.append(f"  {call_id}: profanity[agent].context = {agent_result.context.value!r}")
                file_ok = False
            if agent_result.sentiment.value not in VALID_SENTIMENT:
                failures.append(f"  {call_id}: profanity[agent].sentiment = {agent_result.sentiment.value!r}")
                file_ok = False
            if agent_result.flag and not agent_result.evidence:
                failures.append(f"  {call_id}: profanity[agent] flag=True but evidence is empty")
                file_ok = False

            # Profanity — customer
            if customer_result.severity.value not in VALID_SEVERITY:
                failures.append(f"  {call_id}: profanity[customer].severity = {customer_result.severity.value!r}")
                file_ok = False
            if customer_result.context.value not in VALID_CONTEXT:
                failures.append(f"  {call_id}: profanity[customer].context = {customer_result.context.value!r}")
                file_ok = False
            if customer_result.sentiment.value not in VALID_SENTIMENT:
                failures.append(f"  {call_id}: profanity[customer].sentiment = {customer_result.sentiment.value!r}")
                file_ok = False
            if customer_result.flag and not customer_result.evidence:
                failures.append(f"  {call_id}: profanity[customer] flag=True but evidence is empty")
                file_ok = False

            # Compliance
            if comp_result.violation.value not in VALID_VIOLATION:
                failures.append(f"  {call_id}: compliance.violation = {comp_result.violation.value!r}")
                file_ok = False
            if comp_result.verification_status.value not in VALID_VERIFICATION:
                failures.append(f"  {call_id}: compliance.verification_status = {comp_result.verification_status.value!r}")
                file_ok = False
            if comp_result.violation.value == "YES" and not comp_result.evidence:
                failures.append(f"  {call_id}: compliance violation=YES but evidence is empty")
                file_ok = False

            if file_ok:
                passed += 1

        except Exception as exc:
            crashes.append(f"  {call_id}: CRASH — {exc}")

    total = len(all_files)
    report.line(f"Files processed : {total}")
    report.line(f"Validity passed : {_pct(passed, total)}")
    if failures:
        report.line(f"Validity failures ({len(failures)}):")
        for f in failures:
            report.line(f)
    else:
        report.line("Validity failures: none")
    if crashes:
        report.line(f"Crashes ({len(crashes)}):")
        for c in crashes:
            report.line(c)
    else:
        report.line("Crashes: none")

# ---------------------------------------------------------------------------
# Section 2 — LLM accuracy check
# ---------------------------------------------------------------------------

def section2_llm_accuracy(
    report: Report,
    ingestion: IngestionService,
    prof_annotations: dict[str, dict[str, str]],
    comp_annotations: dict[str, dict[str, str]],
    llm_prof: LLMProfanityDetector,
    llm_comp: LLMComplianceDetector,
) -> tuple[dict[str, tuple], dict[str, object], int]:
    """
    Returns:
        prof_outputs  — {call_id: (agent_result, customer_result)}
        comp_outputs  — {call_id: comp_result}
        api_calls     — number of LLM calls made
    """
    report.line()
    report.line("=" * 62)
    report.line("SECTION 2 — LLM ACCURACY CHECK")
    report.line("=" * 62)

    api_calls = 0
    prof_outputs: dict[str, tuple] = {}
    comp_outputs: dict[str, object] = {}

    # --- Profanity accuracy ---
    p_agent_correct = p_customer_correct = 0
    p_sev_correct = p_ctx_correct = 0
    p_total = len(prof_annotations)

    for call_id, ann in prof_annotations.items():
        try:
            payload = _load_payload(ingestion, call_id)
            agent_r   = llm_prof.analyze(payload, SpeakerRole.AGENT);   api_calls += 1
            customer_r = llm_prof.analyze(payload, SpeakerRole.CUSTOMER); api_calls += 1
            prof_outputs[call_id] = (agent_r, customer_r)

            gt_agent    = ann["agent_profanity"].strip().lower() == "yes"
            gt_customer = ann["customer_profanity"].strip().lower() == "yes"
            gt_severity = ann["severity"].strip().upper()
            gt_context  = ann["context"].strip().upper()

            if agent_r.flag == gt_agent:
                p_agent_correct += 1
            if customer_r.flag == gt_customer:
                p_customer_correct += 1

            pred_severity = _top_profanity_severity(agent_r.severity.value, customer_r.severity.value)
            if pred_severity == gt_severity:
                p_sev_correct += 1

            pred_context = _top_profanity_context(
                agent_r.flag, agent_r.context.value,
                customer_r.flag, customer_r.context.value,
            )
            # N/A context in annotations means no profanity — match against AMBIENT
            if gt_context in {"N/A", ""}:
                gt_context = "AMBIENT"
            if pred_context == gt_context:
                p_ctx_correct += 1

        except Exception as exc:
            report.line(f"  WARNING: profanity LLM failed for {call_id}: {exc}")

    report.line()
    report.line(f"Profanity ({p_total} annotated files):")
    report.line(f"  agent accuracy    : {_pct(p_agent_correct, p_total)}")
    report.line(f"  customer accuracy : {_pct(p_customer_correct, p_total)}")
    report.line(f"  severity accuracy : {_pct(p_sev_correct, p_total)}")
    report.line(f"  context accuracy  : {_pct(p_ctx_correct, p_total)}")

    # --- Compliance accuracy ---
    c_viol_correct  = c_verif_correct = c_vtype_correct = c_exact = 0
    c_total = len(comp_annotations)

    for call_id, ann in comp_annotations.items():
        try:
            payload  = _load_payload(ingestion, call_id)
            result   = llm_comp.analyze(payload); api_calls += 1
            comp_outputs[call_id] = result

            gt_viol  = ann["violation"].strip().upper()
            gt_verif = ann["verification_status"].strip().upper()
            gt_vtype = ann["violation_type"].strip().upper()
            if gt_vtype == "NO_VIOLATION":
                gt_vtype = "NOT_APPLICABLE"

            pred_viol  = result.violation.value
            pred_verif = result.verification_status.value
            pred_vtype = result.violation_type.strip().upper()

            if pred_viol == gt_viol:
                c_viol_correct += 1
            if pred_verif == gt_verif:
                c_verif_correct += 1
            if pred_vtype == gt_vtype:
                c_vtype_correct += 1
            if pred_viol == gt_viol and pred_verif == gt_verif and pred_vtype == gt_vtype:
                c_exact += 1

        except Exception as exc:
            report.line(f"  WARNING: compliance LLM failed for {call_id}: {exc}")

    report.line()
    report.line(f"Compliance ({c_total} annotated files):")
    report.line(f"  violation accuracy         : {_pct(c_viol_correct, c_total)}")
    report.line(f"  verification accuracy      : {_pct(c_verif_correct, c_total)}")
    report.line(f"  violation type accuracy    : {_pct(c_vtype_correct, c_total)}")
    report.line(f"  exact match (all 3 fields) : {_pct(c_exact, c_total)}")

    return prof_outputs, comp_outputs, api_calls

# ---------------------------------------------------------------------------
# Section 3 — Reproducibility check
# ---------------------------------------------------------------------------

def section3_reproducibility(
    report: Report,
    ingestion: IngestionService,
    prof_annotations: dict[str, dict[str, str]],
    llm_prof: LLMProfanityDetector,
) -> int:
    report.line()
    report.line("=" * 62)
    report.line("SECTION 3 — REPRODUCIBILITY CHECK")
    report.line("=" * 62)

    call_ids = list(prof_annotations.keys())[:REPRO_COUNT]
    api_calls = 0

    run1: dict[str, tuple] = {}
    run2: dict[str, tuple] = {}

    report.line(f"Files: {len(call_ids)}  |  Gap between runs: {REPRO_GAP_SECONDS}s")
    report.line("Running first pass...")

    for call_id in call_ids:
        try:
            payload = _load_payload(ingestion, call_id)
            a = llm_prof.analyze(payload, SpeakerRole.AGENT);    api_calls += 1
            c = llm_prof.analyze(payload, SpeakerRole.CUSTOMER); api_calls += 1
            run1[call_id] = (a, c)
        except Exception as exc:
            report.line(f"  WARNING: run1 failed for {call_id}: {exc}")

    report.line(f"Waiting {REPRO_GAP_SECONDS}s before second pass...")
    time.sleep(REPRO_GAP_SECONDS)
    report.line("Running second pass...")

    for call_id in call_ids:
        try:
            payload = _load_payload(ingestion, call_id)
            a = llm_prof.analyze(payload, SpeakerRole.AGENT);    api_calls += 1
            c = llm_prof.analyze(payload, SpeakerRole.CUSTOMER); api_calls += 1
            run2[call_id] = (a, c)
        except Exception as exc:
            report.line(f"  WARNING: run2 failed for {call_id}: {exc}")

    compared = [cid for cid in call_ids if cid in run1 and cid in run2]
    n = len(compared)

    fields = ["flag", "severity", "context", "sentiment"]
    counts: dict[str, int] = {f: 0 for f in fields}
    diffs: list[str] = []

    for call_id in compared:
        for role_idx in (0, 1):
            r1 = run1[call_id][role_idx]
            r2 = run2[call_id][role_idx]
            role = "agent" if role_idx == 0 else "customer"

            vals1 = {
                "flag":      r1.flag,
                "severity":  r1.severity.value,
                "context":   r1.context.value,
                "sentiment": r1.sentiment.value,
            }
            vals2 = {
                "flag":      r2.flag,
                "severity":  r2.severity.value,
                "context":   r2.context.value,
                "sentiment": r2.sentiment.value,
            }
            all_match = True
            for f in fields:
                if vals1[f] == vals2[f]:
                    counts[f] += 1
                else:
                    all_match = False
                    diffs.append(f"  {call_id} [{role}].{f}: run1={vals1[f]!r} vs run2={vals2[f]!r}")

    total_comparisons = n * 2  # agent + customer per file
    report.line()
    for f in fields:
        report.line(f"  {f:<12} reproducibility: {_pct(counts[f], total_comparisons)}")
    if diffs:
        report.line(f"\nDifferences ({len(diffs)}):")
        for d in diffs:
            report.line(d)
    else:
        report.line("\nNo differences detected between runs.")

    return api_calls

# ---------------------------------------------------------------------------
# Section 4 — Hallucination check
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "not",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "that",
    "this", "it", "be", "has", "have", "had", "but", "if", "they",
    "their", "there", "then", "than", "no", "yes", "agent", "customer",
    "llm", "transcript", "information", "verification", "call",
}
_ENUM_VALUES = (
    VALID_SEVERITY | VALID_CONTEXT | VALID_SENTIMENT | VALID_VIOLATION | VALID_VERIFICATION
    | {"NOT_APPLICABLE", "NO_VERIFICATION", "ACCOUNT_DETAILS_BEFORE_VERIFICATION"}
)


def _extract_suspicious_tokens(notes: str) -> dict[str, list[str]]:
    """Extract numbers, quoted strings, and mid-sentence proper nouns from notes."""
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", notes)
    quoted  = re.findall(r'"([^"]{3,})"', notes)
    # Proper nouns: Title-case words not at sentence start, not enums or stop words
    words = re.findall(r"(?<!\.\s)(?<!\A)\b([A-Z][a-z]{2,})\b", notes)
    proper = [w for w in words if w.upper() not in _ENUM_VALUES and w.lower() not in _STOP_WORDS]
    return {"numbers": numbers, "quoted": quoted, "proper_nouns": proper}


def section4_hallucination(
    report: Report,
    prof_outputs: dict[str, tuple],
    comp_outputs: dict[str, object],
) -> None:
    report.line()
    report.line("=" * 62)
    report.line("SECTION 4 — HALLUCINATION CHECK (notes fields)")
    report.line("=" * 62)

    checked = 0
    flagged: list[str] = []

    # Collect all (call_id, notes) pairs from LLM outputs
    notes_to_check: list[tuple[str, str]] = []
    for call_id, (agent_r, customer_r) in prof_outputs.items():
        if agent_r.notes:
            notes_to_check.append((call_id, "profanity[agent]", agent_r.notes))
        if customer_r.notes:
            notes_to_check.append((call_id, "profanity[customer]", customer_r.notes))
    for call_id, comp_r in comp_outputs.items():
        if comp_r.notes:
            notes_to_check.append((call_id, "compliance", comp_r.notes))

    for call_id, label, notes in notes_to_check:
        checked += 1
        transcript_text = _raw_transcript_text(call_id).lower()
        tokens = _extract_suspicious_tokens(notes)
        invented: list[str] = []

        for num in tokens["numbers"]:
            if num not in transcript_text:
                invented.append(f"number {num!r} not in transcript")
        for q in tokens["quoted"]:
            if q.lower() not in transcript_text:
                invented.append(f"quoted text {q!r} not in transcript")
        for pn in tokens["proper_nouns"]:
            if pn.lower() not in transcript_text:
                invented.append(f"proper noun {pn!r} not in transcript")

        if invented:
            flagged.append(f"  {call_id} [{label}]: {'; '.join(invented)}")
            flagged.append(f"    notes: {notes[:120]}{'...' if len(notes)>120 else ''}")

    report.line(f"Notes fields checked : {checked}")
    report.line(f"Flagged              : {len(flagged) // 2}")  # 2 lines per flag
    if flagged:
        report.line("Flagged examples:")
        for line in flagged:
            report.line(line)
    else:
        report.line("No hallucinated content detected.")

# ---------------------------------------------------------------------------
# Section 5 — Blind spot check helpers
# ---------------------------------------------------------------------------

def _evidence_has_lexicon_match(evidence_spans: list) -> bool:
    """Return True if any evidence span text contains a profanity lexicon hit."""
    for span in evidence_spans:
        normalized = normalize_for_matching(span.text)
        if any(pat.search(normalized) for pat in PROFANITY_PATTERNS.values()):
            return True
    return False


def _transcript_has_disclosure(payload) -> bool:
    """Return True if any agent turn contains a disclosure pattern match."""
    for turn in payload.turns:
        if turn.speaker == SpeakerRole.AGENT:
            normalized = normalize_for_verification(turn.text)
            if any(pat.search(normalized) for pat in DISCLOSURE_PATTERNS):
                return True
    return False


def _validity_issues(call_id: str, agent_r, customer_r, comp_r) -> list[str]:
    """Return a list of validity problem strings, empty if all clean."""
    issues: list[str] = []
    for label, r in (("agent", agent_r), ("customer", customer_r)):
        if r.severity.value not in VALID_SEVERITY:
            issues.append(f"profanity[{label}].severity={r.severity.value!r}")
        if r.context.value not in VALID_CONTEXT:
            issues.append(f"profanity[{label}].context={r.context.value!r}")
        if r.sentiment.value not in VALID_SENTIMENT:
            issues.append(f"profanity[{label}].sentiment={r.sentiment.value!r}")
        if r.flag and not r.evidence:
            issues.append(f"profanity[{label}] flag=True but evidence empty")
    if comp_r.violation.value not in VALID_VIOLATION:
        issues.append(f"compliance.violation={comp_r.violation.value!r}")
    if comp_r.verification_status.value not in VALID_VERIFICATION:
        issues.append(f"compliance.verification_status={comp_r.verification_status.value!r}")
    if comp_r.violation.value == "YES" and not comp_r.evidence:
        issues.append("compliance violation=YES but evidence empty")
    return issues


# ---------------------------------------------------------------------------
# Section 5 — Blind spot check
# ---------------------------------------------------------------------------

def section5_blind_spot(
    report: Report,
    ingestion: IngestionService,
    all_files: list[str],
    annotated_ids: set[str],
    llm_prof: LLMProfanityDetector,
    llm_comp: LLMComplianceDetector,
) -> int:
    report.line()
    report.line("=" * 62)
    report.line("SECTION 5 — BLIND SPOT CHECK")
    report.line("=" * 62)

    candidates = [f for f in all_files if f not in annotated_ids]
    sample = random.sample(candidates, min(BLIND_SPOT_COUNT, len(candidates)))

    report.line(f"Unannotated pool : {len(candidates)} files")
    report.line(f"Sample selected  : {len(sample)} files")
    report.line()

    regex_prof = RegexProfanityDetector()
    regex_comp = RegexComplianceDetector()
    api_calls  = 0

    validity_failures: list[str] = []
    rows: list[dict] = []

    for call_id in sample:
        try:
            payload = _load_payload(ingestion, call_id)
            if not payload.validation.is_valid:
                rows.append({
                    "call_id": call_id,
                    "prof_flag": "—",
                    "violation": "—",
                    "verdict": "SKIP",
                    "explanation": "transcript failed validation",
                })
                continue

            # LLM inference
            agent_r    = llm_prof.analyze(payload, SpeakerRole.AGENT);    api_calls += 1
            customer_r = llm_prof.analyze(payload, SpeakerRole.CUSTOMER); api_calls += 1
            comp_r     = llm_comp.analyze(payload);                        api_calls += 1

            # Output validity
            issues = _validity_issues(call_id, agent_r, customer_r, comp_r)
            if issues:
                for issue in issues:
                    validity_failures.append(f"  {call_id}: {issue}")

            prof_flag_str = (
                f"agent={'T' if agent_r.flag else 'F'}, "
                f"customer={'T' if customer_r.flag else 'F'}"
            )
            violation_str = comp_r.violation.value

            verdicts: list[tuple[str, str]] = []

            # --- Profanity plausibility ---
            if agent_r.flag and not _evidence_has_lexicon_match(agent_r.evidence):
                verdicts.append(("SUSPECT", "agent flagged but no lexicon words in evidence"))
            if customer_r.flag and not _evidence_has_lexicon_match(customer_r.evidence):
                verdicts.append(("SUSPECT", "customer flagged but no lexicon words in evidence"))

            # --- Compliance plausibility ---
            if comp_r.violation.value == "YES" and not _transcript_has_disclosure(payload):
                verdicts.append(("SUSPECT", "violation=YES but no disclosure patterns in transcript"))

            # --- Cross-check when both LLM outputs are fully negative ---
            if not agent_r.flag and not customer_r.flag and comp_r.violation.value == "NO":
                rp_a = regex_prof.analyze(payload, SpeakerRole.AGENT)
                rp_c = regex_prof.analyze(payload, SpeakerRole.CUSTOMER)
                rc   = regex_comp.analyze(payload)
                divergences: list[str] = []
                if rp_a.flag:
                    divergences.append("regex flagged agent profanity")
                if rp_c.flag:
                    divergences.append("regex flagged customer profanity")
                if rc.violation.value == "YES":
                    divergences.append("regex found compliance violation")
                if divergences:
                    verdicts.append(("DIVERGENT", "; ".join(divergences)))
                else:
                    verdicts.append(("CONSISTENT", "LLM and regex both negative"))

            # Derive overall verdict (SUSPECT > DIVERGENT > CONSISTENT)
            if any(v[0] == "SUSPECT" for v in verdicts):
                overall = "SUSPECT"
            elif any(v[0] == "DIVERGENT" for v in verdicts):
                overall = "DIVERGENT"
            elif verdicts:
                overall = verdicts[0][0]
            else:
                overall = "CONSISTENT"  # positive and plausibility checks passed

            explanation = "; ".join(v[1] for v in verdicts) or "positive result, evidence consistent with transcript"

            rows.append({
                "call_id":     call_id,
                "prof_flag":   prof_flag_str,
                "violation":   violation_str,
                "verdict":     overall,
                "explanation": explanation,
            })

        except Exception as exc:
            rows.append({
                "call_id":     call_id,
                "prof_flag":   "—",
                "violation":   "—",
                "verdict":     "CRASH",
                "explanation": str(exc),
            })

    # Print results table
    col_w = 52
    report.line(f"{'call_id':<44} {'profanity':<22} {'violation':<12} {'verdict':<12} explanation")
    report.line("-" * 140)
    for row in rows:
        cid  = row["call_id"][:43]
        report.line(
            f"{cid:<44} {row['prof_flag']:<22} {row['violation']:<12} {row['verdict']:<12} {row['explanation']}"
        )

    # Validity summary
    report.line()
    suspect_count   = sum(1 for r in rows if r["verdict"] == "SUSPECT")
    divergent_count = sum(1 for r in rows if r["verdict"] == "DIVERGENT")
    consistent_count = sum(1 for r in rows if r["verdict"] == "CONSISTENT")

    report.line(f"Consistent : {consistent_count}/{len(rows)}")
    report.line(f"Divergent  : {divergent_count}/{len(rows)}  (LLM negative, regex positive — worth human review)")
    report.line(f"Suspect    : {suspect_count}/{len(rows)}  (LLM positive, no supporting evidence/patterns)")

    if validity_failures:
        report.line(f"\nValidity failures ({len(validity_failures)}):")
        for f in validity_failures:
            report.line(f)
    else:
        report.line("Validity failures: none")

    return api_calls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = REPORTS_DIR / "evaluation_report.txt"
    report = Report(report_path)

    report.line("=" * 62)
    report.line("CALL EVALUATION — AUTOMATED EVALUATION REPORT")
    report.line(f"Generated : {timestamp}")
    report.line(f"Report    : {report_path}")
    report.line("=" * 62)

    ingestion = IngestionService()
    all_files = [
        p.name for p in sorted(CONVERSATIONS_DIR.iterdir())
        if p.suffix == ".json" and "checkpoint" not in p.name
    ]

    prof_annotations = _load_annotations(PROFANITY_CSV)
    comp_annotations = _load_annotations(COMPLIANCE_CSV)

    report.line()
    report.line(f"Transcript files      : {len(all_files)}")
    report.line(f"Profanity annotations : {len(prof_annotations)}")
    report.line(f"Compliance annotations: {len(comp_annotations)}")

    # --- Section 1: always runs, free ---
    section1_regex_validity(report, ingestion, all_files)

    # --- Check LLM availability ---
    llm_client = LLMClient()
    llm_state  = llm_client.get_runtime_state()

    if not llm_state.api_available:
        report.line()
        report.line("=" * 62)
        report.line("LLM SECTIONS SKIPPED")
        report.line(f"Reason: {llm_state.message}")
        report.line("Sections 2, 3, 4, and 5 require OPENAI_API_KEY to be set.")
        report.line("=" * 62)
        report.line()
        report.line(f"Total LLM API calls : 0")
        report.line(f"Estimated cost      : $0.00  (0 calls × ${COST_PER_CALL_CENTS/100:.5f})")
        report.close()
        return

    # --- Cost guard ---
    projected = cost_guard(len(prof_annotations), len(comp_annotations))
    report.line()
    report.line(f"Projected LLM calls : {projected} (limit: {MAX_LLM_CALLS})")
    report.line(f"Estimated cost      : {projected * COST_PER_CALL_CENTS:.3f} cents  (${projected * COST_PER_CALL_CENTS / 100:.5f})")

    llm_prof = LLMProfanityDetector(llm_client=llm_client)
    llm_comp = LLMComplianceDetector(llm_client=llm_client)

    total_api_calls = 0

    # --- Sections 2, 3, 4 ---
    prof_outputs, comp_outputs, calls2 = section2_llm_accuracy(
        report, ingestion, prof_annotations, comp_annotations, llm_prof, llm_comp
    )
    total_api_calls += calls2

    calls3 = section3_reproducibility(report, ingestion, prof_annotations, llm_prof)
    total_api_calls += calls3

    section4_hallucination(report, prof_outputs, comp_outputs)

    annotated_ids = set(prof_annotations.keys()) | set(comp_annotations.keys())
    calls5 = section5_blind_spot(
        report, ingestion, all_files, annotated_ids, llm_prof, llm_comp
    )
    total_api_calls += calls5

    # --- Final summary ---
    actual_cost = total_api_calls * COST_PER_CALL_CENTS
    report.line()
    report.line("=" * 62)
    report.line("SUMMARY")
    report.line("=" * 62)
    report.line(f"Total LLM API calls : {total_api_calls}")
    report.line(f"Estimated cost      : {actual_cost:.3f} cents  (${actual_cost / 100:.5f})")
    report.line(f"Report written to   : {report_path}")
    report.line("=" * 62)
    report.close()


if __name__ == "__main__":
    main()
