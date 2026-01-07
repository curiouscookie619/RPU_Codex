from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import os
import re
import time
import uuid
import zipfile
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

import streamlit as st

from core.db import init_db, try_get_conn
from core.event_logger import log_event
from core.pdf_reader import read_pdf
from core.renderers.gis_html import render_gis_renewal_html
from products.registry import detect_product, ProductNotConfigured
from core.output_pdf import render_pdf_from_html
from core.irr import attach_irrs, build_irr_debug, add_years  # type: ignore[attr-defined]


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@st.cache_data(show_spinner=False, ttl=3600, max_entries=128)
def _cached_read_pdf(file_bytes: bytes):
    # cache by content (Streamlit cache uses argument hashing)
    return read_pdf(file_bytes)


def make_case_id(
    product_id: str,
    product_uin: Optional[str],
    bi_date: date,
    ptd: date,
    rcd: date,
    mode: str,
    pt: int,
    ppt: int,
    annualized_premium: Optional[float],
    proposer_name_transient: Optional[str],
) -> str:
    raw = "|".join(
        [
            product_id,
            product_uin or "",
            str(bi_date),
            str(ptd),
            str(rcd),
            mode,
            str(pt),
            str(ppt),
            str(annualized_premium or ""),
            (proposer_name_transient or "").strip().lower(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_case(
    case_id: str,
    session_id: str,
    product_id: str,
    product_confidence: float,
    bi_date: date,
    ptd: date,
    rcd: date,
    rpu_date: date,
    mode: str,
    file_hash: str,
    extracted_json: Dict[str, Any],
    outputs_json: Dict[str, Any],
) -> None:
    conn = try_get_conn()
    if conn is None:
        return

    with conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO cases(case_id, session_id, product_id, product_confidence, bi_date, ptd, rcd, rpu_date,
                             mode, file_hash, extracted, outputs)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
            ON CONFLICT (case_id) DO UPDATE SET
                session_id=EXCLUDED.session_id,
                product_id=EXCLUDED.product_id,
                product_confidence=EXCLUDED.product_confidence,
                bi_date=EXCLUDED.bi_date,
                ptd=EXCLUDED.ptd,
                rcd=EXCLUDED.rcd,
                rpu_date=EXCLUDED.rpu_date,
                mode=EXCLUDED.mode,
                file_hash=EXCLUDED.file_hash,
                extracted=EXCLUDED.extracted,
                outputs=EXCLUDED.outputs,
                updated_at=NOW()
            """,
            (
                case_id,
                session_id,
                product_id,
                product_confidence,
                bi_date,
                ptd,
                rcd,
                rpu_date,
                mode,
                file_hash,
                json.dumps(extracted_json, default=str),
                json.dumps(outputs_json, default=str),
            ),
        )
        conn.commit()


def _segments_from_income_items(items: list[dict]) -> list[dict]:
    """Group income items (calendar_year, amount) into segments with constant amount."""
    # items must have calendar_year:int and amount:float
    clean = []
    for it in items or []:
        y = it.get("calendar_year")
        a = it.get("amount")
        if y is None or a is None:
            continue
        try:
            clean.append((int(y), float(a)))
        except Exception:
            continue
    clean.sort(key=lambda x: x[0])
    segs: list[dict] = []
    if not clean:
        return segs
    cur_start, cur_end, cur_amt = clean[0][0], clean[0][0], clean[0][1]
    for y, a in clean[1:]:
        if a == cur_amt and y == cur_end + 1:
            cur_end = y
        else:
            segs.append({"start_year": cur_start, "end_year": cur_end, "amount": cur_amt, "years": (cur_end - cur_start + 1)})
            cur_start, cur_end, cur_amt = y, y, a
    segs.append({"start_year": cur_start, "end_year": cur_end, "amount": cur_amt, "years": (cur_end - cur_start + 1)})
    return segs


def _fmt_money(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):,.0f}"
    except Exception:
        return str(v)


def _render_income_segments_bullets(segments: list[dict], title: str, scale: float = 1.0):
    """Render income segments in calendar years (Option 1) as short bullets."""
    st.markdown(f"**{title}**")
    if not segments:
        st.write("- (No income rows detected)")
        return

    def fmt(v):
        return _fmt_money((float(v) * scale) if v is not None else None)

    for seg in segments:
        kind = seg.get("kind")
        if kind == "continuous_constant":
            amt = seg.get("amount")
            st.write(f"- ₹{fmt(amt)} every year from **{seg.get('start_year')}** to **{seg.get('end_year')}** ({seg.get('count')} years)")
        elif kind == "continuous_varying":
            st.write(
                f"- From ₹{fmt(seg.get('start_amount'))} to ₹{fmt(seg.get('end_amount'))} every year from **{seg.get('start_year')}** to **{seg.get('end_year')}** ({seg.get('count')} years)"
            )
        elif kind == "discrete_constant":
            years = seg.get("years") or []
            years_s = ", ".join(str(y) for y in years[:10])
            more = f" +{len(years)-10} more" if len(years) > 10 else ""
            st.write(f"- ₹{fmt(seg.get('amount'))} in {years_s}{more} ({seg.get('count')} payouts)")
        elif kind == "discrete_varying":
            items = seg.get("items") or []
            parts = [f"{it.get('year')}: ₹{fmt(it.get('amount'))}" for it in items[:8]]
            more = f" +{len(items)-8} more" if len(items) > 8 else ""
            st.write(f"- " + "; ".join(parts) + f"{more} ({seg.get('count')} payouts)")
        else:
            # Fallback
            st.write(f"- {seg}")


def _bucket_mapping_rows(breakdown: list[dict], rcd: date, start_bucket: int) -> list[dict]:
    rows = []
    for idx in range(start_bucket, len(breakdown)):
        b = breakdown[idx]
        net = (b.get("premium", 0.0) or 0.0) + (b.get("income", 0.0) or 0.0) + (b.get("maturity", 0.0) or 0.0) + (
            b.get("surrender", 0.0) or 0.0
        )
        rows.append(
            {
                "bucket": idx,
                "date": str(add_years(rcd, idx - 1)),
                "premium": b.get("premium", 0.0) or 0.0,
                "income": b.get("income", 0.0) or 0.0,
                "maturity": b.get("maturity", 0.0) or 0.0,
                "surrender": b.get("surrender", 0.0) or 0.0,
                "net": net,
            }
        )
    return rows


def _vector_head_tail(vec: list[float], head: int = 10, tail: int = 5) -> dict:
    return {
        "length": len(vec),
        "head": vec[: min(len(vec), head)],
        "tail": vec[-tail:] if len(vec) > tail else vec[:],
    }


def _compare_lists(actual: list[float], expected: list[float], tol: float) -> dict:
    if len(actual) != len(expected):
        return {"pass": False, "reason": f"length mismatch (got {len(actual)} expected {len(expected)})"}
    diffs = [abs(a - e) for a, e in zip(actual, expected)]
    max_diff = max(diffs) if diffs else 0.0
    return {"pass": all(d <= tol for d in diffs), "reason": f"max diff={max_diff:.4f}"}


def _expected_bi2_vectors():
    expected_rpu_cf = (
        [-1500000.0, -1500000.0, -1002150.0, 497850.0]
        + [101832.9545] * 9
        + [226295.4545] * 23
        + [4726295.4545]
    )
    expected_fp_incr_cf = (
        [-3110611.0]
        + [-1002150.0] * 8
        + [497850.0]
        + [995700.0] * 23
        + [18995700.0]
    )
    return expected_rpu_cf, expected_fp_incr_cf


def _parse_ptd(value: str) -> date:
    text = (value or "").strip()
    if not text:
        raise ValueError("PTD is required")
    if re.fullmatch(r"\d+(\.\d+)?", text):
        days = int(float(text))
        return (datetime(1899, 12, 30) + timedelta(days=days)).date()
    if "/" in text:
        return datetime.strptime(text, "%d/%m/%Y").date()
    if "-" in text:
        return datetime.strptime(text, "%d-%m-%Y").date()
    return datetime.strptime(text, "%Y-%m-%d").date()


def _parse_surrender_value(value: str) -> float:
    text = (value or "").strip()
    if not text:
        raise ValueError("surrender_value is required")
    return float(text.replace(",", ""))


def _normalize_policy_number(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _policy_number_from_filename(filename: str) -> Optional[str]:
    base = os.path.basename(filename or "")
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    matches = list(re.finditer(r"\d+", base))
    if not matches:
        return None
    best_match = max(matches, key=lambda m: (len(m.group(0)), -m.start()))
    return _normalize_policy_number(best_match.group(0)) or None


def _read_batch_csv(csv_bytes: bytes) -> dict[str, dict[str, str]]:
    text = csv_bytes.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    expected = ["policy_number", "ptd", "surrender_value"]
    if header != expected:
        raise ValueError(f"CSV header must be exactly {expected}")
    mapping: dict[str, dict[str, str]] = {}
    for row in reader:
        if not row or all(not (cell or "").strip() for cell in row):
            continue
        if len(row) < 3:
            raise ValueError("CSV row must have 3 columns")
        policy_number = _normalize_policy_number(row[0])
        if not policy_number:
            continue
        mapping[policy_number] = {"ptd": row[1], "surrender_value": row[2]}
    return mapping


def _write_csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _run_single_mode(session_id: str) -> None:
    # Use a form to avoid rerun/button non-responsiveness
    with st.form("main_form"):
        debug = st.checkbox("Debug mode (show what was extracted)", value=False)
        uploaded = st.file_uploader("Upload BI PDF", type=["pdf"])
        ptd = st.date_input("PTD (Next Premium Due Date)", value=None, format="DD/MM/YYYY")
        surrender_value_raw = st.text_input(
            "Surrender Value (₹)",
            value="",
            placeholder="Required for IRR",
            help="Enter surrender value to compute IRRs; leave blank to skip IRR.",
        )
        submitted = st.form_submit_button("Generate")

    if not submitted:
        return

    if uploaded is None or ptd is None:
        st.error("Please upload a PDF and enter PTD.")
        return

    file_bytes = uploaded.getvalue()
    file_hash = _sha256_bytes(file_bytes)

    log_event("pdf_uploaded", session_id, {"file_hash": file_hash, "size_bytes": len(file_bytes), "device": "unknown"})

    # Parse surrender value (optional, but required for IRR)
    surrender_value = None
    if surrender_value_raw.strip():
        try:
            surrender_value = float(surrender_value_raw.replace(",", "").strip())
        except Exception:
            st.error("Please enter a valid numeric surrender value or leave it blank to skip IRR.")
            return

    try:
        # Cached parsing for speed (especially repeated attempts)
        parsed = _cached_read_pdf(file_bytes)
        log_event("pdf_parsed", session_id, {"pages": parsed.page_count})

        handler, conf, dbg = detect_product(parsed)
        log_event("product_detected", session_id, {"product_id": handler.product_id, "confidence": conf, "dbg": dbg})

        extracted = handler.extract(parsed)
        outputs = handler.calculate(extracted, ptd)

        log_event(
            "bi_metadata",
            session_id,
            {
                "product_id": handler.product_id,
                "policyholder_name": (extracted.proposer_name_transient or "").strip() or None,
                "annualized_premium_excl_tax": extracted.annualized_premium_excl_tax,
            },
        )

        extracted_dump = extracted.model_dump(mode="json")
        outputs_dump = outputs.model_dump(mode="json")

        # Build case_id (we store only hashes, not proposer name)
        case_id = make_case_id(
            product_id=handler.product_id,
            product_uin=extracted.product_uin,
            bi_date=extracted.bi_generation_date,
            ptd=ptd,
            rcd=outputs.rcd,
            mode=extracted.mode,
            pt=extracted.policy_term_years,
            ppt=extracted.ppt_years,
            annualized_premium=extracted.annualized_premium_excl_tax,
            proposer_name_transient=extracted.proposer_name_transient,
        )

        save_case(
            case_id=case_id,
            session_id=session_id,
            product_id=handler.product_id,
            product_confidence=conf,
            bi_date=extracted.bi_generation_date,
            ptd=ptd,
            rcd=outputs.rcd,
            rpu_date=outputs.rpu_date,
            mode=extracted.mode,
            file_hash=file_hash,
            extracted_json=extracted_dump,
            outputs_json=outputs_dump,
        )

        log_event("output_generated", session_id, {"case_id": case_id, "product_id": handler.product_id})

        # ---------- DEBUG ----------
        if debug:
            st.subheader("DEBUG: Extracted object (raw)")
            st.json(extracted_dump)

            st.subheader("DEBUG: Schedule preview")
            schedule_rows = extracted_dump.get("schedule_rows") or []
            st.write(f"Schedule rows found = {len(schedule_rows)}")
            if schedule_rows:
                st.dataframe(schedule_rows[:20], use_container_width=True)

        # ---------- OUTPUT SUMMARY ----------
        st.divider()
        st.subheader("Key Dates Summary")
        st.json(
            {
                "BI (Quote) Date": str(extracted.bi_generation_date),
                "RCD (Derived)": str(outputs.rcd),
                "PTD (Input)": str(ptd),
                "Assumed RPU Date (PTD + Grace)": str(outputs.rpu_date),
                "Grace Period Days": outputs.grace_period_days,
            }
        )

        st.divider()
        st.subheader("Renewal decision view")
        # Compute IRRs only when surrender value is provided
        if surrender_value is not None:
            attach_irrs(extracted, outputs, surrender_value)
        else:
            outputs.irr_rpu = None
            outputs.irr_fp_incremental = None
        html_out = render_gis_renewal_html(extracted, outputs, surrender_value)
        st.components.v1.html(html_out, height=1200, scrolling=True)

        # ---------- IRR DEBUG (GIS BI 2) ----------
        if debug and handler.product_id == "GIS" and surrender_value is not None:
            debug_data = build_irr_debug(extracted, outputs, surrender_value or 0.0)
            rpu_vec = debug_data.get("rpu_cf", [])
            fp_vec = debug_data.get("fp_incremental_cf", [])

            st.subheader("Debug: IRR Vectors (GIS)")
            with st.expander("IRR vectors, mapping, and checks", expanded=True):
                st.write(
                    {
                        "rpu_vector": _vector_head_tail(rpu_vec),
                        "fp_incremental_vector": _vector_head_tail(fp_vec),
                        "rpu_irr": debug_data.get("rpu_irr"),
                        "fp_incremental_irr": debug_data.get("fp_incremental_irr"),
                    }
                )

                st.write("First 10 values (RPU / FP incremental)")
                st.json({"rpu": rpu_vec[:10], "fp_incremental": fp_vec[:10]})
                st.write("Last 5 values (RPU / FP incremental)")
                st.json({"rpu": rpu_vec[-5:], "fp_incremental": fp_vec[-5:]})
                st.write("Full vectors (raw)")
                st.text_area("RPU cashflow vector", value=json.dumps(rpu_vec, default=str), height=120)
                st.text_area("FP incremental cashflow vector", value=json.dumps(fp_vec, default=str), height=120)

                st.write("Vector lengths")
                st.json({"rpu_len": len(rpu_vec), "fp_incremental_len": len(fp_vec)})

                st.write("Download full arrays")
                st.download_button(
                    "Download IRR vectors JSON",
                    data=json.dumps({"rpu_cf": rpu_vec, "fp_incremental_cf": fp_vec}, default=str, indent=2),
                    file_name="irr_vectors.json",
                )

                rpu_rows = _bucket_mapping_rows(debug_data.get("rpu_breakdown", []), outputs.rcd, 1)
                fp_rows = _bucket_mapping_rows(
                    debug_data.get("fp_breakdown", []), outputs.rcd, debug_data.get("decision_bucket", 1)
                )
                st.write("Mapping summary – RPU buckets")
                st.dataframe(rpu_rows, use_container_width=True)
                st.write("Mapping summary – FP incremental buckets")
                st.dataframe(fp_rows, use_container_width=True)

                # Expected comparison (only when matching the specified scenario)
                is_target_case = (
                    outputs.rcd == date(2023, 3, 31)
                    and outputs.ptd == date(2026, 3, 31)
                    and abs((surrender_value or 0.0) - 1610611.0) < 0.5
                )
                if is_target_case:
                    expected_rpu_cf, expected_fp_incr_cf = _expected_bi2_vectors()
                    cmp_rpu = _compare_lists(rpu_vec, expected_rpu_cf, 1.0)
                    cmp_fp = _compare_lists(fp_vec, expected_fp_incr_cf, 1.0)
                    irr_rpu_ok = (
                        debug_data.get("rpu_irr") is not None
                        and abs(debug_data.get("rpu_irr") - 0.04618336) <= 0.0005
                    )
                    irr_fp_ok = (
                        debug_data.get("fp_incremental_irr") is not None
                        and abs(debug_data.get("fp_incremental_irr") - 0.06558466) <= 0.0005
                    )

                    st.write("Comparison vs expected (GIS BI 2 reference)")
                    st.json(
                        {
                            "rpu_vector_match": cmp_rpu,
                            "fp_incremental_vector_match": cmp_fp,
                            "rpu_irr_match": irr_rpu_ok,
                            "fp_incremental_irr_match": irr_fp_ok,
                        }
                    )

                    # Explicit invariants
                    invariant_timing = all(b == py + 1 for py, b in debug_data.get("fp_income_bucket_map", []))
                    bucket4 = debug_data.get("decision_bucket", 1)
                    fp_breakdown = debug_data.get("fp_breakdown", [])
                    bucket4_row = fp_breakdown[bucket4] if bucket4 < len(fp_breakdown) else {}
                    invariant_bucket4 = (
                        abs(bucket4_row.get("premium", 0.0) + 1500000.0) < 1e-3
                        and abs(bucket4_row.get("surrender", 0.0) + 1610611.0) < 1e-3
                        and abs(bucket4_row.get("income", 0.0)) < 1e-3
                    )
                    maturity_bucket = (extracted.policy_term_years or 0) + 1
                    maturity_ok = False
                    if maturity_bucket < len(fp_breakdown):
                        maturity_ok = abs(fp_breakdown[maturity_bucket].get("maturity", 0.0) - 18000000.0) < 1e-3

                    st.write("Invariants (PASS/FAIL)")
                    st.json(
                        {
                            "timing_shift_income_in_bucket_t_plus_1": invariant_timing,
                            "bucket4_contains_sv_plus_premium_only": invariant_bucket4,
                            "maturity_in_bucket_37": maturity_ok,
                        }
                    )

        # ---------- PDF download ----------
        st.divider()
        st.subheader("Download PDF")
        pdf_bytes = render_pdf_from_html(html_out)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"rpu_summary_{handler.product_id}.pdf",
            mime="application/pdf",
        )

    except ProductNotConfigured as e:
        st.error(str(e))
        log_event("product_not_configured", session_id, {"error": str(e)})
    except Exception as e:
        st.error(f"Failed: {e}")
        log_event("error", session_id, {"error": str(e)})


def _run_batch_mode(session_id: str) -> None:
    max_batch_mb = int(os.getenv("MAX_BATCH_MB", "50"))
    max_batch_seconds = int(os.getenv("MAX_BATCH_SECONDS", "480"))
    st.write("Upload a ZIP containing PDFs and one CSV input master.")
    uploaded = st.file_uploader("Upload batch ZIP", type=["zip"], key="batch_zip")
    start = st.button("Start batch processing", type="primary")
    cancel = st.button("Cancel batch", type="secondary")

    if cancel:
        st.session_state["batch_cancel"] = True

    if not start:
        return

    if uploaded is None:
        st.error("Please upload a ZIP file.")
        return

    batch_id = str(uuid.uuid4())
    st.session_state["batch_cancel"] = False
    log_event("batch_upload_received", session_id, {"batch_id": batch_id, "size_bytes": uploaded.size})

    if uploaded.size > max_batch_mb * 1024 * 1024:
        st.error(f"ZIP exceeds max size of {max_batch_mb} MB.")
        log_event("batch_upload_failed", session_id, {"batch_id": batch_id, "reason": "size_limit"})
        return

    progress = st.progress(0)
    log_box = st.empty()
    messages: list[str] = []

    def log_msg(msg: str) -> None:
        messages.append(msg)
        log_box.write(messages[-5:])

    start_time = time.time()
    summary_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    outputs: dict[str, bytes] = {}

    try:
        zip_bytes = uploaded.getvalue()
        log_event("batch_unzip_started", session_id, {"batch_id": batch_id})
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            pdf_files = [m for m in members if m.lower().endswith(".pdf")]
            csv_files = [m for m in members if m.lower().endswith(".csv")]
            log_event("batch_unzip_success", session_id, {"batch_id": batch_id, "pdf_count": len(pdf_files)})

            csv_mapping: dict[str, dict[str, str]] = {}
            csv_error_code: Optional[str] = None
            csv_error_message: str = ""
            if len(csv_files) != 1:
                if len(csv_files) == 0:
                    csv_error_code = "CSV_NOT_FOUND"
                    csv_error_message = "CSV input master not found"
                else:
                    csv_error_code = "CSV_MULTIPLE"
                    csv_error_message = "Multiple CSV files found in ZIP"
            else:
                try:
                    csv_mapping = _read_batch_csv(zf.read(csv_files[0]))
                    log_event("batch_csv_parsed", session_id, {"batch_id": batch_id, "rows": len(csv_mapping)})
                except Exception as exc:
                    csv_error_code = "CSV_PARSE_ERROR"
                    csv_error_message = str(exc)

            total = len(pdf_files)
            for idx, pdf_name in enumerate(pdf_files, start=1):
                if st.session_state.get("batch_cancel"):
                    log_msg("Cancel requested; returning partial results.")
                    break
                if time.time() - start_time > max_batch_seconds:
                    log_msg("Batch time limit reached; returning partial results.")
                    break

                progress.progress(idx / max(1, total), text=f"Processing {idx}/{total}")
                log_msg(f"Processing {pdf_name}")

                summary: dict[str, Any] = {
                    "input_pdf": pdf_name,
                    "derived_policy_number": "",
                    "csv_policy_number_match": "N",
                    "customer_name": "",
                    "product_name": "",
                    "ptd": "",
                    "surrender_value": "",
                    "status": "FAILED",
                    "error_code": "",
                    "error_stage": "",
                    "rpu_factor": "",
                    "rpu_irr": "",
                    "fp_incremental_irr": "",
                    "total_premium_payable_over_ppt": "",
                    "total_premium_paid_till_ptd": "",
                    "total_income_fp": "",
                    "total_income_rpu": "",
                    "maturity_fp": "",
                    "maturity_rpu": "",
                }

                derived_policy = _policy_number_from_filename(pdf_name)
                if not derived_policy:
                    summary.update(
                        {
                            "error_code": "FILENAME_POLICY_NUMBER_NOT_FOUND",
                            "error_stage": "filename_parse",
                        }
                    )
                    summary_rows.append(summary)
                    error_rows.append(
                        {
                            "input_pdf": pdf_name,
                            "derived_policy_number": "",
                            "error_code": summary["error_code"],
                            "error_stage": summary["error_stage"],
                            "short_error_message": "No digit sequence found in filename",
                        }
                    )
                    log_event(
                        "filename_policy_parse_fail",
                        session_id,
                        {"batch_id": batch_id, "input_pdf": pdf_name, "error": "not_found"},
                    )
                    continue

                summary["derived_policy_number"] = derived_policy
                log_event(
                    "filename_policy_parsed",
                    session_id,
                    {"batch_id": batch_id, "input_pdf": pdf_name, "policy_number": derived_policy},
                )

                if csv_error_code:
                    summary.update(
                        {
                            "error_code": csv_error_code,
                            "error_stage": "csv_parse",
                        }
                    )
                    summary_rows.append(summary)
                    error_rows.append(
                        {
                            "input_pdf": pdf_name,
                            "derived_policy_number": derived_policy,
                            "error_code": csv_error_code,
                            "error_stage": "csv_parse",
                            "short_error_message": csv_error_message or "CSV input master invalid or missing",
                        }
                    )
                    continue

                if derived_policy not in csv_mapping:
                    summary.update(
                        {
                            "error_code": "POLICY_NOT_IN_CSV",
                            "error_stage": "join",
                        }
                    )
                    summary_rows.append(summary)
                    error_rows.append(
                        {
                            "input_pdf": pdf_name,
                            "derived_policy_number": derived_policy,
                            "error_code": "POLICY_NOT_IN_CSV",
                            "error_stage": "join",
                            "short_error_message": "Policy number not found in CSV",
                        }
                    )
                    log_event(
                        "join_fail",
                        session_id,
                        {"batch_id": batch_id, "input_pdf": pdf_name, "policy_number": derived_policy},
                    )
                    continue

                summary["csv_policy_number_match"] = "Y"
                csv_row = csv_mapping[derived_policy]
                summary["ptd"] = csv_row.get("ptd", "")
                summary["surrender_value"] = csv_row.get("surrender_value", "")
                log_event(
                    "join_success",
                    session_id,
                    {"batch_id": batch_id, "input_pdf": pdf_name, "policy_number": derived_policy},
                )

                try:
                    ptd_value = _parse_ptd(csv_row.get("ptd", ""))
                    surrender_value = _parse_surrender_value(csv_row.get("surrender_value", ""))
                except Exception as exc:
                    summary.update(
                        {
                            "error_code": "CSV_VALUE_INVALID",
                            "error_stage": "compute",
                        }
                    )
                    summary_rows.append(summary)
                    error_rows.append(
                        {
                            "input_pdf": pdf_name,
                            "derived_policy_number": derived_policy,
                            "error_code": "CSV_VALUE_INVALID",
                            "error_stage": "compute",
                            "short_error_message": str(exc),
                        }
                    )
                    log_event(
                        "compute_fail",
                        session_id,
                        {"batch_id": batch_id, "policy_number": derived_policy, "error": str(exc)},
                    )
                    continue

                try:
                    pdf_bytes = zf.read(pdf_name)
                    parsed = read_pdf(pdf_bytes)
                    log_event(
                        "pdf_parse_success",
                        session_id,
                        {"batch_id": batch_id, "policy_number": derived_policy},
                    )
                except Exception as exc:
                    summary.update(
                        {
                            "error_code": "PDF_PARSE_FAIL",
                            "error_stage": "pdf_parse",
                        }
                    )
                    summary_rows.append(summary)
                    error_rows.append(
                        {
                            "input_pdf": pdf_name,
                            "derived_policy_number": derived_policy,
                            "error_code": "PDF_PARSE_FAIL",
                            "error_stage": "pdf_parse",
                            "short_error_message": str(exc),
                        }
                    )
                    log_event(
                        "pdf_parse_fail",
                        session_id,
                        {"batch_id": batch_id, "policy_number": derived_policy, "error": str(exc)},
                    )
                    continue

                try:
                    handler, conf, dbg = detect_product(parsed)
                    extracted = handler.extract(parsed)
                    outputs_obj = handler.calculate(extracted, ptd_value)
                    attach_irrs(extracted, outputs_obj, surrender_value)
                    log_event(
                        "compute_success",
                        session_id,
                        {"batch_id": batch_id, "policy_number": derived_policy, "product_id": handler.product_id},
                    )
                except Exception as exc:
                    summary.update(
                        {
                            "error_code": "COMPUTE_FAIL",
                            "error_stage": "compute",
                        }
                    )
                    summary_rows.append(summary)
                    error_rows.append(
                        {
                            "input_pdf": pdf_name,
                            "derived_policy_number": derived_policy,
                            "error_code": "COMPUTE_FAIL",
                            "error_stage": "compute",
                            "short_error_message": str(exc),
                        }
                    )
                    log_event(
                        "compute_fail",
                        session_id,
                        {"batch_id": batch_id, "policy_number": derived_policy, "error": str(exc)},
                    )
                    continue

                try:
                    html_out = render_gis_renewal_html(extracted, outputs_obj, surrender_value)
                    pdf_out = render_pdf_from_html(html_out)
                    output_name = f"outputs/{derived_policy}__{handler.product_id}__RPU.pdf"
                    outputs[output_name] = pdf_out
                    log_event(
                        "render_success",
                        session_id,
                        {"batch_id": batch_id, "policy_number": derived_policy},
                    )
                except Exception as exc:
                    summary.update(
                        {
                            "error_code": "RENDER_FAIL",
                            "error_stage": "render",
                        }
                    )
                    summary_rows.append(summary)
                    error_rows.append(
                        {
                            "input_pdf": pdf_name,
                            "derived_policy_number": derived_policy,
                            "error_code": "RENDER_FAIL",
                            "error_stage": "render",
                            "short_error_message": str(exc),
                        }
                    )
                    log_event(
                        "render_fail",
                        session_id,
                        {"batch_id": batch_id, "policy_number": derived_policy, "error": str(exc)},
                    )
                    continue

                premium_per_payment = extracted.annualized_premium_excl_tax or 0.0
                total_premium_payable = premium_per_payment * (outputs_obj.months_payable_total or 0)
                total_premium_paid = premium_per_payment * (outputs_obj.months_paid or 0)

                summary.update(
                    {
                        "status": "SUCCESS",
                        "error_code": "",
                        "error_stage": "",
                        "customer_name": extracted.proposer_name_transient or "",
                        "product_name": extracted.product_name or "",
                        "rpu_factor": outputs_obj.rpu_factor,
                        "rpu_irr": outputs_obj.irr_rpu,
                        "fp_incremental_irr": outputs_obj.irr_fp_incremental,
                        "total_premium_payable_over_ppt": total_premium_payable,
                        "total_premium_paid_till_ptd": total_premium_paid,
                        "total_income_fp": (outputs_obj.fully_paid or {}).get("total_income"),
                        "total_income_rpu": (outputs_obj.reduced_paid_up or {}).get("income_payable_after_rpu"),
                        "maturity_fp": (outputs_obj.fully_paid or {}).get("maturity"),
                        "maturity_rpu": (outputs_obj.reduced_paid_up or {}).get("maturity"),
                    }
                )
                summary_rows.append(summary)

                del pdf_bytes, parsed, extracted, outputs_obj
                gc.collect()

            log_event(
                "batch_complete",
                session_id,
                {"batch_id": batch_id, "total": len(pdf_files), "success": sum(1 for r in summary_rows if r["status"] == "SUCCESS")},
            )

    except zipfile.BadZipFile as exc:
        st.error(f"Invalid ZIP file: {exc}")
        log_event("batch_unzip_fail", session_id, {"batch_id": batch_id, "error": str(exc)})
        return

    fieldnames = [
        "input_pdf",
        "derived_policy_number",
        "csv_policy_number_match",
        "customer_name",
        "product_name",
        "ptd",
        "surrender_value",
        "status",
        "error_code",
        "error_stage",
        "rpu_factor",
        "rpu_irr",
        "fp_incremental_irr",
        "total_premium_payable_over_ppt",
        "total_premium_paid_till_ptd",
        "total_income_fp",
        "total_income_rpu",
        "maturity_fp",
        "maturity_rpu",
    ]
    summary_csv = _write_csv_bytes(summary_rows, fieldnames)
    errors_csv = _write_csv_bytes(
        error_rows, ["input_pdf", "derived_policy_number", "error_code", "error_stage", "short_error_message"]
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as out_zip:
        for name, payload in outputs.items():
            out_zip.writestr(name, payload)
        out_zip.writestr("batch_summary.csv", summary_csv)
        out_zip.writestr("errors.csv", errors_csv)

    st.success("Batch processing complete.")
    st.download_button(
        "Download batch results ZIP",
        data=zip_buffer.getvalue(),
        file_name=f"batch_results_{batch_id}.zip",
        mime="application/zip",
    )

def main():
    st.set_page_config(page_title="RPU Calculator", layout="centered")

    init_db()

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = hashlib.sha256(str(st.session_state).encode("utf-8")).hexdigest()
        log_event("session_start", st.session_state["session_id"], {"version": "m1", "device": "unknown"})

    session_id = st.session_state["session_id"]

    st.title("Reduced Paid-Up Calculator (Internal)")
    st.caption("Upload a Benefit Illustration PDF and enter PTD (Next Premium Due Date). No PDFs are stored.")
    tab_single, tab_batch = st.tabs(["Single", "Batch"])
    with tab_single:
        _run_single_mode(session_id)
    with tab_batch:
        _run_batch_mode(session_id)


if __name__ == "__main__":
    main()
