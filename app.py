from __future__ import annotations

import hashlib
import json
from datetime import date
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
            segs.append({"start_year": cur_start, "end_year": cur_end, "amount": cur_amt, "years": (cur_end-cur_start+1)})
            cur_start, cur_end, cur_amt = y, y, a
    segs.append({"start_year": cur_start, "end_year": cur_end, "amount": cur_amt, "years": (cur_end-cur_start+1)})
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

def main():
    st.set_page_config(page_title="RPU Calculator", layout="centered")

    init_db()

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = hashlib.sha256(str(st.session_state).encode("utf-8")).hexdigest()
        log_event("session_start", st.session_state["session_id"], {"version": "m1", "device": "unknown"})

    session_id = st.session_state["session_id"]

    st.title("Reduced Paid-Up Calculator (Internal)")
    st.caption("Upload a Benefit Illustration PDF and enter PTD (Next Premium Due Date). No PDFs are stored.")

    # Use a form to avoid rerun/button non-responsiveness
    with st.form("main_form"):
        debug = st.checkbox("Debug mode (show what was extracted)", value=False)
        uploaded = st.file_uploader("Upload BI PDF", type=["pdf"])
        ptd = st.date_input("PTD (Next Premium Due Date)", value=None, format="DD/MM/YYYY")
        surrender_value_raw = st.text_input("Surrender Value (₹)", value="", placeholder="Required for IRR", help="Enter surrender value to compute IRRs; leave blank to skip IRR.")
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


if __name__ == "__main__":
    main()
