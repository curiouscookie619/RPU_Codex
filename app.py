from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Dict, Optional

import streamlit as st

from core.db import init_db, get_conn
from core.event_logger import log_event
from core.pdf_reader import read_pdf
from products.registry import detect_product
from core.output_pdf import render_one_pager


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
    with get_conn() as conn, conn.cursor() as cur:
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
        debug = st.checkbox("Debug mode (show what was extracted)", value=True)
        uploaded = st.file_uploader("Upload BI PDF", type=["pdf"])
        ptd = st.date_input("PTD (Next Premium Due Date)", value=None, format="DD/MM/YYYY")
        submitted = st.form_submit_button("Generate")

    if not submitted:
        return

    if uploaded is None or ptd is None:
        st.error("Please upload a PDF and enter PTD.")
        return

    file_bytes = uploaded.getvalue()
    file_hash = _sha256_bytes(file_bytes)

    log_event("pdf_uploaded", session_id, {"file_hash": file_hash, "size_bytes": len(file_bytes), "device": "unknown"})

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
        st.subheader("Fully Paid vs Reduced Paid-Up (Summary)")

        fully = outputs.fully_paid or {}
        rpu = outputs.reduced_paid_up or {}

        # Premium
        st.markdown("**Premium (from BI Premium Summary)**")
        st.write(f"- Instalment Premium without GST: ₹{_fmt_money(fully.get('instalment_premium_without_gst'))}")

        st.divider()

        # Fully Paid (stacked - responsive)
        st.markdown("### Fully Paid (as per BI)")
        _render_income_segments_bullets(fully.get("income_segments") or [], "Income pay-outs")
        st.write(f"- Total Income (sum): ₹{_fmt_money(fully.get('total_income'))}")
        st.write(f"- Maturity / Lump Sum: ₹{_fmt_money(fully.get('maturity'))}")
        st.write(f"- Death Benefit (schedule last year): ₹{_fmt_money(fully.get('death_last_year'))}")

        st.divider()

        st.markdown("### Reduced Paid-Up (Assuming non-payment after PTD + grace)")
        st.write(f"- RPU factor (R = Pp/Pt): **{rpu.get('rpu_factor')}**")

        # Remaining schedule (full-pay amounts)
        remaining_items = rpu.get("income_items_remaining_full") or []
        remaining_segments = _segments_from_income_items(remaining_items)
        _render_income_segments_bullets(remaining_segments, "Remaining income schedule (as per BI)")

        st.write(f"- Total Income over term (It): ₹{_fmt_money(rpu.get('income_total_full'))}")
        st.write(f"- Income already paid till RPU date (Ia): ₹{_fmt_money(rpu.get('income_already_paid'))}")
        st.write(f"- Income due after RPU date (full-pay reference): ₹{_fmt_money(rpu.get('income_due_full'))}")
        st.write(f"- **Net Income payable after RPU (SL formula): ₹{_fmt_money(rpu.get('income_payable_after_rpu'))}**")

        st.write(f"- Maturity (scaled): ₹{_fmt_money(rpu.get('maturity'))}")
        st.write(f"- Death Benefit (scaled): ₹{_fmt_money(rpu.get('death_scaled'))}")

        # Notes
        if outputs.notes:
            st.divider()
            st.subheader("Notes")
            for n in outputs.notes:
                st.write(f"- {n}")

        # ---------- PDF download ----------
        st.divider()
        st.subheader("Download one-pager (neutral)")
        pdf_bytes = render_one_pager(
            customer_name=(extracted.proposer_name_transient or "Customer"),
            product_name=extracted.product_name,
            summary={
                "Mode": extracted.mode,
                "PT": extracted.policy_term_years,
                "PPT": extracted.ppt_years,
                "BI Date": str(extracted.bi_generation_date),
                "RCD": str(outputs.rcd),
                "PTD": str(ptd),
                "Assumed RPU Date (PTD + Grace)": str(outputs.rpu_date),
            },
            fully_paid=fully,
            rpu=rpu,
            notes=outputs.notes or [],
        )
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"rpu_summary_{handler.product_id}.pdf",
            mime="application/pdf",
        )

    except Exception as e:
        st.error(f"Failed: {e}")
        log_event("error", session_id, {"error": str(e)})


if __name__ == "__main__":
    main()