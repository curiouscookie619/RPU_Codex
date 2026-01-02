from __future__ import annotations

import math
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional
from datetime import date

from core.date_logic import MODE_MONTHS
from core.models import ComputedOutputs, ExtractedFields


def _fmt_money(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.0f}"
    except Exception:
        return str(v)


def _fmt_percent(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v)*100:.1f}%"
    except Exception:
        return "—"


def _payments_per_year(mode: str) -> int:
    mode_clean = (mode or "Annual").strip()
    months = MODE_MONTHS.get(mode_clean, MODE_MONTHS.get(mode_clean.title(), 12))
    return max(1, 12 // max(1, months)) if months else 1


def _payments_label(count: Optional[int], mode: str) -> str:
    if count is None:
        return "—"
    per_year = _payments_per_year(mode)
    years = count / per_year if per_year else 0
    years_str = f"{years:.1f}".rstrip("0").rstrip(".")
    if mode.strip().lower() == "annual":
        return f"{int(count)} years"
    unit = {
        12: "months",
        4: "quarters",
        6: "half-years",
        2: "half-years",
        1: "years",
    }.get(per_year, "payments")
    return f"{int(count)} {unit} [{years_str} years]"


def _as_date(v: Any) -> Optional[date]:
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return date.fromisoformat(v)
        except Exception:
            return None
    return None


def _income_rows_from_segments(segments: List[Dict[str, Any]], fallback_total: str) -> str:
    """
    Convert income segments into HTML rows (two-line style when possible).
    """
    if not segments:
        return ""

    rows: List[str] = []
    # If continuous with few runs, show up to 2-3 lines; otherwise keep minimal.
    used = 0
    for seg in segments:
        kind = seg.get("kind")
        if kind == "continuous_constant":
            rows.append(
                f"<div class='row'><span>Income for {seg.get('count')} years</span>"
                f"<span class='amount'>₹ {_fmt_money(seg.get('amount'))} / year</span></div>"
            )
            used += 1
        elif kind == "continuous_varying":
            rows.append(
                f"<div class='row'><span>Income (step-up)</span>"
                f"<span class='amount'>₹ {_fmt_money(seg.get('start_amount'))} → {_fmt_money(seg.get('end_amount'))}</span></div>"
            )
            used += 1
        elif kind == "discrete_constant":
            rows.append(
                f"<div class='row'><span>Income ({seg.get('count')} instances)</span>"
                f"<span class='amount'>₹ {_fmt_money(seg.get('amount'))}</span></div>"
            )
            used += 1
        elif kind == "discrete_varying":
            rows.append(
                f"<div class='row'><span>Income (discrete)</span>"
                f"<span class='amount'>Varies</span></div>"
            )
            used += 1
        if used >= 3:
            break

    if not rows:
        rows.append(
            f"<div class='row'><span>Income</span><span class='amount'>₹ {fallback_total}</span></div>"
        )
    return "\n      ".join(rows)


def _segments_from_income_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build simplified segments from income items (calendar_year, amount).
    """
    clean: List[Dict[str, Any]] = []
    for it in items or []:
        yr = it.get("calendar_year")
        amt = it.get("amount")
        if yr is None or amt is None:
            continue
        try:
            clean.append({"year": int(yr), "amount": float(amt)})
        except Exception:
            continue
    clean.sort(key=lambda x: x["year"])
    if not clean:
        return []

    segments: List[List[Dict[str, Any]]] = [[clean[0]]]
    for e in clean[1:]:
        if abs(e["amount"] - segments[-1][-1]["amount"]) < 0.01 and e["year"] == segments[-1][-1]["year"] + 1:
            segments[-1].append(e)
        else:
            segments.append([e])

    out: List[Dict[str, Any]] = []
    for run in segments:
        amt = run[0]["amount"]
        if len(run) >= 1:
            out.append(
                {
                    "kind": "continuous_constant",
                    "amount": amt,
                    "count": len(run),
                    "start_year": run[0]["year"],
                    "end_year": run[-1]["year"],
                }
            )
    return out


def render_gis_renewal_html(
    extracted: ExtractedFields,
    computed: ComputedOutputs,
    surrender_value: Optional[float],
) -> str:
    tmpl_path = Path("templates/gis_renewal_v2.html")
    template = Template(tmpl_path.read_text(encoding="utf-8"))

    instalment = extracted.annualized_premium_excl_tax
    mode = extracted.mode or "Annual"
    payments_per_year = _payments_per_year(mode)

    payments_paid = computed.months_paid
    payments_total = computed.months_payable_total

    payments_remaining = (payments_total - payments_paid) if payments_total is not None else None
    fp_give_raw = (instalment or 0) * payments_remaining if instalment is not None else None
    rpu_give_raw = (instalment or 0) * payments_paid if instalment is not None else None

    ptd = computed.ptd
    rcd = computed.rcd

    # Eligibility rule: for RCD < 1 Oct 2024, require ≥2 years of premiums (mode-aware) for RPU payouts
    required_paid = 2 * payments_per_year
    cutoff = date(2024, 10, 1)
    rpu_ineligible = False
    try:
        rpu_ineligible = (rcd < cutoff) and (payments_paid < required_paid)
    except Exception:
        rpu_ineligible = False

    def _payout_date(it: Dict[str, Any]) -> Optional[date]:
        pd = _as_date(it.get("payout_date"))
        if pd:
            return pd
        cy = it.get("calendar_year")
        try:
            return date(int(cy), 12, 31)
        except Exception:
            return None

    fp_items_all = (computed.fully_paid or {}).get("income_items") or []
    fp_items_future = []
    for it in fp_items_all:
        pd = _payout_date(it) or rcd
        if pd > ptd:
            fp_items_future.append({**it, "payout_date": pd})

    fp_income_total = sum(it.get("amount", 0) or 0 for it in fp_items_future)
    fp_maturity = (computed.fully_paid or {}).get("maturity")
    fp_get_raw = (fp_income_total or 0) + (fp_maturity or 0 if fp_maturity is not None else 0)

    fp_death = (computed.fully_paid or {}).get("death_last_year")

    rpu_income_total = (computed.reduced_paid_up or {}).get("income_payable_after_rpu")
    rpu_maturity = (computed.reduced_paid_up or {}).get("maturity")
    rpu_get_raw = (rpu_income_total or 0) + (rpu_maturity or 0 if rpu_maturity is not None else 0)

    def _ratio(get_v: Optional[float], give_v: Optional[float]) -> str:
        if get_v is None or give_v in (None, 0):
            return "—"
        try:
            return f"{float(get_v) / float(give_v):.2f}×"
        except Exception:
            return "—"

    # Income segments
    fp_segments = _segments_from_income_items(fp_items_future)
    fp_income_rows = _income_rows_from_segments(fp_segments, _fmt_money(fp_income_total))

    rpu_items = (computed.reduced_paid_up or {}).get("income_items") or []
    rpu_future_items = [it for it in rpu_items if it.get("bucket") == "future_rpu"]
    rpu_segments = _segments_from_income_items(rpu_future_items)
    rpu_income_rows = _income_rows_from_segments(rpu_segments, _fmt_money(rpu_income_total))

    rpu_extra_class = ""
    rpu_note = (
        "You stop paying premiums after PTD (then grace). You still receive benefits, but future payouts reduce versus full benefits."
    )
    if rpu_ineligible:
        rpu_extra_class = "card-rpu-danger"
        rpu_income_total = 0
        rpu_maturity = 0
        rpu_get_raw = 0
        rpu_income_rows = "<div class='row'><span>Not eligible for RPU payouts (less than 2 years of premiums before 01-Oct-2024)</span><span class='amount'>—</span></div>"
        rpu_note = "Not eligible for RPU payouts because fewer than 2 policy-year premiums were paid before 01-Oct-2024. All RPU payouts are nil."

    # Surrender
    surrender_present = surrender_value is not None
    surrender_blocked = rpu_ineligible  # blanket rule: if RPU not possible, surrender not possible
    surrender_card_style = "" if surrender_present or surrender_blocked else "display:none;"
    surrender_pill_style = "" if surrender_present else "display:none;"
    surrender_diff = None

    if surrender_blocked:
        surrender_value_text = "Not available"
        surrender_diff = rpu_give_raw or 0
        surrender_note = "Surrender not available because fewer than 2 policy-year premiums were paid before 01-Oct-2024. Premiums paid are forfeited."
        surrender_extra_class = "card-surrender-blocked"
    else:
        surrender_value_text = surrender_value
        if surrender_present and rpu_give_raw is not None:
            surrender_diff = (rpu_give_raw or 0) - (surrender_value or 0)
        surrender_note = "Surrender ends the policy and future benefits stop. The value shown here depends on the surrender value you input."
        surrender_extra_class = ""

    premiums_paid_label = _payments_label(payments_paid, mode)

    subs: Dict[str, str] = {
        "customer_name": (extracted.proposer_name_transient or "Customer"),
        "plan_name": extracted.product_name,
        "policy_term": f"{extracted.policy_term_years} years",
        "ppt": f"{extracted.ppt_years} years",
        "mode": mode,
        "instalment_premium": _fmt_money(instalment),
        "premiums_paid_label": premiums_paid_label,
        "surrender_value": _fmt_money(surrender_value_text) if surrender_value_text is not None else "—",
        "surrender_card_style": surrender_card_style,
        "surrender_pill_style": surrender_pill_style,
        "surrender_diff": _fmt_money(surrender_diff) if surrender_diff is not None else "—",
        "surrender_note": surrender_note,
        "surrender_extra_class": surrender_extra_class,
        "rpu_give": _fmt_money(rpu_give_raw),
        "fp_give": _fmt_money(fp_give_raw),
        "fp_income_total": _fmt_money(fp_income_total),
        "fp_maturity": _fmt_money(fp_maturity),
        "fp_get": _fmt_money(fp_get_raw),
        "fp_ratio": _ratio(fp_get_raw, fp_give_raw),
        "fp_death": _fmt_money(fp_death),
        "rpu_income_rows": rpu_income_rows,
        "fp_income_rows": fp_income_rows,
        "rpu_income_total": _fmt_money(rpu_income_total),
        "rpu_maturity": _fmt_money(rpu_maturity),
        "rpu_get": _fmt_money(rpu_get_raw),
        "rpu_ratio": _ratio(rpu_get_raw, rpu_give_raw),
        "rpu_extra_class": rpu_extra_class,
        "rpu_note": rpu_note,
        "rpu_irr": _fmt_percent(getattr(computed, "irr_rpu", None)),
        "fp_irr": _fmt_percent(getattr(computed, "irr_fp_incremental", None)),
    }

    return template.safe_substitute(subs)
