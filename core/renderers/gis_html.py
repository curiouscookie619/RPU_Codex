from __future__ import annotations

import math
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

from core.date_logic import MODE_MONTHS
from core.models import ComputedOutputs, ExtractedFields


def _fmt_money(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.0f}"
    except Exception:
        return str(v)


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

    fp_give_raw = (instalment or 0) * payments_total if instalment is not None else None
    rpu_give_raw = (instalment or 0) * payments_paid if instalment is not None else None

    fp_income_total = (computed.fully_paid or {}).get("total_income")
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
    fp_segments = (computed.fully_paid or {}).get("income_segments") or []
    if not fp_segments:
        fp_segments = _segments_from_income_items((computed.fully_paid or {}).get("income_items") or [])
    fp_income_rows = _income_rows_from_segments(fp_segments, _fmt_money(fp_income_total))

    rpu_items = (computed.reduced_paid_up or {}).get("income_items") or []
    rpu_future_items = [it for it in rpu_items if it.get("bucket") == "future_rpu"]
    rpu_segments = _segments_from_income_items(rpu_future_items)
    rpu_income_rows = _income_rows_from_segments(rpu_segments, _fmt_money(rpu_income_total))

    # FD numbers
    fd_principal = fp_give_raw
    fd_interest = fd_principal * 0.07 if fd_principal is not None else None
    fd_tax = fd_interest * 0.20 if fd_interest is not None else None
    fd_net = fd_interest - fd_tax if fd_interest is not None and fd_tax is not None else None

    # Surrender
    surrender_present = surrender_value is not None
    surrender_card_style = "" if surrender_present else "display:none;"
    surrender_pill_style = "" if surrender_present else "display:none;"
    surrender_diff = None
    if surrender_present and rpu_give_raw is not None:
        surrender_diff = (rpu_give_raw or 0) - (surrender_value or 0)

    premiums_paid_label = _payments_label(payments_paid, mode)

    subs: Dict[str, str] = {
        "customer_name": (extracted.proposer_name_transient or "Customer"),
        "plan_name": extracted.product_name,
        "policy_term": f"{extracted.policy_term_years} years",
        "ppt": f"{extracted.ppt_years} years",
        "mode": mode,
        "instalment_premium": _fmt_money(instalment),
        "premiums_paid_label": premiums_paid_label,
        "surrender_value": _fmt_money(surrender_value) if surrender_present else "—",
        "surrender_card_style": surrender_card_style,
        "surrender_pill_style": surrender_pill_style,
        "surrender_diff": _fmt_money(surrender_diff) if surrender_diff is not None else "—",
        "rpu_give": _fmt_money(rpu_give_raw),
        "fp_give": _fmt_money(fp_give_raw),
        "fp_income_total": _fmt_money(fp_income_total),
        "fp_maturity": _fmt_money(fp_maturity),
        "fp_get": _fmt_money(fp_get_raw),
        "fp_ratio": _ratio(fp_get_raw, fp_give_raw),
        "fp_death": _fmt_money(fp_death),
        "fd_principal": _fmt_money(fd_principal),
        "fd_interest": _fmt_money(fd_interest),
        "fd_tax": _fmt_money(fd_tax),
        "fd_net": _fmt_money(fd_net),
        "rpu_income_rows": rpu_income_rows,
        "fp_income_rows": fp_income_rows,
        "rpu_income_total": _fmt_money(rpu_income_total),
        "rpu_maturity": _fmt_money(rpu_maturity),
        "rpu_get": _fmt_money(rpu_get_raw),
        "rpu_ratio": _ratio(rpu_get_raw, rpu_give_raw),
    }

    return template.safe_substitute(subs)
