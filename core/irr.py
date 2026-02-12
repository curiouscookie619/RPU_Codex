from __future__ import annotations

from datetime import date
from typing import Iterable, List, Optional

from core.date_logic import MODE_MONTHS

# -------------------------
# Utilities
# -------------------------


def add_months(d: date, months: int) -> date:
    """Safe add months (keeps end-of-month)."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = d.day
    while True:
        try:
            return date(y, m, day)
        except ValueError:
            day -= 1
            if day < 1:
                return date(y, m, 1)


def add_years(d: date, years: int) -> date:
    return add_months(d, years * 12)


# -------------------------
# Periodic IRR (annual buckets)
# -------------------------


def _npv_periodic(cashflows: List[float], rate: float) -> float:
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


def periodic_irr(cashflows: Iterable[float], guess: float = 0.05) -> Optional[float]:
    cfs = list(cashflows)
    if not cfs:
        return None
    has_pos = any(cf > 0 for cf in cfs)
    has_neg = any(cf < 0 for cf in cfs)
    if not (has_pos and has_neg):
        return None

    low, high = -0.99, 1.5
    f_low = _npv_periodic(cfs, low)
    f_high = _npv_periodic(cfs, high)
    for _ in range(200):
        mid = (low + high) / 2
        f_mid = _npv_periodic(cfs, mid)
        if abs(f_mid) < 1e-9 or (high - low) < 1e-7:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return mid


# -------------------------
# IRR attachment helpers
# -------------------------


def _payments_per_year(mode: str) -> int:
    mode_clean = (mode or "Annual").strip()
    months = MODE_MONTHS.get(mode_clean, MODE_MONTHS.get(mode_clean.title(), 12))
    return max(1, 12 // max(1, months)) if months else 1


def _py_for_payout(rcd: date, payout_date: date, pt_years: int) -> Optional[int]:
    for py in range(1, pt_years + 1):
        if add_years(rcd, py) == payout_date:
            return py
    # fallback approximation
    delta_years = (payout_date - rcd).days / 365.0
    py = round(delta_years)
    if 1 <= py <= pt_years:
        return int(py)
    return None


def _build_income_by_py(items, rcd: date, pt_years: int) -> dict:
    out = {}
    for it in items or []:
        py = it.get("policy_year")
        pd = it.get("payout_date")
        if py is None:
            if pd:
                py = _py_for_payout(rcd, pd, pt_years)
        if py is None:
            continue
        amt = it.get("amount") or 0.0
        out[int(py)] = float(amt)
    return out


def compute_irr_cashflows(extracted, outputs, surrender_value: float = 0.0):
    """
    Build cashflow vectors (annual buckets) used for IRR computation.
    Returns:
      - rpu_cfs: full-length list (bucket index = policy year)
      - fp_incremental_cfs: full-length list (bucket index = policy year)
      - rpu_breakdown / fp_breakdown: per-bucket components
      - decision_bucket: start bucket for FP incremental series
    """
    mode = extracted.mode or "Annual"
    per_year = _payments_per_year(mode)
    instalment = extracted.annualized_premium_excl_tax or 0.0
    premium_per_year = instalment * per_year

    payments_paid = outputs.months_paid or 0
    payments_total = outputs.months_payable_total or 0
    paid_years = int(payments_paid // per_year)

    pt_years = extracted.policy_term_years or 0
    ppt_years = extracted.ppt_years or 0

    rcd: date = outputs.rcd
    ptd: date = outputs.ptd
    rpu_date: date = outputs.rpu_date

    fp_items = (outputs.fully_paid or {}).get("income_items") or []
    rpu_items = (outputs.reduced_paid_up or {}).get("income_items") or []

    fp_by_py = _build_income_by_py(fp_items, rcd, pt_years)
    rpu_by_py = _build_income_by_py(rpu_items, rcd, pt_years)

    full_maturity = (outputs.fully_paid or {}).get("maturity") or 0.0
    rpu_maturity = (outputs.reduced_paid_up or {}).get("maturity") or 0.0

    # ---------- RPU ----------
    rpu_cfs = [0.0 for _ in range(pt_years + 2)]  # bucket index = policy year number
    rpu_breakdown = [
        {"premium": 0.0, "income": 0.0, "maturity": 0.0, "surrender": 0.0} for _ in range(pt_years + 2)
    ]
    rpu_income_bucket_map: List[tuple[int, int]] = []

    for y in range(1, paid_years + 1):
        rpu_cfs[y] += -premium_per_year
        rpu_breakdown[y]["premium"] += -premium_per_year

    for py, amt_full in fp_by_py.items():
        payout_bucket = py + 1
        payout_date = add_years(rcd, py)
        if payout_date == rpu_date:
            continue
        if payout_date <= ptd:
            amt = amt_full
        else:
            amt = rpu_by_py.get(py, 0.0)
        if payout_bucket < len(rpu_cfs):
            rpu_cfs[payout_bucket] += amt
            rpu_breakdown[payout_bucket]["income"] += amt
            rpu_income_bucket_map.append((py, payout_bucket))

    if pt_years + 1 < len(rpu_cfs):
        rpu_cfs[pt_years + 1] += rpu_maturity
        rpu_breakdown[pt_years + 1]["maturity"] += rpu_maturity

    # ---------- FP vs surrender ----------
    inc_cfs = [0.0 for _ in range(pt_years + 2)]
    fp_breakdown = [
        {"premium": 0.0, "income": 0.0, "maturity": 0.0, "surrender": 0.0} for _ in range(pt_years + 2)
    ]
    fp_income_bucket_map: List[tuple[int, int]] = []
    decision_bucket = paid_years + 1

    # t0 bucket: subtract SV and (if continuing) premium due now
    if decision_bucket < len(inc_cfs):
        inc_cfs[decision_bucket] += -float(surrender_value)
        inc_cfs[decision_bucket] += -premium_per_year  # premium due now if continuing
        fp_breakdown[decision_bucket]["surrender"] += -float(surrender_value)
        fp_breakdown[decision_bucket]["premium"] += -premium_per_year

    # Remaining premiums after decision bucket until end of PPT
    for y in range(decision_bucket + 1, ppt_years + 1):
        inc_cfs[y] += -premium_per_year
        fp_breakdown[y]["premium"] += -premium_per_year

    # Income: apply agreed timing shift => income for PY(py) is recognized in bucket (py+1)
    for py, amt_full in fp_by_py.items():
        payout_date = add_years(rcd, py)

        # If payout is on/before PTD, it is already received / locked-in and excluded from FP incremental series
        if payout_date <= ptd:
            continue

        payout_bucket = py + 1  # <-- FIXED (was py)

        if payout_bucket < len(inc_cfs):
            inc_cfs[payout_bucket] += amt_full
            fp_breakdown[payout_bucket]["income"] += amt_full
            fp_income_bucket_map.append((py, payout_bucket))

    # Maturity: recognized in last bucket (pt_years + 1) with the same end-of-year convention
    if pt_years + 1 < len(inc_cfs):
        maturity_date = add_years(rcd, pt_years)
        if maturity_date > ptd:
            inc_cfs[pt_years + 1] += full_maturity
            fp_breakdown[pt_years + 1]["maturity"] += full_maturity

    return {
        "rpu_cfs": rpu_cfs,
        "fp_incremental_cfs": inc_cfs,
        "decision_bucket": decision_bucket,
        "rpu_breakdown": rpu_breakdown,
        "fp_breakdown": fp_breakdown,
        "rpu_income_bucket_map": rpu_income_bucket_map,
        "fp_income_bucket_map": fp_income_bucket_map,
    }


def build_irr_debug(extracted, outputs, surrender_value: float = 0.0):
    """Return cashflow vectors, breakdowns, and IRRs (no side-effects)."""
    data = compute_irr_cashflows(extracted, outputs, surrender_value)
    rpu_cf = data["rpu_cfs"][1:]
    fp_cf = data["fp_incremental_cfs"][data["decision_bucket"] :]
    return {
        **data,
        "rpu_cf": rpu_cf,
        "fp_incremental_cf": fp_cf,
        "rpu_irr": periodic_irr(rpu_cf),
        "fp_incremental_irr": periodic_irr(fp_cf),
    }


def attach_irrs(extracted, outputs, surrender_value: float = 0.0) -> None:
    """
    Compute annual periodic IRRs for RPU (full policy) and FP continuation vs surrender.
    """
    data = compute_irr_cashflows(extracted, outputs, surrender_value)
    outputs.irr_rpu = periodic_irr(data["rpu_cfs"][1:])  # ignore bucket 0 (unused)
    outputs.irr_fp_incremental = periodic_irr(data["fp_incremental_cfs"][data["decision_bucket"] :])
