from __future__ import annotations
from datetime import date, timedelta

MODE_MONTHS = {
    "Annual": 12,
    "Half-Yearly": 6,
    "Half Yearly": 6,
    "Halfyearly": 6,
    "Quarterly": 3,
    "Monthly": 1,
}

def _is_leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)

def _days_in_month(y: int, m: int) -> int:
    if m == 2:
        return 29 if _is_leap(y) else 28
    if m in (4, 6, 9, 11):
        return 30
    return 31

def _subtract_months(d: date, months: int) -> date:
    y, m = d.year, d.month - months
    while m <= 0:
        m += 12
        y -= 1
    day = min(d.day, _days_in_month(y, m))
    return date(y, m, day)

def derive_rcd_and_rpu_dates(bi_date: date, ptd: date, mode: str):
    """
    RCD rule (per your instruction):
      - RCD is the smallest date in the {PTD, PTD-interval, PTD-2*interval, ...}
        such that RCD >= BI date.
    RPU effective date:
      - PTD + grace period
    Grace period:
      - monthly: 15 days
      - all other modes: 30 days
    """
    mode = (mode or "Annual").strip()
    months = MODE_MONTHS.get(mode, MODE_MONTHS.get(mode.title(), 12))
    grace_days = 15 if mode.lower() == "monthly" else 30

    candidate = ptd
    while True:
        prev = _subtract_months(candidate, months)
        if prev < bi_date:
            rcd = candidate
            break
        candidate = prev

    rpu_date = ptd + timedelta(days=grace_days)
    return rcd, rpu_date, grace_days
