from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from core.date_logic import MODE_MONTHS, derive_rcd_and_rpu_dates
from core.models import ComputedOutputs, ExtractedFields, ParsedPDF
from core.pdf_reader import extract_bi_generation_date
from products.base import ProductHandler


def _clean(s: Any) -> str:
    return " ".join(str(s or "").replace("\n", " ").split()).strip()


def _sanitize_field(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = _clean(raw)
    low = s.lower()
    markers = [
        "additional",
        "plan information",
        "product information",
        "policy option",
        "option:",
        "remarks",
    ]
    cut = len(s)
    for m in markers:
        idx = low.find(m)
        if idx > 0:
            cut = min(cut, idx)
    s = s[:cut].strip(" -:|,") if cut < len(s) else s
    return s or None


def _sanitize_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = _clean(raw)
    low = s.lower()
    markers = [
        "name of the product",
        "product:",
        "plan:",
        "uin",
        "policy term",
        "premium payment term",
    ]
    cut = len(s)
    for m in markers:
        idx = low.find(m)
        if idx > 0:
            cut = min(cut, idx)
    s = s[:cut].strip(" -:|,") if cut < len(s) else s
    return s or None


def _to_number(text: Any) -> Optional[float]:
    s = _clean(text)
    if not s or s in {"-", "—"}:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _last_non_null(vals: List[Optional[float]]) -> Optional[float]:
    for v in reversed(vals or []):
        if v is not None:
            return v
    return None


def _parse_first_page_fields(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def grab(pattern: str) -> Optional[str]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return _clean(m.group(1)) if m else None

    out["product_name"] = _sanitize_field(grab(r"Name of the Product:\s*([^\n]+)"))
    out["proposer"] = _sanitize_name(grab(r"Name of the Prospect/Policyholder\s*:\s*([^\n]+)"))
    out["life_assured"] = grab(r"Name of the Life Assured\s*:\s*([^\n]+)")
    out["mode"] = _sanitize_field(grab(r"Mode of Payment of Premium\s*:\s*([A-Za-z\- ]+)"))
    out["policy_term"] = grab(r"PolicyTerm\s*\(in years\)\s*:\s*([0-9]+)")
    out["ppt"] = grab(r"Premium PaymentTerm\s*\(in years\)\s*:\s*([0-9]+)")
    out["income_start_year"] = _sanitize_field(grab(r"Income Start Year\s*:\s*([0-9PpTt\+]+)"))
    out["plan_option"] = _sanitize_field(grab(r"Policy Option\s*[,:\-]*\s*([^\n]+)"))
    out["instalment_wo_gst"] = grab(r"Instalment Premium without GST\s*([0-9,]+)")
    out["sam"] = grab(r"Sum Assured on Maturity\s*Rs\.??\s*([0-9,]+)")
    out["sad"] = grab(r"Sum Assured on Death.*?Rs\.??\s*([0-9,]+)")
    out["accrual_sb"] = _sanitize_field(grab(r"Accrual of Survival Benefit[s]?\s*:\s*([^\n]+)"))
    return out


def _parse_schedule_from_text(text_by_page: List[str], policy_term_years: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    header_seen = False
    for page_txt in text_by_page or []:
        for ln in (page_txt or "").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            tokens = re.findall(r"[0-9,]+|\\-", ln)
            if not header_seen and tokens == [str(i) for i in range(1, 25)]:
                header_seen = True
                continue
            if not header_seen:
                continue
            if len(tokens) != 24 or not tokens[0].isdigit():
                continue
            nums = [_to_number(t) for t in tokens]
            py_val = int(nums[0]) if nums[0] is not None else None
            if policy_term_years is not None and py_val is not None and py_val > policy_term_years:
                continue

            rows.append(
                {
                    "policy_year": py_val,
                    "annual_premium": nums[2],
                    "gi": nums[5],  # Guaranteed income / survival benefit component
                    "rb_payout_8": nums[13],
                    "cb_8": nums[14],
                    "sb_total_8": nums[17],
                    "maturity_8": nums[21],
                    "death_8": nums[23],
                }
            )
    return rows


def _looks_like_column_index_row(row: List[Any]) -> bool:
    """
    Detect numeric column-index rows (e.g., '1 2 3 4 5 ... 24') that appear immediately
    below table headers. These should be skipped when building schedule rows.
    """
    clean = [_clean(c) for c in row if c is not None]
    if not clean:
        return False
    if not all(item.isdigit() for item in clean):
        return False
    nums = [int(item) for item in clean]
    return nums == list(range(1, len(nums) + 1))


def _parse_schedule_from_tables(parsed: ParsedPDF, policy_term_years: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Optional[bool]]:
    """
    Parse schedule from tables (preferred for FSP BIs).
    Returns (rows, accrual_flag) where accrual_flag is True if any accrual column shows a value.
    """
    accrual_flag = False
    aggregated_rows: Dict[int, Dict[str, Any]] = {}
    all_tables = parsed.tables_by_page or []
    for page_tables in all_tables:
        for tb in page_tables or []:
            if not tb:
                continue
            rows_with_py = []
            for r in tb:
                if not r or len(r) < 24:
                    continue
                py_text = _clean(r[0])
                if not py_text.isdigit():
                    continue
                if _looks_like_column_index_row(r):
                    continue
                rows_with_py.append(r)

            if not rows_with_py:
                continue

            header_flat = " ".join(_clean(c) for c in (tb[0] or []) if c)
            header_mentions_policy = "policy" in header_flat.lower()

            for r in rows_with_py:
                py = int(_clean(r[0]))
                if policy_term_years is not None and py > policy_term_years:
                    continue
                if py in aggregated_rows:
                    continue
                # Column positions based on BI table (0-indexed)
                income_8 = _to_number(r[17])  # Total Survival Benefit @8% (5+14+15)
                maturity_8 = _to_number(r[21])  # Total Maturity Benefit @8%
                death_8 = _to_number(r[23])  # Total Death Benefit @8%
                accrual_val = _clean(r[19]) if len(r) > 19 else ""
                if accrual_val and accrual_val not in {"-", "—", ""}:
                    accrual_flag = True
                # if header missing, still accept as continuation table
                aggregated_rows[py] = {
                    "policy_year": py,
                    "annual_premium": _to_number(r[2]),
                    "gi": None,  # not used directly for RPU in this option
                    "rb_payout_8": None,
                    "cb_8": None,
                    "sb_total_8": income_8,
                    "maturity_8": maturity_8,
                    "death_8": death_8,
                }

    rows_sorted = [aggregated_rows[k] for k in sorted(aggregated_rows.keys())]
    return rows_sorted, accrual_flag


def _safe_anniversary(d: date, years_to_add: int) -> date:
    y = d.year + int(years_to_add)
    m = d.month
    day = d.day
    while True:
        try:
            return date(y, m, day)
        except ValueError:
            if day > 28:
                day -= 1
                continue
            raise


class FSPHandler(ProductHandler):
    product_id = "FSP"

    def detect(self, parsed: ParsedPDF) -> Tuple[float, Dict[str, Any]]:
        t = "\n".join(parsed.text_by_page or []).lower()
        if "flexi-savings plan" in t:
            return 0.95, {"match": "contains 'flexi-savings plan'"}
        if "flexi income" in t:
            return 0.7, {"match": "contains 'flexi income'"}
        return 0.0, {"match": "no"}

    def extract(self, parsed: ParsedPDF) -> ExtractedFields:
        page1_text = (parsed.text_by_page or [""])[0]
        bi_date = extract_bi_generation_date(page1_text) or date.today()
        meta = _parse_first_page_fields(page1_text)

        policy_term_years = int(meta.get("policy_term")) if meta.get("policy_term") else None

        schedule_rows, accrual_flag = _parse_schedule_from_tables(parsed, policy_term_years=policy_term_years)
        if not schedule_rows:
            schedule_rows = _parse_schedule_from_text(parsed.text_by_page, policy_term_years=policy_term_years)
        if accrual_flag is None:
            accrual_flag = False

        accrual_text = (meta.get("accrual_sb", "") or "").strip().lower()
        if accrual_text.startswith("y"):
            accrual_survival_benefits = True
        elif accrual_text.startswith("n"):
            accrual_survival_benefits = False
        else:
            accrual_survival_benefits = bool(accrual_flag)

        return ExtractedFields(
            product_name=meta.get("product_name") or "Edelweiss Life- Flexi-Savings Plan",
            product_uin=None,
            bi_generation_date=bi_date,
            proposer_name_transient=meta.get("proposer"),
            life_assured_age=None,
            life_assured_gender=None,
            mode=(meta.get("mode") or "Annual").title(),
            policy_term_years=int(meta.get("policy_term") or 0),
            ppt_years=int(meta.get("ppt") or 0),
            annualized_premium_excl_tax=_to_number(meta.get("instalment_wo_gst")),
            income_start_point_text=meta.get("income_start_year"),
            income_duration_years=None,
            income_payout_frequency="Annual",
            income_payout_type="Level",
            sum_assured_on_death=_to_number(meta.get("sad")),
            accrual_survival_benefits=accrual_survival_benefits,
            schedule_rows=schedule_rows,
        )

    def calculate(self, extracted: ExtractedFields, ptd: date) -> ComputedOutputs:
        rcd, rpu_date, grace_days = derive_rcd_and_rpu_dates(
            bi_date=extracted.bi_generation_date,
            ptd=ptd,
            mode=extracted.mode,
        )

        mode_clean = extracted.mode or "Annual"
        interval_months = MODE_MONTHS.get(mode_clean, MODE_MONTHS.get(mode_clean.title(), 12))
        months_between_rcd_ptd = (ptd.year - rcd.year) * 12 + (ptd.month - rcd.month)
        if ptd.day < rcd.day:
            months_between_rcd_ptd -= 1
        months_between_rcd_ptd = max(0, months_between_rcd_ptd)
        payments_paid = months_between_rcd_ptd // max(1, interval_months)
        payments_total = int((extracted.ppt_years or 0) * (12 / max(1, interval_months)))

        R = (payments_paid / payments_total) if payments_total > 0 else 0.0
        R = max(0.0, min(1.0, R))

        # Blanket RPU blocks
        rpu_blocked = False
        rpu_blocked_reason = None
        cutoff = date(2024, 10, 1)
        if extracted.accrual_survival_benefits:
            rpu_blocked = True
            rpu_blocked_reason = "Accrual of Survival Benefits = Yes; RPU not available."
        elif (rcd < cutoff) and (payments_paid < (2 * max(1, 12 // max(1, interval_months)))):
            rpu_blocked = True
            rpu_blocked_reason = "Less than 2 policy-year premiums were paid before 01-Oct-2024; RPU not available."

        income_events: List[Dict[str, Any]] = []
        full_income_vals: List[Optional[float]] = []
        maturity_vals: List[Optional[float]] = []
        death_vals: List[Optional[float]] = []

        for r in extracted.schedule_rows or []:
            py = r.get("policy_year")
            if py is None:
                continue
            py_i = int(py)
            gi = _to_number(r.get("gi"))
            rb = _to_number(r.get("rb_payout_8"))
            cb = _to_number(r.get("cb_8"))
            sb_total_8 = _to_number(r.get("sb_total_8"))
            sb_full = sb_total_8
            full_income_vals.append(sb_full)
            maturity_vals.append(_to_number(r.get("maturity_8")))
            death_vals.append(_to_number(r.get("death_8")))

            payout_date = _safe_anniversary(rcd, py_i)

            income_events.append(
                {
                    "policy_year": py_i,
                    "calendar_year": rcd.year + py_i,
                    "payout_date": payout_date,
                    "gi": gi or 0.0,
                    "rb": rb or 0.0,
                    "cb": cb or 0.0,
                    "amount": float(sb_full) if sb_full is not None else 0.0,
                }
            )

        income_events.sort(key=lambda x: x["policy_year"])

        # Fully paid totals (using SB @ 8% column as instructed)
        total_income_full = sum(e["amount"] for e in income_events)
        maturity_full = _last_non_null(maturity_vals)
        death_full = _last_non_null(death_vals)

        # Buckets for paid / grace / future
        income_paid_full = [e for e in income_events if e["payout_date"] <= ptd]
        income_within_grace = [e for e in income_events if ptd < e["payout_date"] <= rpu_date]
        income_future = [e for e in income_events if e["payout_date"] > rpu_date]

        Ia = sum(e["amount"] for e in income_paid_full)
        income_due_full = sum(e["amount"] for e in income_future)

        income_items_rpu: List[Dict[str, Any]] = []
        future_payable_sum = 0.0
        income_items_remaining_full: List[Dict[str, Any]] = []

        for e in income_events:
            payout_date = e["payout_date"]
            if payout_date <= ptd:
                final_amt = e["amount"]
                bucket = "already_paid"
            elif payout_date <= rpu_date:
                final_amt = e["amount"]
                bucket = "within_grace_full"
            else:
                if rpu_blocked:
                    final_amt = 0.0
                else:
                    # Scale SB @ 8% (includes guaranteed + bonuses) by RPU factor
                    final_amt = e["amount"] * R
                bucket = "future_rpu"
                future_payable_sum += final_amt
                income_items_remaining_full.append(
                    {
                        "policy_year": e["policy_year"],
                        "calendar_year": e["calendar_year"],
                        "payout_date": payout_date,
                        "amount": e["amount"],
                    }
                )

            income_items_rpu.append(
                {
                    "policy_year": e["policy_year"],
                    "calendar_year": e["calendar_year"],
                    "payout_date": payout_date,
                    "amount": round(final_amt, 2),
                    "original_amount": e["amount"],
                    "bucket": bucket,
                }
            )

        rpu_income_total = round(future_payable_sum, 2)
        if rpu_blocked:
            rpu_income_total = 0.0
            maturity_full = None

        fully_paid = {
            "instalment_premium_without_gst": extracted.annualized_premium_excl_tax,
            "total_income": float(total_income_full),
            "income_segments": [],
            "income_items": income_events,
            "maturity": float(maturity_full) if maturity_full is not None else None,
            "death_last_year": float(death_full) if death_full is not None else None,
        }

        reduced_paid_up = {
            "rpu_factor": round(R, 6),
            "income_total_full": float(total_income_full),
            "income_already_paid": float(Ia),
            "income_due_full": float(income_due_full),
            "income_payable_after_rpu": float(rpu_income_total),
            "income_segments": [],
            "income_items": income_items_rpu,
            "income_items_remaining_full": income_items_remaining_full,
            "income_paid_full_count": len(income_paid_full),
            "income_within_grace_count": len(income_within_grace),
            "income_future_count": len(income_future),
            "excess_paid_income": 0.0,
            "deduction_per_remaining": 0.0,
            "maturity": (float(maturity_full) * R) if maturity_full is not None else None,
            "death_scaled": (float(death_full) * R) if death_full is not None else None,
            "debug": {
                "payments_paid": int(payments_paid),
                "payments_total": int(payments_total),
                "months_between_rcd_ptd": int(months_between_rcd_ptd),
                "interval_months": int(interval_months),
                "grace_days": grace_days,
            },
            "rpu_blocked": rpu_blocked,
            "rpu_blocked_reason": rpu_blocked_reason,
        }

        notes = [
            "Premiums are assumed at the start of the policy year; payouts are at the end of the policy year.",
            "Grace: 30 days (non-monthly), 15 days (monthly).",
            "RPU factor = months paid / months payable. Survival benefits post RPU scale guaranteed + cash bonus by the factor; accrued bonuses are taken as-is.",
            "Survival benefit amounts use SB @ 8% (includes guaranteed and non-guaranteed, per BI). Maturity and death values come from the BI (incl. terminal bonus), scaled by the RPU factor.",
        ]
        if rpu_blocked_reason:
            notes.append(rpu_blocked_reason)

        return ComputedOutputs(
            rcd=rcd,
            ptd=ptd,
            rpu_date=rpu_date,
            grace_period_days=grace_days,
            months_paid=int(payments_paid),
            months_payable_total=int(payments_total),
            rpu_factor=R,
            fully_paid=fully_paid,
            reduced_paid_up=reduced_paid_up,
            notes=notes,
        )
