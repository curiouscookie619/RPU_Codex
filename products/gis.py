from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from core.models import ParsedPDF, ExtractedFields, ComputedOutputs
from core.pdf_reader import extract_bi_generation_date
from core.date_logic import derive_rcd_and_rpu_dates
from products.base import ProductHandler


# -------------------------
# Helpers
# -------------------------

def _clean_text(s: Any) -> str:
    return " ".join(str(s or "").replace("\n", " ").split()).strip()


def _norm_key(s: Any) -> str:
    s = _clean_text(s)
    s = s.replace(" :", ":")
    if s.endswith(":"):
        s = s[:-1]
    return _clean_text(s).lower()


def _to_int(text: Any) -> Optional[int]:
    s = _clean_text(text)
    if not s:
        return None
    m = re.search(r"\d+", s.replace(",", ""))
    return int(m.group()) if m else None


def _to_number(text: Any) -> Optional[float]:
    s = _clean_text(text)
    if not s or s in {"-", "—"}:
        return None
    s = s.replace(",", "").replace("₹", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def _header_key(h: str) -> str:
    h = _clean_text(h).lower()
    if "policy year" in h:
        return "policy_year"
    if "income" in h or "survival" in h or "loyalty addition" in h:
        return "income"
    if "maturity" in h or "lump sum" in h or "lumpsum" in h:
        return "maturity"
    if "death" in h:
        return "death"
    return ""


def _flatten_tables(parsed: ParsedPDF) -> List[List[List[Optional[str]]]]:
    out: List[List[List[Optional[str]]]] = []
    for page_tables in (parsed.tables_by_page or []):
        for tb in (page_tables or []):
            if tb:
                out.append(tb)
    return out


def _join_text(parsed: ParsedPDF) -> str:
    return "\n".join(parsed.text_by_page or [])


def _find_value_in_tables(
    tables_by_page: List[List[List[List[Optional[str]]]]],
    row_contains: str,
) -> Optional[float]:
    """
    For multi-column tables (like Premium Summary), find a row that contains row_contains
    and return the LAST numeric cell in that row.
    """
    needle = row_contains.lower()
    for page_tables in (tables_by_page or []):
        for tb in (page_tables or []):
            for row in (tb or []):
                if not row:
                    continue
                row_text = " ".join(_clean_text(c).lower() for c in row if c is not None)
                if needle in row_text:
                    for c in reversed(row):
                        n = _to_number(c)
                        if n is not None:
                            return n
    return None


def _safe_anniversary(d: date, years_to_add: int) -> date:
    """
    Shift date by `years_to_add` years keeping month/day when possible.
    Clamps Feb 29 -> Feb 28 on non-leap years and handles month-end safely.
    """
    y = d.year + int(years_to_add)
    m = d.month
    day = d.day
    try:
        return date(y, m, day)
    except ValueError:
        if m == 2 and day == 29:
            return date(y, 2, 28)
        for dd in (31, 30, 29, 28):
            try:
                return date(y, m, dd)
            except ValueError:
                continue
        return date(y, m, 28)


def _income_segments(schedule_rows: List[Dict[str, Any]], rcd: date) -> List[Dict[str, Any]]:
    """
    Build human-readable income segments using calendar years (not policy years).

    Rules (Option 1):
    - Consecutive payout years with the same income are grouped as a continuous range.
    - Consecutive payout years with varying income are grouped as a continuous range (with start/end amounts).
    - Non-consecutive payouts:
        * if all amounts are the same -> one discrete segment with a year list
        * if amounts vary -> one discrete segment listing year: amount (truncated in UI; PDF will show full table)
    Returned segment dicts are meant for display only.
    """
    events: List[Dict[str, Any]] = []
    for r in (schedule_rows or []):
        py = r.get("policy_year")
        inc = r.get("income")
        if py is None:
            continue
        try:
            inc_f = float(inc) if inc is not None else 0.0
        except Exception:
            inc_f = 0.0
        if inc_f <= 0:
            continue
        cal_year = int(rcd.year + int(py) - 1)
        events.append({"policy_year": int(py), "calendar_year": cal_year, "income": inc_f})

    events.sort(key=lambda x: x["policy_year"])
    if not events:
        return []

    continuous = all(
        events[i]["policy_year"] == events[i - 1]["policy_year"] + 1 for i in range(1, len(events))
    )

    segments: List[Dict[str, Any]] = []

    if continuous:
        runs: List[List[Dict[str, Any]]] = [[events[0]]]
        for e in events[1:]:
            if abs(e["income"] - runs[-1][-1]["income"]) < 0.0001:
                runs[-1].append(e)
            else:
                runs.append([e])

        if len(runs) > 4:
            segments.append(
                {
                    "kind": "continuous_varying",
                    "start_amount": events[0]["income"],
                    "end_amount": events[-1]["income"],
                    "start_year": events[0]["calendar_year"],
                    "end_year": events[-1]["calendar_year"],
                    "count": len(events),
                }
            )
            return segments

        for run in runs:
            segments.append(
                {
                    "kind": "continuous_constant",
                    "amount": run[0]["income"],
                    "start_year": run[0]["calendar_year"],
                    "end_year": run[-1]["calendar_year"],
                    "count": len(run),
                }
            )
        return segments

    by_amount: Dict[float, List[int]] = {}
    items: List[Tuple[int, float]] = []
    for e in events:
        by_amount.setdefault(e["income"], []).append(e["calendar_year"])
        items.append((e["calendar_year"], e["income"]))

    if len(by_amount) == 1:
        amt = next(iter(by_amount.keys()))
        years = sorted(next(iter(by_amount.values())))
        segments.append(
            {"kind": "discrete_constant", "amount": amt, "years": years, "count": len(years)}
        )
        return segments

    used_years: set[int] = set()
    for amt, years in sorted(by_amount.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(years) >= 2:
            ys = sorted(years)
            segments.append({"kind": "discrete_constant", "amount": amt, "years": ys, "count": len(ys)})
            used_years.update(ys)

    varying = [(y, a) for (y, a) in items if y not in used_years]
    varying.sort(key=lambda x: x[0])
    if varying:
        segments.append(
            {
                "kind": "discrete_varying",
                "items": [{"year": y, "amount": a} for y, a in varying],
                "count": len(varying),
            }
        )

    return segments


def _last_non_null(schedule_rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    for r in reversed(schedule_rows or []):
        v = r.get(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
    return None


# -------------------------
# Handler
# -------------------------

class GISHandler(ProductHandler):
    product_id = "GIS"

    def detect(self, parsed: ParsedPDF) -> Tuple[float, Dict[str, Any]]:
        t = _join_text(parsed).lower()
        if "guaranteed income star" in t:
            return 0.95, {"match": "contains 'guaranteed income star'"}
        if "guaranteed income" in t and "star" in t:
            return 0.70, {"match": "contains 'guaranteed income' and 'star'"}
        return 0.0, {"match": "no"}

    def extract(self, parsed: ParsedPDF) -> ExtractedFields:
        page1 = (parsed.text_by_page or [""])[0]
        bi_date = extract_bi_generation_date(page1) or date.today()

        kv: Dict[str, str] = {}
        for tb in _flatten_tables(parsed):
            for row in (tb or []):
                if not row or len(row) < 2:
                    continue
                k = _norm_key(row[0])
                v = _clean_text(row[1])
                if k:
                    kv[k] = v

        product_name = kv.get("name of the product") or "Edelweiss Tokio Life- Guaranteed Income STAR"
        uin = kv.get("unique identification no.") or kv.get("uin")
        proposer = kv.get("name of the prospect/policyholder")

        mode = (kv.get("mode of payment of premium") or "Annual").title()

        pt = _to_int(kv.get("policy term (in years)") or kv.get("policy term")) or 0
        ppt = _to_int(kv.get("premium payment term (in years)") or kv.get("premium payment term")) or 0

        age = _to_int(kv.get("age (years)") or kv.get("age") or "")
        gender = (kv.get("gender of the life assured") or kv.get("gender") or "").title() or None

        instalment_premium_wo_gst = _find_value_in_tables(
            parsed.tables_by_page,
            "Instalment Premium without GST",
        )

        sum_assured = _to_number(
            kv.get("sum assured on death (at inception of the policy) rs.")
            or kv.get("sum assured on death (at inception of the policy) rs")
            or kv.get("sum assured on death (at inception of the policy)")
            or ""
        )

        schedule_rows = self._extract_schedule(parsed, pt)

        income_duration = _to_int(
            kv.get("income duration (in years)")
            or kv.get("'income duration' (in years)")
            or ""
        )

        payout_freq = (kv.get("income benefit pay-out frequency") or "Annual").title() or None
        payout_type = (kv.get("income benefit pay-out type") or "").title() or None

        return ExtractedFields(
            product_name=product_name,
            product_uin=uin,
            bi_generation_date=bi_date,
            proposer_name_transient=proposer,
            life_assured_age=age,
            life_assured_gender=gender,
            mode=mode,
            policy_term_years=pt,
            ppt_years=ppt,
            annualized_premium_excl_tax=instalment_premium_wo_gst,
            income_start_point_text=kv.get("income start point"),
            income_duration_years=income_duration,
            income_payout_frequency=payout_freq,
            income_payout_type=payout_type,
            sum_assured_on_death=sum_assured,
            schedule_rows=schedule_rows,
        )

    def _extract_schedule(self, parsed: ParsedPDF, pt_years: Optional[int]) -> List[Dict[str, Any]]:
        rows_out: List[Dict[str, Any]] = []
        headers: Optional[List[str]] = None
        header_keys: Optional[List[str]] = None
        reached_end = False

        all_tables = _flatten_tables(parsed)

        for tb in all_tables:
            if reached_end:
                break
            if not tb or len(tb) < 2:
                continue

            header_row_idx = None
            for i in range(min(6, len(tb))):
                row = tb[i] or []
                txt = " ".join((_clean_text(c) for c in row)).lower()
                if "policy" in txt and "year" in txt:
                    header_row_idx = i
                    break

            if header_row_idx is not None:
                base = tb[header_row_idx] or []
                merged = [(_clean_text(c) if c is not None else "") for c in base]

                for j in range(header_row_idx + 1, min(header_row_idx + 3, len(tb))):
                    r2 = tb[j] or []
                    for col in range(min(len(merged), len(r2))):
                        nxt = _clean_text(r2[col])
                        if nxt:
                            merged[col] = f"{merged[col]} {nxt}".strip()

                headers = merged
                header_keys = [_header_key(h) for h in headers]
                data_rows = tb[min(len(tb), header_row_idx + 3):]
            else:
                if headers is None or header_keys is None:
                    continue
                data_rows = tb

            for r in (data_rows or []):
                if not r:
                    continue

                row_obj: Dict[str, Any] = {}
                for idx, cell in enumerate(r):
                    if idx >= len(header_keys):
                        continue
                    key = header_keys[idx]
                    if not key:
                        continue
                    if key == "policy_year":
                        row_obj[key] = _to_int(cell)
                    else:
                        row_obj[key] = _to_number(cell)

                py_val = row_obj.get("policy_year")
                if py_val:
                    rows_out.append(row_obj)
                    if pt_years and py_val >= int(pt_years):
                        reached_end = True
                        break

        return rows_out

    def calculate(self, extracted: ExtractedFields, ptd: date) -> ComputedOutputs:
        """Compute Fully Paid vs Reduced Paid-Up values for GIS.

        Income RPU logic (as per SL):
          R = Pp / Pt
          Reduced paid-up income payable = (It * R) - (Ia * (1 - R))

        where:
          - Pp = premiums payable from RCD up to PTD (in months for this product)
          - Pt = total premiums payable during PPT (in months)
          - It = total income benefits over the full income payout term (as per BI schedule)
          - Ia = income already paid up to the RPU date (assume no payout on RCD; premium is paid first, then payout)
        """

        rcd, rpu_date, grace_days = derive_rcd_and_rpu_dates(
            bi_date=extracted.bi_generation_date,
            ptd=ptd,
            mode=extracted.mode,
        )

        # ---- Premium months (Pp, Pt) ----
        # IMPORTANT: months_paid is based on PTD (not PTD+grace), per agreed logic.
        months_paid = max(0, (ptd.year - rcd.year) * 12 + (ptd.month - rcd.month))
        months_payable_total = int(extracted.ppt_years) * 12 if extracted.ppt_years else 0

        R = (months_paid / months_payable_total) if months_payable_total > 0 else 0.0
        R = max(0.0, min(1.0, R))

        # ---- Build income event schedule from BI (calendar years) ----
        income_events: List[Dict[str, Any]] = []
        for r in (extracted.schedule_rows or []):
            py = r.get("policy_year")
            inc = r.get("income")
            if py is None:
                continue
            inc_f = float(inc) if inc is not None else 0.0
            if inc_f <= 0:
                continue
            py_i = int(py)

            # Calendar year label: RCD.year + PolicyYear - 1
            cal_year = int(rcd.year + py_i - 1)

            # Payout date at end of nth policy year = RCD + (n-1) years
            payout_date = _safe_anniversary(rcd, py_i - 1)

            income_events.append(
                {
                    "policy_year": py_i,
                    "calendar_year": cal_year,
                    "payout_date": payout_date,
                    "amount": inc_f,
                }
            )
        income_events.sort(key=lambda x: x["policy_year"])

        It = sum(e["amount"] for e in income_events)

        # Income already paid up to RPU date (strictly before RPU date)
        Ia = sum(e["amount"] for e in income_events if e["payout_date"] < rpu_date)

        # ---- Reduced paid-up income payable (net after adjustment) ----
        rpu_income_total = (It * R) - (Ia * (1.0 - R))
        if rpu_income_total < 0:
            rpu_income_total = 0.0

        # Remaining full-pay income after RPU date (for reference)
        income_due_full = sum(e["amount"] for e in income_events if e["payout_date"] >= rpu_date)

        maturity = _last_non_null(extracted.schedule_rows, "maturity")
        last_death = _last_non_null(extracted.schedule_rows, "death")

        segments = _income_segments(extracted.schedule_rows, rcd)

        fully_paid = {
            "instalment_premium_without_gst": extracted.annualized_premium_excl_tax,
            "total_income": float(It),
            "income_segments": segments,
            "income_items": income_events,
            "maturity": float(maturity) if maturity is not None else None,
            "death_last_year": float(last_death) if last_death is not None else None,
        }

        remaining_events = [e for e in income_events if e["payout_date"] >= rpu_date]
        excess_paid = Ia * (1.0 - R)
        deduction_per_remaining = (excess_paid / len(remaining_events)) if remaining_events else 0.0

        income_items_rpu: List[Dict[str, Any]] = []
        for e in remaining_events:
            scaled = float(e["amount"]) * R
            adj = max(0.0, scaled - deduction_per_remaining)
            income_items_rpu.append(
                {
                    "policy_year": e["policy_year"],
                    "calendar_year": e["calendar_year"],
                    "payout_date": e["payout_date"],
                    "amount": round(adj, 2),
                }
            )

        reduced_paid_up = {
            "rpu_factor": round(R, 6),
            "income_total_full": float(It),
            "income_already_paid": float(Ia),
            "income_due_full": float(income_due_full),
            "income_payable_after_rpu": float(rpu_income_total),
            "income_segments": segments,
            "income_items": income_items_rpu,
            "excess_paid_income": float(excess_paid),
            "deduction_per_remaining": float(deduction_per_remaining),
            "maturity": (float(maturity) * R) if maturity is not None else None,
            "death_scaled": (float(last_death) * R) if last_death is not None else None,
        }

        notes = [
            "Device is logged as 'unknown' (internal prototype).",
            "Calendar year = RCD.year + PolicyYear - 1 (as per derived RCD).",
            "Income already paid (Ia) includes payouts with payout_date < RPU date (PTD + grace).",
            "Reduced paid-up income payable uses: (It × R) − (Ia × (1 − R)), where R = Pp/Pt.",
        ]

        return ComputedOutputs(
            rcd=rcd,
            ptd=ptd,
            rpu_date=rpu_date,
            grace_period_days=grace_days,
            months_paid=months_paid,
            months_payable_total=months_payable_total,
            rpu_factor=R,
            fully_paid=fully_paid,
            reduced_paid_up=reduced_paid_up,
            notes=notes,
        )
