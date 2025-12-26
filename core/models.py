from __future__ import annotations

from datetime import date, datetime
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class ParsedPDF(BaseModel):
    text_by_page: List[str]
    tables_by_page: List[List[List[List[Optional[str]]]]]
    page_count: int




    @property
    def text(self) -> str:
        """Backward-compatible full text (all pages concatenated)."""
        return "\n".join(self.text_by_page or [])

class ExtractedFields(BaseModel):
    product_name: str
    product_uin: Optional[str] = None
    bi_generation_date: date
    proposer_name_transient: Optional[str] = None  # DO NOT persist
    life_assured_age: Optional[int] = None
    life_assured_gender: Optional[str] = None

    mode: str  # Annual/Half-yearly/Quarterly/Monthly
    policy_term_years: int
    ppt_years: int

    annualized_premium_excl_tax: Optional[float] = None

    income_start_point_text: Optional[str] = None
    income_duration_years: Optional[int] = None
    income_payout_frequency: Optional[str] = None  # Yearly/Half-yearly/Quarterly/Monthly
    income_payout_type: Optional[str] = None  # Increasing/Level

    sum_assured_on_death: Optional[float] = None

    schedule_rows: List[Dict[str, Any]] = Field(default_factory=list)  # normalized schedule per policy year


class ComputedOutputs(BaseModel):
    rcd: date
    ptd: date
    rpu_date: date
    grace_period_days: int

    months_paid: int
    months_payable_total: int
    rpu_factor: float

    fully_paid: Dict[str, Any]
    reduced_paid_up: Dict[str, Any]
    notes: List[str] = Field(default_factory=list)
