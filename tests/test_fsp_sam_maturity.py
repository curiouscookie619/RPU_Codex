from datetime import date

import pytest

from core.models import ExtractedFields
from products.fsp import FSPHandler


def _sample_extracted(plan_option: str) -> ExtractedFields:
    return ExtractedFields(
        product_name="Edelweiss Life- Flexi-Savings Plan",
        bi_generation_date=date(2020, 1, 1),
        plan_option=plan_option,
        mode="Annual",
        policy_term_years=5,
        ppt_years=10,
        annualized_premium_excl_tax=100000,
        sum_assured_on_death=2000000,
        sum_assured_on_maturity=500000,
        accrual_survival_benefits=False,
        schedule_rows=[
            {"policy_year": 1, "sb_total_8": 10000, "maturity_8": None, "death_8": 1000000},
            {"policy_year": 2, "sb_total_8": 10000, "maturity_8": None, "death_8": 1100000},
            {"policy_year": 3, "sb_total_8": 10000, "maturity_8": None, "death_8": 1200000},
            {"policy_year": 4, "sb_total_8": 10000, "maturity_8": None, "death_8": 1300000},
            {"policy_year": 5, "sb_total_8": 10000, "maturity_8": 999999, "death_8": 1400000},
        ],
    )


def test_fsp_uses_sam_for_rpu_maturity_not_maturity8():
    handler = FSPHandler()
    extracted = _sample_extracted("Flexi-Income Option")

    outputs = handler.calculate(extracted, ptd=date(2023, 1, 1))

    # R = 3 / 10 based on dates and annual mode
    assert outputs.rpu_factor == pytest.approx(0.3)
    assert outputs.reduced_paid_up["maturity"] == pytest.approx(150000.0)
    assert outputs.reduced_paid_up["maturity"] != pytest.approx(999999 * outputs.rpu_factor)


def test_fsp_accepts_flexi_income_variants_and_rejects_pro():
    handler = FSPHandler()
    accepted = _sample_extracted("Flexi Income Option")
    handler.calculate(accepted, ptd=date(2023, 1, 1))

    rejected = _sample_extracted("Flexi-Income PRO Option")
    with pytest.raises(ValueError, match="Flexi Income option"):
        handler.calculate(rejected, ptd=date(2023, 1, 1))
