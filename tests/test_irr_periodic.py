from datetime import date

from core.irr import attach_irrs, periodic_irr
from core.models import ExtractedFields, ComputedOutputs
from core.pdf_reader import read_pdf
from products.registry import detect_product


def test_periodic_irr_basic():
    irr = periodic_irr([-1000.0, 0.0, 1200.0])
    assert irr is not None
    assert abs(irr - 0.095445) < 1e-4  # (1+r)^2 = 1.2


def test_gis_bi2_expected_irrs():
    # GIS BI 2 reference values
    import os, urllib.request
    if not os.path.exists("GIS BI 2.pdf"):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/curiouscookie619/RPU_Codex/main/GIS%20BI%202.pdf",
            "GIS BI 2.pdf",
        )
    with open("GIS BI 2.pdf", "rb") as f:
        data = f.read()
    parsed = read_pdf(data)
    handler, _, _ = detect_product(parsed)
    ptd = date(2026, 3, 31)
    sv = 2346278  # use the stricter case
    extracted = handler.extract(parsed)
    outputs = handler.calculate(extracted, ptd)
    attach_irrs(extracted, outputs, sv)

    assert outputs.irr_rpu is not None
    assert outputs.irr_fp_incremental is not None

    # Expected references (tolerance ±0.0005)
    assert abs(outputs.irr_rpu - 0.04618336) < 0.0005
    assert abs(outputs.irr_fp_incremental - 0.06135075) < 0.0005
