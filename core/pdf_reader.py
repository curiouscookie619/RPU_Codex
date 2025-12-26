from __future__ import annotations

import re
from datetime import date
from io import BytesIO
from typing import List, Optional

import pdfplumber
from pypdf import PdfReader

from core.models import ParsedPDF


def read_pdf(file_bytes: bytes) -> ParsedPDF:
    """Read a BI PDF into text-by-page and tables-by-page.

    Key requirements for these BIs:
    - Page 1 contains summary tables (premium summary, GST, UIN, etc.).
    - The benefit schedule table (Policy Year rows) can span multiple pages.

    Strategy:
    - Always extract *text* for all pages.
    - Extract *tables* for page 1 & 2 (index 0,1) always.
    - Detect when the schedule starts (a page that contains 'policy year' OR
      looks like it contains schedule rows), then extract tables for that page
      and ALL subsequent pages.

    This ensures we capture schedules even if they spill into page 4+.
    """

    text_by_page: List[str] = []
    tables_by_page: List[List[List[List[Optional[str]]]]] = []

    def _looks_like_schedule_page(page_text: str) -> bool:
        """Heuristic: detect schedule pages even when the header is missing.

        Continuation pages sometimes omit the 'Policy Year' header, but the body
        still contains many rows that begin with numeric columns (age/policy-year).
        """
        if not page_text:
            return False
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        # count lines that start with: <age> <policy_year> ...
        pat = re.compile(r"^\d{1,3}\s+\d{1,3}\s+[-–—]|^\d{1,3}\s+\d{1,3}\s+\d", re.I)
        hits = 0
        for ln in lines[:120]:
            if pat.search(ln):
                hits += 1
            if hits >= 5:
                return True
        return False

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        # First pass: extract text for all pages
        for p in pdf.pages:
            text_by_page.append(p.extract_text() or "")

        # Second pass: extract tables selectively
        schedule_started = False
        for idx, p in enumerate(pdf.pages):
            page_txt = text_by_page[idx] or ""
            txt_l = page_txt.lower()

            has_policy_year = "policy year" in txt_l
            looks_like_schedule = _looks_like_schedule_page(page_txt)

            # Always include first 2 pages (summary blocks), and once schedule starts,
            # include all subsequent pages.
            should_extract_tables = (idx in (0, 1)) or schedule_started or has_policy_year or looks_like_schedule

            if should_extract_tables:
                try:
                    tables = p.extract_tables() or []
                except Exception:
                    tables = []
            else:
                tables = []

            tables_by_page.append(tables)

            if has_policy_year or looks_like_schedule:
                schedule_started = True

    # Fallback: if almost no text, try pypdf extraction
    if sum(len(t.strip()) for t in text_by_page) < 50:
        reader = PdfReader(BytesIO(file_bytes))
        text_by_page = [(page.extract_text() or "") for page in reader.pages]

    return ParsedPDF(
        text_by_page=text_by_page,
        tables_by_page=tables_by_page,
        page_count=len(text_by_page),
    )


_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def extract_bi_generation_date(page_text: str) -> Optional[date]:
    """Extract BI/Quote generation date from page-1 text.

    Supports patterns like:
      - "BI (Quote) Date : 31/03/2023"
      - "Date of Quote: 31-03-2023"
      - unlabelled month-name date: "31 Mar 2023"
    """

    t = (page_text or "").replace("\n", " ")

    # 1) Labelled numeric dates
    labelled_patterns = [
        r"(?:BI\s*\(Quote\)\s*Date|Quote\s*Date|Quotation\s*Date|Date\s*of\s*Quote|BI\s*Date|BI\s*Generation\s*Date)\s*[:\-]?\s*([0-3]?\d)[/\-\.]([01]?\d)[/\-\.]((?:19|20)\d{2})",
    ]
    for p in labelled_patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return date(yy, mm, dd)
            except Exception:
                return None

    # 2) Unlabelled month-name date (common on top-right of BI): e.g. "31 Mar 2023"
    m2 = re.search(
        r"\b([0-3]?\d)\s+([A-Za-z]{3,9})\s+((?:19|20)\d{2})\b",
        t,
        flags=re.IGNORECASE,
    )
    if m2:
        dd = int(m2.group(1))
        mon_raw = m2.group(2).strip().lower()
        yy = int(m2.group(3))
        mm = _MONTHS.get(mon_raw)
        if mm:
            try:
                return date(yy, mm, dd)
            except Exception:
                return None

    return None
