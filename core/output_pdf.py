from __future__ import annotations

from weasyprint import HTML


def render_pdf_from_html(html: str) -> bytes:
    """
    Render a PDF from the provided HTML string (A4 styling handled inside HTML).
    """
    pdf = HTML(string=html).write_pdf()
    return pdf
