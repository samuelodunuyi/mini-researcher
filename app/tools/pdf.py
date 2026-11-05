"""Render validated report content to HTML (Jinja2), print it to PDF (Chromium) and verify the file."""

import re
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape
from pypdf import PdfReader

from app.schemas import CITATION_RE, ExcludedTask, ReportDocument, ResearchSource

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

FOOTER = (
    '<div style="font-size:8px; width:100%; text-align:center; color:#7b8794;">'
    'Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'
)
MARGINS = {"top": "18mm", "bottom": "20mm", "left": "18mm", "right": "18mm"}


def cite(text: str) -> Markup:
    """Escape text, then turn [S1] markers into links to the references list."""
    escaped = str(escape(text))
    return Markup(CITATION_RE.sub(lambda m: f'<a class="cite" href="#ref-{m[1]}">[{m[1]}]</a>', escaped))


ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:[Tt][\d:.]+(?:[Zz]|[+-]\d{2}:?\d{2})?)?")


def readable_dates(text: str) -> str:
    """'published 2016-05-17T00:00:00+01:00; last updated 2025-11-27' -> 'Published 17 May 2016; last updated 27 November 2025'."""

    def fmt(m: re.Match) -> str:
        try:
            return datetime.strptime(m[1], "%Y-%m-%d").strftime("%d %B %Y").lstrip("0")
        except ValueError:
            return m[0]

    text = ISO_DATE.sub(fmt, text)
    return text[:1].upper() + text[1:]


def show_publisher(source: ResearchSource) -> bool:
    """Skip the publisher when the page title already names it, e.g. 'Switch supplier | Ofgem'."""
    return bool(source.publisher) and source.publisher.lower() not in source.title.lower()


_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(["html"]))
_env.filters["cite"] = cite
_env.filters["readable_dates"] = readable_dates
_env.tests["publisher_shown"] = show_publisher


def render_html(
    report: ReportDocument,
    sources: list[ResearchSource],
    generated_at: datetime,
    run_id: str = "",
    excluded_tasks: list[ExcludedTask] | None = None,
) -> str:
    cited = report.cited_source_ids()
    references = sorted((s for s in sources if s.source_id in cited), key=lambda s: int(s.source_id[1:]))
    return _env.get_template("report.html").render(
        report=report,
        references=references,
        generated=generated_at.strftime("%d %B %Y, %H:%M UTC"),
        run_id=run_id,
        excluded_tasks=excluded_tasks or [],
    )


def write_pdf(browser_context, html: str, path: Path) -> None:
    """Print HTML to an A4 PDF using the run's existing Chromium context."""
    page = browser_context.new_page()
    try:
        page.set_content(html, wait_until="load")
        page.pdf(
            path=str(path),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=FOOTER,
            margin=MARGINS,
        )
    finally:
        page.close()


def verify_pdf(path: Path, report: ReportDocument) -> list[str]:
    """Return problems with the generated PDF; an empty list means it is valid."""
    if not path.exists():
        return ["PDF file was not created"]
    data = path.read_bytes()
    if not data:
        return ["PDF file is empty"]
    if not data.startswith(b"%PDF"):
        return ["file does not start with a PDF header"]
    try:
        reader = PdfReader(path)
        pages = len(reader.pages)
        text = " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # pypdf raises many types on malformed files
        return [f"PDF could not be read: {exc}"]

    problems = []
    if pages < 1:
        problems.append("PDF has no pages")
    flat = " ".join(text.split())
    if " ".join(report.title.split()) not in flat:
        problems.append("report title not found in PDF text")
    if "References" not in flat:
        problems.append("References heading not found in PDF text")
    if not re.search(r"\[S\d+\]", flat):
        problems.append("no inline [S#] citation found in PDF text")
    return problems
