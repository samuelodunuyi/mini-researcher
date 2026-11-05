from datetime import datetime, timezone

from app.schemas import ReportDocument, ReportParagraph, ReportSection, ResearchSource
from app.tools.pdf import render_html, verify_pdf, write_pdf

NOW = datetime(2026, 11, 3, 21, 0, tzinfo=timezone.utc)


def sources():
    return [
        ResearchSource(source_id="S1", title="Get a business energy contract", publisher="Ofgem",
                       url="https://www.ofgem.gov.uk/get-business-energy-contract",
                       published_date="published 2024-03-05", retrieval_status="read", extracted_content="x"),
        ResearchSource(source_id="S2", title="Energy Ombudsman", publisher="Ombudsman Services",
                       url="https://www.energyombudsman.org/", retrieval_status="read", extracted_content="x"),
        ResearchSource(source_id="S3", title="Uncited page", url="https://example.org/", retrieval_status="read",
                       extracted_content="x"),
    ]


def report(title="What a UK business should know <before> signing"):
    return ReportDocument(
        title=title,
        topic="UK business electricity contracts",
        introduction="This report summarises official guidance [S1].",
        sections=[
            ReportSection(heading="Contract types", paragraphs=[
                ReportParagraph(text="Fixed contracts lock the unit rate [S1].", source_ids=["S1"]),
                ReportParagraph(text="Disputes can go to the Energy Ombudsman [S2].", source_ids=["S2"]),
            ]),
        ],
    )


def test_template_renders_content_citations_and_references():
    html = render_html(report(), sources(), NOW, run_id="run-1")
    assert "What a UK business should know &lt;before&gt; signing" in html  # autoescaped
    assert '<a class="cite" href="#ref-S1">[S1]</a>' in html
    assert 'id="ref-S2"' in html and "https://www.energyombudsman.org/" in html
    assert "Ofgem" in html and "Published 5 March 2024" in html
    assert "Accessed" in html and "03 November 2026, 21:00 UTC" in html
    assert "Uncited page" not in html  # only cited sources are listed


def test_reference_dates_are_readable_and_publisher_not_repeated():
    from app.tools.pdf import readable_dates, show_publisher

    assert readable_dates("published 2016-05-17T00:00:00+01:00; last updated 2025-11-27T14:23:14+00:00") == \
        "Published 17 May 2016; last updated 27 November 2025"
    assert readable_dates("published 2026-03-11T00:00:00.000Z") == "Published 11 March 2026"
    repeated = ResearchSource(source_id="S1", title="Switch supplier | Ofgem", publisher="Ofgem",
                              url="https://www.ofgem.gov.uk/x", retrieval_status="read", extracted_content="x")
    assert not show_publisher(repeated)
    assert show_publisher(sources()[1])


def test_cite_filter_escapes_text():
    html = render_html(report().model_copy(update={"introduction": "<script>alert(1)</script> [S1]"}), sources(), NOW)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_real_pdf_generation_and_verification(chromium, tmp_path):
    path = tmp_path / "report.pdf"
    write_pdf(chromium.context, render_html(report(), sources(), NOW), path)
    assert path.read_bytes().startswith(b"%PDF")
    assert verify_pdf(path, report()) == []


def test_verify_detects_missing_empty_and_non_pdf(tmp_path):
    assert verify_pdf(tmp_path / "nope.pdf", report()) == ["PDF file was not created"]
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    assert verify_pdf(empty, report()) == ["PDF file is empty"]
    html = tmp_path / "fake.pdf"
    html.write_text("<html>not a pdf</html>")
    assert verify_pdf(html, report()) == ["file does not start with a PDF header"]


def test_verify_detects_corrupt_pdf(tmp_path):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.7\n garbage without objects")
    problems = verify_pdf(corrupt, report())
    assert problems and ("could not be read" in problems[0] or "no pages" in problems[0])


def test_verify_detects_wrong_content(chromium, tmp_path):
    path = tmp_path / "report.pdf"
    write_pdf(chromium.context, "<html><body><p>Unrelated document</p></body></html>", path)
    problems = verify_pdf(path, report())
    assert "report title not found in PDF text" in problems
    assert "References heading not found in PDF text" in problems
    assert "no inline [S#] citation found in PDF text" in problems
