"""Snippet guard: every figure in a finding or report paragraph must appear in a page it cites."""

from helpers import FakeAnthropic, make_llm, make_settings, read_trace, tool_reply

from app.agents.researcher import Researcher
from app.schemas import (
    ReportDocument,
    ReportParagraph,
    ReportSection,
    ResearchSource,
    ResearchTask,
    unsupported_numbers,
    validate_citations,
)
from app.sessions import AgentSession
from app.schemas import PageContent
from app.trace import Trace

TPI_PAGE = ("The government intends to legislate to appoint Ofgem as regulator of third-party intermediaries "
            "when parliamentary time allows. Ofgem launched a market review on 4 June 2026. ") * 3


def source(sid="S1", text=TPI_PAGE, dates=None):
    return ResearchSource(source_id=sid, title="TPI regulation", url="https://www.gov.uk/tpi",
                          retrieval_status="read", extracted_content=text, published_date=dates)


# ---- the check itself ----

def test_figures_are_normalised_before_matching():
    page = source(text="Main rate 0.00801 per kWh. Discount 92% for 1,000 kWh a month, 2026–2027 and £13.74/MWh.")
    text = "Rate £0.00801, 92 percent, 1000 kWh, 2026-2027, 13.74 per MWh [S1]"
    assert unsupported_numbers(text, [page]) == []


def test_single_digits_are_ignored_and_dates_count():
    page = source(dates="published 2015-07-29T12:00:00Z")
    assert unsupported_numbers("Respond within 5 working days; published 29 July 2015", [page]) == []


def test_unsupported_figures_are_listed_in_order():
    assert unsupported_numbers("From 2027, fees capped at £250 and 15%", [source()]) == ["2027", "250", "15"]


# ---- the creator side (report paragraphs) ----

def report(text, ids=("S1",)):
    return ReportDocument(title="R", topic="t", introduction="Intro [S1].",
                          sections=[ReportSection(heading="Brokers", paragraphs=[
                              ReportParagraph(text=text, source_ids=list(ids))])])


def test_live_run_snippet_case_is_caught_in_the_report():
    # The exact wording from live run 20261103-221117-78fd68, where 2027 came only from a search snippet.
    text = "No start date for regulation was found, and a broker-blog claim of mandatory registration from 2027 was not confirmed. [S1]"
    problems = validate_citations(report(text), [source()])
    assert problems == ["paragraph 1 of 'Brokers' states 2027, which its cited pages do not contain"]


def test_supported_report_passes():
    assert validate_citations(report("Ofgem launched a market review on 4 June 2026 [S1]."), [source()]) == []


def intro_report(intro):
    return ReportDocument(title="R", topic="t", introduction=intro, sections=[
        ReportSection(heading="Brokers", paragraphs=[ReportParagraph(text="Ofgem launched a review [S1].", source_ids=["S1"])])])


def test_uncited_introduction_may_not_contain_figures():
    # Previously an uncited introduction fell back to checking against every source.
    problems = validate_citations(intro_report("Ofgem launched a market review in 2026."), [source()])
    assert problems == ["the introduction states 2026, which its cited pages do not contain"]


def test_scope_only_introduction_passes():
    intro = "This report covers broker regulation, drawing on GOV.UK and Ofgem guidance."
    assert validate_citations(intro_report(intro), [source()]) == []


def test_cited_introduction_is_checked_against_its_citations():
    assert validate_citations(intro_report("Ofgem launched a market review in 2026 [S1]."), [source()]) == []
    other = source("S2", text="Unrelated page about something else entirely. " * 3)
    problems = validate_citations(intro_report("A review began in 2026 [S2]."), [source(), other])
    assert problems == ["the introduction states 2026, which its cited pages do not contain"]


def test_figure_from_another_source_is_not_enough():
    other = source("S2", text="Registration becomes mandatory from 2027 according to this page. " * 3)
    problems = validate_citations(report("Mandatory registration from 2027 [S1]."), [source(), other])
    assert problems and "2027" in problems[0]
    assert validate_citations(report("Mandatory registration from 2027 [S2].", ids=("S2",)), [source(), other]) == []


# ---- the researcher side (findings) ----

TASK = ResearchTask(id="T1", question="Broker regulation status?", evidence_required="GOV.UK")
URL = "https://www.gov.uk/tpi"


def submit(limitations):
    return ("submit_findings", {"findings": [
        {"claim": "Ofgem will regulate brokers.", "evidence": "Government intends to legislate.",
         "source_ids": ["S1"], "limitations": limitations},
        {"claim": "Ofgem launched a review on 4 June 2026.", "evidence": "Stated on the page.",
         "source_ids": ["S1"], "limitations": ""},
    ], "irrelevant_source_ids": [], "incomplete": False, "notes": ""})


def run(*replies, tmp_path):
    trace = Trace(tmp_path / "t.jsonl", "run-1")
    fake = FakeAnthropic([tool_reply(("open_page", {"url": URL})), *replies])
    page = PageContent(url=URL, ok=True, title="TPI regulation", text=TPI_PAGE)
    r = Researcher(AgentSession("researcher", "p"), make_llm(fake, trace=trace), make_settings(),
                   lambda q: [], lambda u: page)
    result = r.run_tasks([TASK])
    return result, fake, read_trace(tmp_path / "t.jsonl")


SNIPPET = "A broker blog claims mandatory registration from 2027; not confirmed."


def test_snippet_figure_in_finding_is_sent_back_once(tmp_path):
    result, fake, records = run(tool_reply(submit(SNIPPET)), tool_reply(submit("No start date given.")),
                                tmp_path=tmp_path)
    rejection = fake.calls[2]["messages"][-1]["content"][0]
    assert rejection["is_error"] and "finding 1: 2027" in rejection["content"]
    assert "Search snippets are leads only" in rejection["content"]
    assert [f.id for f in result.findings] == ["F1", "F2"]
    assert [r["status"] for r in records if r["event"] == "number_check"] == ["warning", "ok"]


def test_repeated_unsupported_figure_drops_only_that_finding(tmp_path):
    result, _, records = run(tool_reply(submit(SNIPPET)), tool_reply(submit(SNIPPET)), tmp_path=tmp_path)
    assert [f.claim for f in result.findings] == ["Ofgem launched a review on 4 June 2026."]
    assert result.findings[0].id == "F1"  # IDs stay sequential after a drop
    dropped = [r for r in records if r["event"] == "finding_rejected_unsupported_number"]
    assert len(dropped) == 1 and dropped[0]["metadata"]["numbers"] == ["2027"]
