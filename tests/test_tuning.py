"""Plan-size and follow-up limits, relevance selection for long pages, source reuse and config overrides."""

import pytest
from helpers import FakeAnthropic, json_reply, make_llm, make_settings, read_trace, tool_reply
from test_workflow import ACCEPT, PLAN, StubResearcher, make_workflow

from app.agents.orchestrator import Orchestrator
from app.agents.researcher import Researcher, select_relevant, task_keywords
from app.config import REQUIRED_VARS, load_settings
from app.llm import MalformedAgentResponse
from app.schemas import PageContent, ResearchTask, SearchHit
from app.sessions import AgentSession


def plan_with(n):
    return {"topic": "t", "tasks": [{"id": f"T{i}", "question": f"q{i}", "evidence_required": "e"} for i in range(1, n + 1)]}


# ---- plan size ----

def test_oversized_plan_gets_one_repair():
    fake = FakeAnthropic([json_reply(plan_with(8)), json_reply(plan_with(5))])
    plan = Orchestrator(AgentSession("orchestrator", "p"), make_llm(fake), max_plan_tasks=6).create_plan("topic")
    assert len(plan.tasks) == 5
    assert "the plan has 8 tasks; it must have between 4 and 6" in fake.calls[1]["messages"][-1]["content"]
    assert "with 4 to 6 tasks" in fake.calls[0]["messages"][0]["content"]


def test_oversized_plan_twice_fails():
    fake = FakeAnthropic([json_reply(plan_with(7)), json_reply(plan_with(9))])
    with pytest.raises(MalformedAgentResponse, match="between 4 and 6"):
        Orchestrator(AgentSession("orchestrator", "p"), make_llm(fake), max_plan_tasks=6).create_plan("topic")


# ---- follow-up cap ----

def test_follow_ups_are_trimmed_to_the_limit(tmp_path):
    more = {"outcome": "RESEARCH_MORE", "rationale": "gaps", "follow_up_tasks": [
        {"id": f"T{i}", "question": f"q{i}", "evidence_required": "e"} for i in range(3, 9)]}
    researcher = StubResearcher()
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(more), json_reply(ACCEPT)], researcher,
                             pdf_writer=lambda html, path: path.write_bytes(b"x"))
    wf.run("run-test", "topic")
    assert researcher.calls[1] == ["T3", "T4", "T5"]
    trimmed = next(r for r in read_trace(tmp_path / "run" / "trace.jsonl") if r["event"] == "followup_trimmed")
    assert trimmed["metadata"] == {"requested": ["T3", "T4", "T5", "T6", "T7", "T8"], "kept": ["T3", "T4", "T5"], "limit": 3}


# ---- relevance selection ----

def test_short_text_is_unchanged():
    assert select_relevant("short page", {"vat"}, 1000) == ("short page", False)


def test_long_text_keeps_opening_and_relevant_passages_in_order():
    lines = ["Intro: HMRC notice 701/19 on fuel and power."] + [f"Filler line {i} about forms." for i in range(300)]
    lines[150] = "Section 4: the VAT de minimis threshold is 33 kWh a day."
    lines[290] = "Section 9: reduced rate VAT applies below the de minimis threshold."
    text, truncated = select_relevant("\n".join(lines), {"minimis", "threshold"}, 3000)
    assert truncated and len(text) <= 3000
    assert text.startswith("Intro: HMRC notice")
    assert text.index("Section 4") < text.index("Section 9")
    assert "[...]" in text


def test_task_keywords_drop_stopwords():
    task = ResearchTask(id="T1", question="What are the VAT rules for business electricity?",
                        evidence_required="Official HMRC guidance")
    assert task_keywords(task) == {"rules", "electricity", "hmrc"}


def test_pdfs_get_a_larger_limit_than_html():
    text = "\n".join(f"Paragraph {i} about standing charges and unit rates." for i in range(2000))
    pages = {"https://a.gov.uk/doc.pdf": PageContent(url="https://a.gov.uk/doc.pdf", ok=True, content_type="pdf",
                                                    title="PDF", text=text),
             "https://a.gov.uk/page": PageContent(url="https://a.gov.uk/page", ok=True, title="HTML", text=text)}
    fake = FakeAnthropic([
        tool_reply(("open_page", {"url": "https://a.gov.uk/doc.pdf"}), ("open_page", {"url": "https://a.gov.uk/page"})),
        tool_reply(("submit_findings", {"findings": [], "irrelevant_source_ids": [], "incomplete": True, "notes": ""})),
    ])
    r = Researcher(AgentSession("researcher", "p"), make_llm(fake),
                   make_settings(page_char_limit=5000, pdf_char_limit=20000), lambda q: [], pages.get)
    result = r.run_tasks([ResearchTask(id="T1", question="standing charges", evidence_required="e")])
    pdf, html = result.sources
    assert pdf.truncated and html.truncated
    assert 15000 < len(pdf.extracted_content) <= 20000 and len(html.extracted_content) <= 5000
    assert pdf.full_chars == html.full_chars == len(text)


# ---- source reuse nudge ----

def test_later_tasks_are_told_which_sources_are_already_read():
    page = PageContent(url="https://www.ofgem.gov.uk/a", ok=True, title="Ofgem contracts page", text="Contract text " * 30)
    submit = ("submit_findings", {"findings": [{"claim": "c", "evidence": "e", "source_ids": ["S1"], "limitations": ""}],
                                  "irrelevant_source_ids": [], "incomplete": False, "notes": ""})
    fake = FakeAnthropic([tool_reply(("open_page", {"url": page.url})), tool_reply(submit), tool_reply(submit)])
    r = Researcher(AgentSession("researcher", "p"), make_llm(fake), make_settings(),
                   lambda q: [SearchHit(title="x", url=page.url)], lambda u: page)
    r.run_tasks([ResearchTask(id="T1", question="q", evidence_required="e"),
                 ResearchTask(id="T2", question="q2", evidence_required="e")])
    first_task = fake.calls[0]["messages"][0]["content"]
    second_task = fake.calls[2]["messages"][-1]["content"]
    assert "already read" not in first_task
    assert "- S1: Ofgem contracts page" in second_task


# ---- config ----

def test_every_integer_limit_can_be_overridden(monkeypatch):
    for name in REQUIRED_VARS:
        monkeypatch.setenv(name, "x" * 10)
    monkeypatch.setenv("MAX_PLAN_TASKS", "4")
    monkeypatch.setenv("MAX_FOLLOW_UP_TASKS", "2")
    monkeypatch.setenv("PDF_CHAR_LIMIT", "25000")
    s = load_settings(env_file=None)
    assert (s.max_plan_tasks, s.max_follow_up_tasks, s.pdf_char_limit, s.page_char_limit) == (4, 2, 25000, 12000)
