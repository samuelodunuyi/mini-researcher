import pytest
from pydantic import ValidationError

from app.schemas import (
    ReportDocument,
    ReportParagraph,
    ReportSection,
    ResearchFinding,
    ResearchPlan,
    ResearchResult,
    ResearchSource,
    ResearchTask,
    ReviewDecision,
    ReviewOutcome,
    validate_citations,
)


def task(id="T1"):
    return ResearchTask(id=id, question="What contract types exist?", evidence_required="Official guidance")


def source(sid="S1", status="read", url="https://www.ofgem.gov.uk/page", content="Body text"):
    return ResearchSource(
        source_id=sid, title="Ofgem guidance", url=url, retrieval_status=status, extracted_content=content
    )


def finding(source_ids=("S1",), task_id="T1"):
    return ResearchFinding(
        id="F1", task_id=task_id, claim="Fixed contracts exist.", evidence="Quote", source_ids=list(source_ids)
    )


# --- plans and tasks ---

def test_valid_plan():
    plan = ResearchPlan(topic="Electricity contracts", tasks=[task("T1"), task("T2")])
    assert len(plan.tasks) == 2


def test_plan_requires_tasks():
    with pytest.raises(ValidationError):
        ResearchPlan(topic="x", tasks=[])


def test_plan_rejects_duplicate_task_ids():
    with pytest.raises(ValidationError, match="unique"):
        ResearchPlan(topic="x", tasks=[task("T1"), task("T1")])


def test_plan_from_malformed_json():
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate_json('{"topic": "x", "tasks": [{"id": "T1"}]}')


# --- sources ---

@pytest.mark.parametrize("url", ["not a url", "ftp://example.com/file", "javascript:alert(1)", ""])
def test_source_rejects_invalid_urls(url):
    with pytest.raises(ValidationError):
        source(url=url)


def test_source_rejects_bad_id():
    with pytest.raises(ValidationError):
        source(sid="source-1")


def test_read_source_needs_content():
    with pytest.raises(ValidationError, match="no extracted content"):
        source(content="  ")


def test_failed_source_may_be_empty():
    s = source(status="failed", content="")
    assert s.retrieval_status.value == "failed"


# --- findings and results ---

def test_finding_requires_sources():
    with pytest.raises(ValidationError):
        finding(source_ids=())


def test_valid_result_round_trips_json():
    result = ResearchResult(task_ids=["T1"], sources=[source()], findings=[finding()])
    again = ResearchResult.model_validate_json(result.model_dump_json())
    assert again == result


def test_result_rejects_unknown_source_id():
    with pytest.raises(ValidationError, match="unknown source S9"):
        ResearchResult(task_ids=["T1"], sources=[source()], findings=[finding(source_ids=("S9",))])


def test_result_rejects_citing_failed_source():
    with pytest.raises(ValidationError, match="not successfully read"):
        ResearchResult(task_ids=["T1"], sources=[source(status="failed", content="")], findings=[finding()])


def test_result_rejects_unknown_task():
    with pytest.raises(ValidationError, match="unknown task"):
        ResearchResult(task_ids=["T1"], sources=[source()], findings=[finding(task_id="T7")])


def test_result_rejects_duplicate_source_ids():
    with pytest.raises(ValidationError, match="unique"):
        ResearchResult(task_ids=["T1"], sources=[source(), source()])


# --- review decisions ---

def test_research_more_requires_follow_up():
    with pytest.raises(ValidationError, match="follow-up"):
        ReviewDecision(outcome="RESEARCH_MORE", rationale="Need termination fees")


def test_research_more_with_follow_up():
    d = ReviewDecision(outcome="RESEARCH_MORE", rationale="Gap", follow_up_tasks=[task("T7")])
    assert d.outcome is ReviewOutcome.RESEARCH_MORE


def test_accept_rejects_follow_up():
    with pytest.raises(ValidationError):
        ReviewDecision(outcome="ACCEPT", rationale="Fine", follow_up_tasks=[task()])


def test_unknown_outcome_rejected():
    with pytest.raises(ValidationError):
        ReviewDecision(outcome="MAYBE", rationale="?")


# --- report and citations ---

def report(source_ids=("S1",), text="Fixed deals lock prices [S1]."):
    return ReportDocument(
        title="Report",
        topic="Electricity contracts",
        introduction="Intro.",
        sections=[ReportSection(heading="Types", paragraphs=[ReportParagraph(text=text, source_ids=list(source_ids))])],
    )


def test_paragraph_requires_sources():
    with pytest.raises(ValidationError):
        ReportParagraph(text="Unsupported claim.", source_ids=[])


def test_paragraph_inline_citation_must_be_declared():
    with pytest.raises(ValidationError, match="S2"):
        ReportParagraph(text="Claim [S2].", source_ids=["S1"])


def test_citations_valid():
    assert validate_citations(report(), [source()]) == []


def test_citations_unknown_source():
    problems = validate_citations(report(source_ids=("S1", "S4"), text="Claim [S1][S4]."), [source()])
    assert problems == ["S4 is cited but does not exist"]


def test_citations_require_inline_marker_for_each_listed_source():
    problems = validate_citations(report(source_ids=("S1", "S2"), text="Claim [S1]."), [source(), source("S2")])
    assert problems == ["paragraph 1 of 'Types' lists S2 in source_ids but has no matching inline marker"]


def test_citations_failed_source():
    problems = validate_citations(report(), [source(status="failed", content="")])
    assert problems == ["S1 is cited but was not successfully read"]
