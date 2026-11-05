from types import SimpleNamespace

import anthropic
import pytest
from helpers import FakeAnthropic, json_reply, make_llm, text_reply

from app.agents.creator import Creator
from app.llm import LLMError, MalformedAgentResponse
from app.schemas import ApprovedResearch, ResearchFinding, ResearchPlan, ResearchSource, ReviewDecision
from app.sessions import AgentSession

PLAN = {"topic": "t", "tasks": [{"id": "T1", "question": "q", "evidence_required": "e"}]}


def session():
    return AgentSession("orchestrator", "system prompt")


def test_structured_reply_with_prose_and_fences():
    fake = FakeAnthropic([text_reply("Here is the plan:\n```json\n" + json_reply(PLAN).content[0].text + "\n```")])
    plan = make_llm(fake).call_structured(session(), "plan it", ResearchPlan)
    assert plan.tasks[0].id == "T1"


def test_malformed_reply_gets_one_repair_attempt():
    fake = FakeAnthropic([text_reply("not json at all"), json_reply(PLAN)])
    s = session()
    plan = make_llm(fake).call_structured(s, "plan it", ResearchPlan)
    assert plan.topic == "t"
    assert len(fake.calls) == 2
    assert "previous reply was invalid" in fake.calls[1]["messages"][-1]["content"]


def test_malformed_twice_raises():
    fake = FakeAnthropic([text_reply("{}"), json_reply({"outcome": "RESEARCH_MORE", "rationale": "x"})])
    with pytest.raises(MalformedAgentResponse, match="ReviewDecision"):
        make_llm(fake).call_structured(session(), "review", ReviewDecision)
    assert len(fake.calls) == 2


def test_transient_errors_are_retried():
    fake = FakeAnthropic([anthropic.APIConnectionError(request=None), json_reply(PLAN)])
    assert make_llm(fake).call_structured(session(), "plan", ResearchPlan).topic == "t"


def test_retries_are_bounded():
    fake = FakeAnthropic([anthropic.APIConnectionError(request=None)] * 3)
    with pytest.raises(LLMError, match="after 3 attempts"):
        make_llm(fake).call_structured(session(), "plan", ResearchPlan)


def test_client_errors_are_not_retried():
    response = SimpleNamespace(status_code=400, request=None, headers={})
    fake = FakeAnthropic([anthropic.BadRequestError("bad", response=response, body=None)])
    with pytest.raises(LLMError, match="rejected"):
        make_llm(fake).call_structured(session(), "plan", ResearchPlan)
    assert len(fake.calls) == 1


def test_refusal_raises():
    fake = FakeAnthropic([SimpleNamespace(content=[], stop_reason="refusal")])
    with pytest.raises(LLMError, match="declined"):
        make_llm(fake).call_structured(session(), "plan", ResearchPlan)


def test_creator_repairs_bad_citations():
    sources = [ResearchSource(source_id="S1", title="t", url="https://a.gov.uk", retrieval_status="read", extracted_content="x")]
    doc = {
        "title": "R", "topic": "t", "introduction": "I.",
        "sections": [{"heading": "H", "paragraphs": [{"text": "Claim [S9].", "source_ids": ["S9"]}]}],
    }
    fixed = {**doc, "sections": [{"heading": "H", "paragraphs": [{"text": "Claim [S1].", "source_ids": ["S1"]}]}]}
    fake = FakeAnthropic([json_reply(doc), json_reply(fixed)])
    creator = Creator(AgentSession("creator", "p"), make_llm(fake))
    finding = ResearchFinding(id="F1", task_id="T1", claim="c", evidence="e", source_ids=["S1"])
    report = creator.write_report(ApprovedResearch(topic="topic", findings=[finding], sources=sources))
    assert report.sections[0].paragraphs[0].source_ids == ["S1"]
    assert "S9 is cited but does not exist" in fake.calls[1]["messages"][-1]["content"]
