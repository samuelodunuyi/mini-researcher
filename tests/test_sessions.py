import json

from helpers import FakeAnthropic, json_reply, make_llm, make_settings, tool_reply

from app.agents.creator import Creator
from app.agents.orchestrator import Orchestrator
from app.agents.researcher import Researcher
from app.schemas import PageContent, ResearchTask, SearchHit
from app.sessions import AGENT_NAMES, AgentSession, SessionStore

PLAN = {
    "topic": "t",
    "tasks": [{"id": "T1", "question": "What contract types exist?", "evidence_required": "Ofgem guidance"}],
}
ACCEPT = {"outcome": "ACCEPT", "rationale": "Coverage is adequate."}


def test_store_creates_three_independent_sessions():
    store = SessionStore("run-1")
    sessions = [store.get(n) for n in AGENT_NAMES]
    assert len({s.session_id for s in sessions}) == 3
    assert len({s.system_prompt for s in sessions}) == 3
    store.get("orchestrator").add_user("hello")
    assert len(store.get("orchestrator")) == 1
    assert len(store.get("researcher")) == 0 and len(store.get("creator")) == 0


def test_history_is_a_copy():
    s = AgentSession("x", "prompt")
    s.add_user("one")
    s.history().append({"role": "user", "content": "injected"})
    s.history()[0]["content"] = "changed"
    assert s.history() == [{"role": "user", "content": "one"}]


def test_orchestrator_reuses_one_session_for_plan_and_review():
    store = SessionStore("run-1")
    fake = FakeAnthropic([json_reply(PLAN), json_reply(ACCEPT)])
    orch = Orchestrator(store.get("orchestrator"), make_llm(fake), min_plan_tasks=1)
    plan = orch.create_plan("UK business electricity contracts")
    assert plan.topic == "UK business electricity contracts"

    from app.schemas import ResearchResult

    decision = orch.review(ResearchResult(task_ids=["T1"]), [], research_more_allowed=True)
    assert decision.outcome.value == "ACCEPT"
    assert len(store.get("orchestrator")) == 4  # plan request/reply + review request/reply
    # The review call carried the planning turn in its history.
    review_messages = fake.calls[1]["messages"]
    assert "Create a research plan" in review_messages[0]["content"]


def test_agents_never_see_each_others_transcripts():
    store = SessionStore("run-1")
    fake = FakeAnthropic()
    llm = make_llm(fake)
    orch = Orchestrator(store.get("orchestrator"), llm, min_plan_tasks=1)
    researcher = Researcher(
        store.get("researcher"),
        llm,
        make_settings(),
        search_fn=lambda q: [SearchHit(title="Ofgem", url="https://www.ofgem.gov.uk/a", snippet="snip")],
        open_fn=lambda u: PageContent(url=u, ok=True, title="Ofgem page", text="Fixed contracts exist."),
    )
    creator = Creator(store.get("creator"), llm)

    fake.queue(json_reply(PLAN, prose="ORCH-PRIVATE-REASONING"))
    plan = orch.create_plan("topic")

    fake.queue(
        tool_reply(("web_search", {"query": "ofgem contract types"}), prose="RESEARCHER-PRIVATE-THOUGHT"),
        tool_reply(("open_page", {"url": "https://www.ofgem.gov.uk/a"})),
        tool_reply(("submit_findings", {
            "findings": [{"claim": "Fixed contracts exist.", "evidence": "Page says so.", "source_ids": ["S1"], "limitations": ""}],
            "irrelevant_source_ids": [], "incomplete": False, "notes": "",
        })),
    )
    result = researcher.run_tasks(plan.tasks)

    fake.queue(json_reply(ACCEPT))
    orch.review(result, [], research_more_allowed=False)

    fake.queue(json_reply({
        "title": "Report", "topic": "topic", "introduction": "Intro.",
        "sections": [{"heading": "Types", "paragraphs": [{"text": "Fixed contracts exist [S1].", "source_ids": ["S1"]}]}],
    }))
    from app.schemas import ApprovedResearch

    creator.write_report(ApprovedResearch.from_result("topic", result))

    prompts = {n: store.get(n).system_prompt for n in AGENT_NAMES}
    by_agent = {n: [c for c in fake.calls if c["system"] == prompts[n]] for n in AGENT_NAMES}
    assert [len(by_agent[n]) for n in AGENT_NAMES] == [2, 3, 1]

    def seen(agent):
        return json.dumps([c["messages"] for c in by_agent[agent]])

    assert "ORCH-PRIVATE-REASONING" not in seen("researcher") + seen("creator")
    assert "RESEARCHER-PRIVATE-THOUGHT" not in seen("orchestrator") + seen("creator")
    assert "Fixed contracts exist." in seen("creator")  # the hand-off itself does arrive
    # Page text reaches the researcher but is withheld from the orchestrator's review.
    assert "extracted_content" not in seen("orchestrator")


def test_researcher_session_persists_across_rounds():
    store = SessionStore("run-1")
    fake = FakeAnthropic()
    researcher = Researcher(store.get("researcher"), make_llm(fake), make_settings(), lambda q: [], lambda u: None)
    submit = ("submit_findings", {"findings": [], "irrelevant_source_ids": [], "incomplete": True, "notes": "nothing"})
    fake.queue(tool_reply(submit), tool_reply(submit))
    researcher.run_tasks([ResearchTask(id="T1", question="q", evidence_required="e")])
    researcher.run_tasks([ResearchTask(id="T2", question="q2", evidence_required="e")])
    assert "T1" in json.dumps(fake.calls[1]["messages"])  # round 2 sees round 1 history
