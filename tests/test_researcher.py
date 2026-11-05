from helpers import FakeAnthropic, json_reply, make_llm, make_settings, text_reply, tool_reply

from app.agents.researcher import FINAL_SUBMISSION_MESSAGE, SUBMISSION_SCHEMA, Researcher
from app.schemas import PageContent, ResearchTask, RetrievalStatus, SearchHit
from app.sessions import AgentSession

TASK = ResearchTask(id="T1", question="What are standing charges?", evidence_required="Ofgem guidance")
URL = "https://www.ofgem.gov.uk/standing-charges"


def submit(source_ids=("S1",), irrelevant=(), incomplete=False):
    findings = [{"claim": "c", "evidence": "e", "source_ids": list(source_ids), "limitations": ""}] if source_ids else []
    return ("submit_findings", {
        "findings": findings, "irrelevant_source_ids": list(irrelevant), "incomplete": incomplete, "notes": "",
    })


class Tools:
    def __init__(self, pages=None):
        self.queries, self.opened = [], []
        self.pages = pages or {}

    def search(self, query):
        self.queries.append(query)
        return [SearchHit(title="Ofgem", url=URL, snippet="Standing charges are daily fees.")]

    def open(self, url):
        self.opened.append(url)
        return self.pages.get(url) or PageContent(url=url, ok=True, title="Standing charges", text="A daily fixed fee.")


def make_researcher(fake, tools=None, clock=None, **settings):
    tools = tools or Tools()
    kwargs = {"clock": clock} if clock else {}
    r = Researcher(AgentSession("researcher", "p"), make_llm(fake), make_settings(**settings), tools.search, tools.open, **kwargs)
    return r, tools


def test_happy_path_produces_sourced_findings():
    fake = FakeAnthropic([
        tool_reply(("web_search", {"query": "standing charges"})),
        tool_reply(("open_page", {"url": URL})),
        tool_reply(submit()),
    ])
    r, tools = make_researcher(fake)
    result = r.run_tasks([TASK])
    assert [s.source_id for s in result.sources] == ["S1"]
    assert result.sources[0].retrieval_status is RetrievalStatus.READ
    assert result.findings[0].id == "F1" and result.findings[0].task_id == "T1"
    assert not result.incomplete


def test_snippet_alone_cannot_be_cited():
    fake = FakeAnthropic([
        tool_reply(("web_search", {"query": "standing charges"})),
        tool_reply(submit(source_ids=("S1",))),  # never opened the page
        tool_reply(submit(source_ids=(), incomplete=True)),
    ])
    r, tools = make_researcher(fake)
    result = r.run_tasks([TASK])
    assert result.sources == [] and result.findings == []
    assert result.incomplete
    errors = [b for b in fake.calls[2]["messages"][-1]["content"] if b.get("is_error")]
    assert "not a source ID you were given" in errors[0]["content"]


def test_failed_page_is_recorded_and_cannot_be_cited():
    tools = Tools(pages={URL: PageContent(url=URL, ok=False, http_status=403, error="HTTP 403")})
    fake = FakeAnthropic([
        tool_reply(("open_page", {"url": URL})),
        tool_reply(submit(source_ids=("S1",))),
        tool_reply(submit(source_ids=(), incomplete=True)),
    ])
    r, _ = make_researcher(fake, tools)
    result = r.run_tasks([TASK])
    assert result.sources[0].retrieval_status is RetrievalStatus.FAILED
    assert result.sources[0].error == "HTTP 403"
    assert "not read successfully" in fake.calls[2]["messages"][-1]["content"][0]["content"]


def test_browser_exception_becomes_failed_source():
    def boom(url):
        raise TimeoutError("navigation timeout")

    fake = FakeAnthropic([tool_reply(("open_page", {"url": URL})), tool_reply(submit(source_ids=(), incomplete=True))])
    r = Researcher(AgentSession("researcher", "p"), make_llm(fake), make_settings(), lambda q: [], boom)
    result = r.run_tasks([TASK])
    assert result.sources[0].retrieval_status is RetrievalStatus.FAILED
    assert "navigation timeout" in result.sources[0].error


def test_duplicate_query_and_url_are_blocked():
    fake = FakeAnthropic([
        tool_reply(("web_search", {"query": "Standing charges"}), ("web_search", {"query": "standing  CHARGES"})),
        tool_reply(("open_page", {"url": URL}), ("open_page", {"url": URL + "#section"})),
        tool_reply(submit()),
    ])
    r, tools = make_researcher(fake)
    r.run_tasks([TASK])
    assert len(tools.queries) == 1 and len(tools.opened) == 1
    first_turn = fake.calls[1]["messages"][-1]["content"]
    assert first_turn[1]["is_error"] and "already ran this search" in first_turn[1]["content"]
    second_turn = fake.calls[2]["messages"][-1]["content"]
    assert "Already opened as S1" in second_turn[1]["content"]


def test_search_cap_enforced():
    fake = FakeAnthropic([
        tool_reply(("web_search", {"query": "a"}), ("web_search", {"query": "b"})),
        tool_reply(submit(source_ids=(), incomplete=True)),
    ])
    r, tools = make_researcher(fake, task_max_searches=1)
    r.run_tasks([TASK])
    assert tools.queries == ["a"]
    assert "Search limit reached" in fake.calls[1]["messages"][-1]["content"][1]["content"]


def test_page_open_cap_enforced():
    fake = FakeAnthropic([
        tool_reply(("open_page", {"url": URL}), ("open_page", {"url": URL + "/2"})),
        tool_reply(submit()),
    ])
    r, tools = make_researcher(fake, task_max_page_opens=1)
    r.run_tasks([TASK])
    assert tools.opened == [URL]


def final_json(source_ids=("S1",)):
    return json_reply(submit(source_ids=source_ids)[1])


def test_turn_cap_triggers_structured_final_submission():
    fake = FakeAnthropic([
        tool_reply(("web_search", {"query": "a"})),
        tool_reply(("open_page", {"url": URL})),  # turn 2 = cap
        final_json(),
    ])
    r, tools = make_researcher(fake, task_max_model_turns=2)
    result = r.run_tasks([TASK])
    final_call = fake.calls[2]
    assert final_call["tool_choice"] == {"type": "none"}
    assert final_call["output_config"]["format"]["schema"] == SUBMISSION_SCHEMA
    assert final_call["messages"][-1]["content"] == FINAL_SUBMISSION_MESSAGE
    assert len(result.findings) == 1 and result.findings[0].source_ids == ["S1"]
    assert result.incomplete  # stopped by a limit
    assert result.failed_task_ids == []


def test_final_submission_is_validated_and_repaired_once():
    fake = FakeAnthropic([text_reply("thinking..."), final_json(("S5",)), final_json(source_ids=())])
    r, _ = make_researcher(fake, task_max_model_turns=1)
    result = r.run_tasks([TASK])
    assert "not a source ID you were given" in fake.calls[2]["messages"][-1]["content"]
    assert result.findings == [] and result.incomplete and result.failed_task_ids == []


def test_invalid_final_submission_fails_the_task():
    fake = FakeAnthropic([text_reply("thinking..."), text_reply("no json"), final_json(("S9",))])
    r, _ = make_researcher(fake, task_max_model_turns=1)
    result = r.run_tasks([TASK])
    assert len(fake.calls) == 3  # one turn, the final call, one repair: always bounded
    assert result.failed_task_ids == ["T1"]
    assert result.incomplete and "task failed" in result.notes


def test_timeout_triggers_final_submission():
    now = [0.0]

    def clock():
        now[0] += 100
        return now[0]

    fake = FakeAnthropic([tool_reply(("web_search", {"query": "a"})), final_json(source_ids=())])
    r, _ = make_researcher(fake, clock=clock, task_timeout_s=50)
    r.run_tasks([TASK])
    assert len(fake.calls) == 2
    assert fake.calls[1]["tool_choice"] == {"type": "none"}


def test_source_cited_by_one_task_stays_read_when_another_task_calls_it_irrelevant():
    # Regression: in the second live run T5 flagged S2 irrelevant after T1 had cited it.
    other = URL + "/other"
    fake = FakeAnthropic([
        tool_reply(("open_page", {"url": URL})),
        tool_reply(submit(source_ids=("S1",))),
        tool_reply(("open_page", {"url": other})),
        tool_reply(submit(source_ids=("S2",), irrelevant=("S1",))),
    ])
    r, _ = make_researcher(fake)
    result = r.run_tasks([TASK, ResearchTask(id="T2", question="q2", evidence_required="e")])
    assert {s.source_id: s.retrieval_status for s in result.sources} == {
        "S1": RetrievalStatus.READ, "S2": RetrievalStatus.READ}


def test_source_flagged_irrelevant_earlier_can_still_be_cited_later():
    fake = FakeAnthropic([
        tool_reply(("open_page", {"url": URL})),
        tool_reply(submit(source_ids=(), irrelevant=("S1",), incomplete=True)),
        tool_reply(submit(source_ids=("S1",))),
    ])
    r, _ = make_researcher(fake)
    result = r.run_tasks([TASK, ResearchTask(id="T2", question="q2", evidence_required="e")])
    assert result.findings[0].source_ids == ["S1"]
    assert result.sources[0].retrieval_status is RetrievalStatus.READ


def test_requests_use_prompt_caching():
    fake = FakeAnthropic([tool_reply(submit(source_ids=(), incomplete=True))])
    r, _ = make_researcher(fake)
    r.run_tasks([TASK])
    assert fake.calls[0]["cache_control"] == {"type": "ephemeral"}


def test_irrelevant_sources_marked_and_ids_unique_across_rounds():
    other = "https://example.com/blog"
    fake = FakeAnthropic([
        tool_reply(("open_page", {"url": URL}), ("open_page", {"url": other})),
        tool_reply(submit(source_ids=("S1",), irrelevant=("S2",))),
        tool_reply(("open_page", {"url": URL + "/more"})),
        tool_reply(submit(source_ids=("S3",))),
    ])
    r, _ = make_researcher(fake)
    first = r.run_tasks([TASK])
    assert {s.source_id: s.retrieval_status.value for s in first.sources} == {"S1": "read", "S2": "irrelevant"}
    second = r.run_tasks([ResearchTask(id="T2", question="q", evidence_required="e")])
    assert [s.source_id for s in second.sources] == ["S3"]
    assert second.findings[0].id == "F2"
