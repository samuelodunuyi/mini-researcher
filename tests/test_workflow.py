"""LangGraph workflow paths with a fake model and a stub researcher (no network)."""

import json

import pytest
from helpers import FakeAnthropic, json_reply, make_llm, make_settings

from app.agents.creator import Creator
from app.agents.orchestrator import Orchestrator
from app.checks import check_research, merge_results
from app.schemas import ResearchFinding, ResearchResult, ResearchSource, ResearchTask
from app.sessions import AgentSession, SessionStore
from app.tools.pdf import write_pdf
from app.trace import Trace
from app.workflow import Workflow

PLAN = {"topic": "t", "tasks": [
    {"id": "T1", "question": "Contract types?", "evidence_required": "Ofgem guidance"},
    {"id": "T2", "question": "Termination?", "evidence_required": "Ofgem guidance"},
]}
ACCEPT = {"outcome": "ACCEPT", "rationale": "Sufficient."}
STOP = {"outcome": "STOP_INSUFFICIENT", "rationale": "Evidence too thin."}
MORE = {"outcome": "RESEARCH_MORE", "rationale": "T2 lacks detail.", "follow_up_tasks": [
    {"id": "T3", "question": "Termination fees?", "evidence_required": "Supplier terms"}]}
REPORT = {"title": "R", "topic": "t", "introduction": "I.", "sections": [
    {"heading": "H", "paragraphs": [{"text": "Claim [S1].", "source_ids": ["S1"]}]}]}


def source(n):
    return ResearchSource(source_id=f"S{n}", title=f"Page {n}", url=f"https://www.ofgem.gov.uk/{n}",
                          retrieval_status="read", extracted_content="text")


def finding(n, task_id, sid):
    return ResearchFinding(id=f"F{n}", task_id=task_id, claim="c", evidence="e", source_ids=[sid])


class StubResearcher:
    """Returns one finding per task, each on its own new source. Records what it was asked."""

    def __init__(self, fail=False):
        self.session = AgentSession("researcher", "stub prompt")
        self.calls: list[list[str]] = []
        self.fail = fail
        self.n = 0

    def run_tasks(self, tasks):
        if self.fail:
            raise RuntimeError("browser crashed")
        self.calls.append([t.id for t in tasks])
        sources, findings = [], []
        for t in tasks:
            self.n += 1
            sources.append(source(self.n))
            findings.append(finding(self.n, t.id, f"S{self.n}"))
        return ResearchResult(task_ids=[t.id for t in tasks], sources=sources, findings=findings)


def no_pdf(html, path):
    raise AssertionError("this test should not reach PDF generation")


def make_workflow(tmp_path, orchestrator_replies, researcher=None, creator_replies=(json_reply(REPORT),),
                  pdf_writer=no_pdf, **settings):
    store = SessionStore("run-test")
    fake = FakeAnthropic(list(orchestrator_replies) + list(creator_replies))
    trace = Trace(tmp_path / "run" / "trace.jsonl", "run-test", secrets=("test-key", "test-tavily"))
    llm = make_llm(fake, trace=trace)
    s = make_settings(**settings)
    wf = Workflow(
        s,
        # These tests use two-task plans; plan-size limits are tested in test_depth.py and test_tuning.py.
        Orchestrator(store.get("orchestrator"), llm, min_plan_tasks=1, max_follow_up_tasks=s.max_follow_up_tasks),
        researcher or StubResearcher(),
        Creator(store.get("creator"), llm),
        pdf_writer,
        tmp_path / "run",
        trace=trace,
    )
    return wf, fake, store


@pytest.fixture
def real_pdf(chromium):
    return lambda html, path: write_pdf(chromium.context, html, path)


def test_accept_path_produces_verified_pdf(tmp_path, real_pdf):
    researcher = StubResearcher()
    wf, fake, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(ACCEPT)], researcher, pdf_writer=real_pdf)
    state = wf.run("run-test", "UK business electricity contracts")
    assert state["status"] == "completed", state.get("error")
    assert researcher.calls == [["T1", "T2"]]
    assert state["report"].topic == "UK business electricity contracts"
    assert [s.source_id for s in state["approved"].sources] == ["S1", "S2"]
    run_dir = tmp_path / "run"
    files = {p.name for p in run_dir.iterdir()}
    assert {"plan.json", "research_round_1.json", "checks_round_1.json", "review_round_1.json",
            "approved_research.json", "report_content.json", "report.html", "report.pdf", "run_summary.json"} <= files
    assert (run_dir / "report.pdf").read_bytes().startswith(b"%PDF")
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["status"] == "completed" and summary["decisions"] == ["ACCEPT"]
    assert summary["pdf"].endswith("report.pdf")


def test_pdf_writer_error_fails_the_run(tmp_path):
    def broken(html, path):
        raise RuntimeError("chromium crashed")

    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(ACCEPT)], pdf_writer=broken)
    state = wf.run("run-test", "topic")
    assert state["status"] == "failed" and state["error"] == "generate_pdf: RuntimeError: chromium crashed"
    assert json.loads((tmp_path / "run" / "run_summary.json").read_text())["pdf"] is None


def test_invalid_pdf_fails_verification(tmp_path):
    def junk(html, path):
        path.write_bytes(b"not a pdf")

    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(ACCEPT)], pdf_writer=junk)
    state = wf.run("run-test", "topic")
    assert state["status"] == "failed"
    assert state["error"].startswith("verify_pdf: ValueError: PDF verification failed: file does not start with a PDF header")


def test_bad_citations_after_repair_fail_the_run(tmp_path):
    bad = {**REPORT, "sections": [{"heading": "H", "paragraphs": [{"text": "Claim [S9].", "source_ids": ["S9"]}]}]}
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(ACCEPT)],
                             creator_replies=(json_reply(bad), json_reply(bad)))
    state = wf.run("run-test", "topic")
    assert state["status"] == "failed"
    assert "create: MalformedAgentResponse" in state["error"] and "S9 is cited but does not exist" in state["error"]
    assert not (tmp_path / "run" / "report.pdf").exists()


def test_research_more_then_accept_runs_only_follow_up_tasks(tmp_path, real_pdf):
    researcher = StubResearcher()
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(MORE), json_reply(ACCEPT)], researcher,
                             pdf_writer=real_pdf)
    state = wf.run("run-test", "topic")
    assert state["status"] == "completed"
    assert researcher.calls == [["T1", "T2"], ["T3"]]
    assert state["round"] == 2
    assert [f.task_id for f in state["research"].findings] == ["T1", "T2", "T3"]
    assert [s.source_id for s in state["research"].sources] == ["S1", "S2", "S3"]


def test_round_limit_converts_second_research_more(tmp_path, real_pdf):
    researcher = StubResearcher()
    more_again = {**MORE, "follow_up_tasks": [{"id": "T4", "question": "q", "evidence_required": "e"}]}
    wf, fake, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(MORE), json_reply(more_again)], researcher,
                                pdf_writer=real_pdf)
    state = wf.run("run-test", "topic")
    assert len(researcher.calls) == 2  # never a third round
    final = state["decisions"][-1]
    assert final.outcome.value == "ACCEPT"  # checks passed, so Python accepted
    assert "Overridden from RESEARCH_MORE" in final.rationale
    assert "No more research rounds are allowed" in fake.calls[2]["messages"][-1]["content"]


def test_round_limit_with_failed_checks_drops_the_failing_task(tmp_path, real_pdf):
    # Previously this stopped the run; the drop rule now excludes the failing task instead.
    class Thin(StubResearcher):
        def run_tasks(self, tasks):  # never covers T2
            result = super().run_tasks(tasks)
            keep = [f for f in result.findings if f.task_id != "T2"]
            return result.model_copy(update={"findings": keep})

    more_again = {**MORE, "follow_up_tasks": [{"id": "T4", "question": "q", "evidence_required": "e"}]}
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(MORE), json_reply(more_again)], Thin(),
                             pdf_writer=real_pdf)
    state = wf.run("run-test", "topic")
    assert state["status"] == "completed", state.get("error")
    assert "round limit reached and deterministic checks failed: excluded T2" in state["decisions"][-1].rationale
    assert [e.task_id for e in state["approved"].excluded_tasks] == ["T2"]
    assert all(f.task_id != "T2" for f in state["approved"].findings)


def test_stop_insufficient(tmp_path):
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(STOP)], creator_replies=())
    state = wf.run("run-test", "topic")
    assert state["status"] == "stopped"
    assert state["error"] == "STOP_INSUFFICIENT: Evidence too thin."
    assert "report" not in state


def test_accept_with_no_findings_is_overridden(tmp_path):
    class Empty(StubResearcher):
        def run_tasks(self, tasks):
            return ResearchResult(task_ids=[t.id for t in tasks], incomplete=True)

    # The first ACCEPT is refused (blocking problems, rounds left); the repair returns ACCEPT again.
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(ACCEPT), json_reply(ACCEPT)], Empty(),
                             creator_replies=())
    state = wf.run("run-test", "topic")
    assert state["status"] == "stopped" and "no findings" in state["error"]


def test_node_exception_fails_the_run(tmp_path):
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN)], StubResearcher(fail=True), creator_replies=())
    state = wf.run("run-test", "topic")
    assert state["status"] == "failed"
    assert state["error"] == "research: RuntimeError: browser crashed"
    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text())
    assert summary["status"] == "failed"


def test_malformed_orchestrator_output_fails_the_run(tmp_path):
    wf, _, _ = make_workflow(tmp_path, [json_reply({"bad": 1}), json_reply({"bad": 2})], creator_replies=())
    state = wf.run("run-test", "topic")
    assert state["status"] == "failed" and "MalformedAgentResponse" in state["error"]


def test_run_timeout_fails(tmp_path):
    now = [0.0]

    def clock():
        now[0] += 10
        return now[0]

    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN)], creator_replies=(), run_timeout_s=15)
    wf.clock = clock
    state = wf.run("run-test", "topic")
    assert state["status"] == "failed" and "timed out" in state["error"]


def test_state_never_contains_transcripts(tmp_path, real_pdf):
    wf, _, store = make_workflow(tmp_path, [json_reply(PLAN, prose="PRIVATE-ORCH-PROSE"), json_reply(ACCEPT)],
                                 pdf_writer=real_pdf)
    state = wf.run("run-test", "topic")
    dumped = json.dumps({k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else
                             [x.model_dump(mode="json") if hasattr(x, "model_dump") else x for x in v]
                             if isinstance(v, list) else v) for k, v in state.items()})
    assert "PRIVATE-ORCH-PROSE" not in dumped
    assert '"role"' not in dumped
    assert len(store.get("orchestrator")) > 0  # the transcript exists, just not in graph state


# ---- deterministic checks ----

def test_checks_flag_missing_coverage_and_failed_pages():
    tasks = [ResearchTask(id="T1", question="q", evidence_required="e", min_sources=2),
             ResearchTask(id="T2", question="q", evidence_required="e")]
    failed = ResearchSource(source_id="S2", title="x", url="https://a.org", retrieval_status="failed", error="HTTP 403")
    result = ResearchResult(task_ids=["T1", "T2"], sources=[source(1), failed], findings=[finding(1, "T1", "S1")])
    report = check_research(tasks, result)
    assert not report.passed
    assert "Task T1 is supported by 1 source(s); the plan asked for at least 2" in report.problems
    assert "Task T2 has no findings" in report.problems
    assert any("could not be read: HTTP 403" in n for n in report.notes)


def test_checks_require_some_read_source():
    result = ResearchResult(task_ids=["T1"], incomplete=True)
    report = check_research([ResearchTask(id="T1", question="q", evidence_required="e")], result)
    assert "No source pages were read successfully" in report.problems


def test_follow_up_findings_count_towards_the_task_they_address():
    tasks = [ResearchTask(id="T1", question="q", evidence_required="e", min_sources=2),
             ResearchTask(id="T2", question="q", evidence_required="e"),
             ResearchTask(id="T3", question="q", evidence_required="e", addresses="T1", min_sources=2),
             ResearchTask(id="T4", question="q", evidence_required="e", addresses="T2")]
    result = ResearchResult(task_ids=["T1", "T2", "T3", "T4"], sources=[source(1), source(2)],
                            findings=[finding(1, "T1", "S1"), finding(2, "T3", "S2"), finding(3, "T2", "S1")])
    report = check_research(tasks, result)
    # T1 has 2 sources together with T3; T3's own shortfall is forgiven because T1 now passes.
    # T4 addresses T2, which passes on its own, but T4 itself has no findings, so it still fails.
    assert report.failing_tasks == {"T4": "no findings"}
    assert report.problems == ["Task T4 has no findings"]


def test_merge_keeps_ids_stable_and_latest_status():
    a = ResearchResult(task_ids=["T1"], sources=[source(1), source(2)], findings=[finding(1, "T1", "S1")])
    irrelevant = source(2).model_copy(update={"retrieval_status": "irrelevant"})
    b = ResearchResult(task_ids=["T3"], sources=[irrelevant, source(10)], findings=[finding(2, "T3", "S10")])
    merged = merge_results(a, b)
    assert [s.source_id for s in merged.sources] == ["S1", "S2", "S10"]
    assert merged.sources[1].retrieval_status == "irrelevant"
    assert merged.task_ids == ["T1", "T3"]


def test_merge_rejects_findings_on_unknown_sources():
    a = ResearchResult(task_ids=["T1"], sources=[source(1)])
    b = ResearchResult.model_construct(task_ids=["T2"], sources=[], findings=[finding(1, "T2", "S7")],
                                       incomplete=False, notes="")
    with pytest.raises(ValueError, match="unknown source S7"):
        merge_results(a, b)
