"""ACCEPT is refused while deterministic checks fail; on the last round failing tasks are dropped."""

import json

import pytest
from helpers import json_reply, read_trace
from test_workflow import ACCEPT, MORE, PLAN, REPORT, StubResearcher, make_workflow, source

from app.schemas import ResearchResult, RetrievalStatus
from app.tools.pdf import write_pdf


class Thin(StubResearcher):
    """Never produces findings for T2."""

    def run_tasks(self, tasks):
        result = super().run_tasks(tasks)
        return result.model_copy(update={"findings": [f for f in result.findings if f.task_id != "T2"]})


@pytest.fixture
def real_pdf(chromium):
    return lambda html, path: write_pdf(chromium.context, html, path)


def events(tmp_path, name):
    return [r for r in read_trace(tmp_path / "run" / "trace.jsonl") if r["event"] == name]


MORE_FOR_T2 = {**MORE, "follow_up_tasks": [{**MORE["follow_up_tasks"][0], "addresses": "T2"}]}


def test_accept_with_blocking_problems_is_refused_once_and_repaired(tmp_path, real_pdf):
    researcher = Thin()
    wf, fake, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(ACCEPT), json_reply(MORE_FOR_T2),
                                           json_reply(ACCEPT)], researcher, pdf_writer=real_pdf)
    state = wf.run("run-test", "topic")
    refused = events(tmp_path, "accept_refused")
    assert len(refused) == 1 and refused[0]["metadata"]["blocking_issues"] == ["Task T2 has no findings"]
    assert "Task T2 has no findings" in fake.calls[2]["messages"][-1]["content"]  # the repair request
    assert researcher.calls == [["T1", "T2"], ["T3"]]  # the repair's RESEARCH_MORE ran a follow-up round
    # T3 addresses T2, so T3's findings fill T2's gap: the round-2 checks pass and nothing is excluded.
    assert state["status"] == "completed"
    assert state["checks_passed"] and state["approved"].excluded_tasks == []
    assert events(tmp_path, "decision_override") == []


def test_follow_up_that_also_fails_leaves_the_original_excluded(tmp_path, real_pdf):
    class Thinner(StubResearcher):  # nothing for T2 or its follow-up T3
        def run_tasks(self, tasks):
            result = super().run_tasks(tasks)
            return result.model_copy(update={"findings": [f for f in result.findings if f.task_id not in ("T2", "T3")]})

    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(ACCEPT), json_reply(MORE_FOR_T2),
                                        json_reply(ACCEPT)], Thinner(), pdf_writer=real_pdf)
    state = wf.run("run-test", "topic")
    assert state["status"] == "completed", state.get("error")
    assert [(e.task_id, e.reason) for e in state["approved"].excluded_tasks] == [("T2", "no findings"),
                                                                                ("T3", "no findings")]


def test_follow_up_with_unknown_addresses_is_cleared_not_fatal(tmp_path, real_pdf):
    bad = {**MORE, "follow_up_tasks": [{**MORE["follow_up_tasks"][0], "addresses": "T99"}]}
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(bad), json_reply(ACCEPT)], StubResearcher(),
                             pdf_writer=real_pdf)
    state = wf.run("run-test", "topic")
    assert state["status"] == "completed"
    assert next(t for t in state["tasks"] if t.id == "T3").addresses is None
    logged = events(tmp_path, "followup_addresses_invalid")[0]
    assert logged["task_id"] == "T3" and logged["metadata"]["addresses"] == "T99"


def test_repair_that_accepts_again_triggers_the_drop_rule(tmp_path, real_pdf):
    researcher = Thin()
    wf, fake, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(ACCEPT), json_reply(ACCEPT)], researcher,
                                pdf_writer=real_pdf)
    state = wf.run("run-test", "topic")
    assert len(researcher.calls) == 1
    assert state["status"] == "completed", state.get("error")
    override = events(tmp_path, "decision_override")[0]["metadata"]
    assert override["original_outcome"] == "ACCEPT" and override["new_outcome"] == "ACCEPT"
    assert override["excluded_task_ids"] == ["T2"]
    assert "Termination?" in fake.calls[3]["messages"][-1]["content"]  # the creator is told what was excluded


def test_last_round_accept_drops_failing_task_and_renders_limitations(tmp_path, real_pdf):
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(ACCEPT)], Thin(), pdf_writer=real_pdf,
                             max_research_rounds=1)
    state = wf.run("run-test", "topic")
    assert state["status"] == "completed", state.get("error")
    assert events(tmp_path, "accept_refused") == []  # no rounds left, so no repair; straight to the drop rule
    approved = state["approved"]
    assert [e.task_id for e in approved.excluded_tasks] == ["T2"]
    assert {f.task_id for f in approved.findings} == {"T1"} and [s.source_id for s in approved.sources] == ["S1"]
    html = (tmp_path / "run" / "report.html").read_text(encoding="utf-8")
    assert "Limitations of this report" in html and "Termination?" in html and "(no findings)" in html
    assert (tmp_path / "run" / "report.pdf").read_bytes().startswith(b"%PDF")
    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text())
    assert summary["decision_overrides"] == 1
    assert summary["excluded_tasks"] == [{"task_id": "T2", "question": "Termination?", "reason": "no findings"}]


def test_all_tasks_failing_stops(tmp_path):
    plan = {**PLAN, "tasks": [{**t, "min_sources": 2} for t in PLAN["tasks"]]}  # each task finds only 1 source
    wf, _, _ = make_workflow(tmp_path, [json_reply(plan), json_reply(ACCEPT)], StubResearcher(), creator_replies=(),
                             max_research_rounds=1)
    state = wf.run("run-test", "topic")
    assert state["status"] == "stopped"
    assert "no task passed the deterministic checks" in state["error"]


def test_no_pages_read_stops(tmp_path):
    class NothingRead(StubResearcher):
        def run_tasks(self, tasks):
            failed = source(1).model_copy(update={"retrieval_status": RetrievalStatus.FAILED, "extracted_content": "",
                                                  "error": "HTTP 403"})
            return ResearchResult(task_ids=[t.id for t in tasks], sources=[failed], incomplete=True)

    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(MORE)], NothingRead(), creator_replies=(),
                             max_research_rounds=1)
    state = wf.run("run-test", "topic")
    assert state["status"] == "stopped"
    assert "no source pages were read successfully" in state["error"]


def test_reviewer_caveats_reach_the_creator_but_not_the_pdf_verbatim(tmp_path, real_pdf):
    accept = {**ACCEPT, "caveats": ["T2 rests only on a commercial source.", "F2 supersedes F1."]}
    wf, fake, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(accept)], StubResearcher(), pdf_writer=real_pdf)
    state = wf.run("run-test", "topic")
    assert state["status"] == "completed"
    assert state["approved"].reviewer_caveats == accept["caveats"]
    creator_request = fake.calls[2]["messages"][-1]["content"]
    assert "The reviewer accepted this research with these caveats" in creator_request
    assert "- F2 supersedes F1." in creator_request
    html = (tmp_path / "run" / "report.html").read_text(encoding="utf-8")
    assert "commercial source" not in html  # only the creator's own cited paragraphs reach the PDF
    approved = events(tmp_path, "research_approved")[0]["metadata"]
    assert approved["reviewer_caveats"] == 2


def test_caveats_survive_the_drop_rule(tmp_path, real_pdf):
    accept = {**ACCEPT, "caveats": ["T1 relies on one page."]}
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(accept)], Thin(), pdf_writer=real_pdf,
                             max_research_rounds=1)
    state = wf.run("run-test", "topic")
    assert [e.task_id for e in state["approved"].excluded_tasks] == ["T2"]
    assert state["approved"].reviewer_caveats == ["T1 relies on one page."]


def test_passing_checks_leave_accept_unchanged(tmp_path, real_pdf):
    wf, fake, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(ACCEPT)], StubResearcher(), pdf_writer=real_pdf)
    state = wf.run("run-test", "topic")
    assert state["status"] == "completed"
    assert state["decisions"][-1].rationale == ACCEPT["rationale"]
    assert state["approved"].excluded_tasks == []
    assert events(tmp_path, "accept_refused") == [] and events(tmp_path, "decision_override") == []
    assert "Limitations of this report" not in (tmp_path / "run" / "report.html").read_text(encoding="utf-8")
    assert len(fake.calls) == 3  # plan, review, creator: no extra calls
    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text())
    assert summary["decision_overrides"] == 0 and summary["excluded_tasks"] == []
    assert REPORT["title"] in (tmp_path / "run" / "report.html").read_text(encoding="utf-8")
