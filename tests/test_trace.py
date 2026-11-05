"""JSONL trace: fields, redaction, workflow/researcher events, run summary totals and CLI reporting."""

import json

import pytest
from helpers import FakeAnthropic, json_reply, make_llm, make_settings, read_trace, text_reply, tool_reply
from test_researcher import URL, Tools, submit
from test_workflow import ACCEPT, MORE, PLAN, STOP, StubResearcher, make_workflow

from app import cli
from app.agents.researcher import Researcher
from app.schemas import PageContent, ResearchTask
from app.sessions import AgentSession
from app.trace import REDACTED, Trace

FIELDS = {"timestamp", "run_id", "agent", "session_id", "event", "task_id", "handoff_id",
          "status", "duration_ms", "metadata", "error"}


def events(records):
    return [r["event"] for r in records]


# ---- the trace itself ----

def test_every_record_has_the_same_fields(tmp_path):
    trace = Trace(tmp_path / "t.jsonl", "run-1")
    trace.emit("x", agent="researcher", session_id="s", task_id="T1", duration_ms=12.345, count=3)
    trace.emit("y", status="error", error=ValueError("bad"))
    lines = read_trace(tmp_path / "t.jsonl")
    assert all(set(r) == FIELDS for r in lines)
    assert lines[0]["metadata"] == {"count": 3} and lines[0]["duration_ms"] == 12.3
    assert lines[1]["error"] == {"type": "ValueError", "message": "bad"}
    assert lines[0]["timestamp"].endswith("+00:00")


def test_secrets_are_redacted_everywhere(tmp_path):
    key = "my-configured-secret-value-123"
    trace = Trace(tmp_path / "t.jsonl", "run-1", secrets=(key,))
    trace.emit(
        "leak_attempt",
        status="error",
        error=RuntimeError(f"auth failed for {key}"),
        nested={"headers": [f"x-api-key: {key}", "Bearer sk-ant-api03-abcdefghijklmnop"]},
        tavily="key tvly-dev-ABCDEFGH12345678 rejected",
    )
    text = (tmp_path / "t.jsonl").read_text(encoding="utf-8")
    for secret in (key, "sk-ant-api03-abcdefghijklmnop", "tvly-dev-ABCDEFGH12345678"):
        assert secret not in text
    assert text.count(REDACTED) == 4


def test_handoff_ids_are_sequential():
    trace = Trace(None)
    assert [trace.new_handoff("plan"), trace.new_handoff("review")] == ["H1-plan", "H2-review"]


# ---- workflow events ----

def test_successful_run_trace_and_summary(tmp_path, chromium):
    from app.tools.pdf import write_pdf

    wf, _, store = make_workflow(tmp_path, [json_reply(PLAN), json_reply(MORE), json_reply(ACCEPT)],
                                 pdf_writer=lambda html, path: write_pdf(chromium.context, html, path))
    state = wf.run("run-test", "topic")
    assert state["status"] == "completed"
    records = read_trace(tmp_path / "run" / "trace.jsonl")
    assert all(set(r) == FIELDS and r["run_id"] == "run-test" for r in records)

    sequence = [e for e in events(records) if e not in ("llm_call", "step_finished")]
    assert sequence == [
        "run_started", "plan_created",
        "research_round_started", "research_returned", "checks_completed", "review_decision", "research_more_requested",
        "research_round_started", "research_returned", "checks_completed", "review_decision",
        "research_approved", "creator_invoked", "report_returned",
        "citation_validation", "html_rendered", "pdf_generated", "pdf_verified", "run_completed",
    ]
    by_event = {r["event"]: r for r in records}
    orch = store.get("orchestrator")
    assert by_event["plan_created"]["agent"] == "orchestrator"
    assert by_event["plan_created"]["session_id"] == orch.session_id
    assert by_event["plan_created"]["handoff_id"] == "H1-plan"
    assert by_event["creator_invoked"]["session_id"] == store.get("creator").session_id
    assert by_event["review_decision"]["metadata"]["outcome"] == "ACCEPT"

    llm_calls = [r for r in records if r["event"] == "llm_call"]
    assert len(llm_calls) == 4 and all(r["metadata"]["input_tokens"] == 100 for r in llm_calls)

    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text())
    assert summary["status"] == "completed" and summary["research_rounds"] == 2
    # Cached input counts towards the total: 4 calls x (100 uncached + 10 written + 50 read).
    assert summary["tokens"]["total"] == {"calls": 4, "input_tokens": 400, "cache_write_tokens": 40,
                                          "cache_read_tokens": 200, "output_tokens": 80, "total_input_tokens": 640}
    assert summary["tokens"]["by_agent"]["orchestrator"]["calls"] == 3
    assert {"plan", "research", "review", "create", "generate_pdf", "verify_pdf"} <= set(summary["stage_seconds"])
    assert summary["sources_read"] == 3 and summary["pdf"].endswith("report.pdf")
    assert by_event["run_completed"]["metadata"]["tokens"]["total"]["calls"] == 4


def test_failed_run_records_reason(tmp_path):
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN)], StubResearcher(fail=True), creator_replies=())
    wf.run("run-test", "topic")
    records = read_trace(tmp_path / "run" / "trace.jsonl")
    failed_step = next(r for r in records if r["event"] == "step_failed")
    assert failed_step["metadata"]["step"] == "research"
    assert failed_step["error"] == {"type": "RuntimeError", "message": "browser crashed"}
    final = records[-1]
    assert final["event"] == "run_failed" and final["status"] == "error"
    assert final["error"]["message"] == "research: RuntimeError: browser crashed"


def test_stopped_run_and_round_limit_override_are_traced(tmp_path):
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(STOP)], creator_replies=())
    wf.run("run-test", "topic")
    final = read_trace(tmp_path / "run" / "trace.jsonl")[-1]
    assert final["event"] == "run_stopped" and "Evidence too thin" in final["error"]["message"]

    more_again = {**MORE, "follow_up_tasks": [{"id": "T9", "question": "q", "evidence_required": "e"}]}
    wf, _, _ = make_workflow(tmp_path / "b", [json_reply(PLAN), json_reply(MORE), json_reply(more_again)],
                             pdf_writer=lambda html, path: path.write_bytes(b"x"))
    wf.run("run-test", "topic")
    override = next(r for r in read_trace(tmp_path / "b" / "run" / "trace.jsonl") if r["event"] == "round_limit_override")
    assert override["metadata"] == {"original_outcome": "RESEARCH_MORE", "new_outcome": "ACCEPT",
                                    "reason": "round limit reached and deterministic checks passed",
                                    "excluded_task_ids": []}


def test_malformed_response_is_traced(tmp_path):
    wf, _, _ = make_workflow(tmp_path, [text_reply("nope"), text_reply("still nope")], creator_replies=())
    wf.run("run-test", "topic")
    records = read_trace(tmp_path / "run" / "trace.jsonl")
    malformed = [r for r in records if r["event"] == "malformed_response"]
    assert [r["status"] for r in malformed] == ["warning", "error"]
    assert malformed[0]["metadata"]["repair_requested"] is True


# ---- researcher events ----

def test_researcher_events(tmp_path):
    bad = "https://www.ofgem.gov.uk/missing"
    tools = Tools(pages={bad: PageContent(url=bad, ok=False, http_status=404, error="HTTP 404")})
    trace = Trace(tmp_path / "t.jsonl", "run-1")
    fake = FakeAnthropic([
        tool_reply(("web_search", {"query": "standing charges"}), ("web_search", {"query": "standing charges"})),
        tool_reply(("open_page", {"url": URL}), ("open_page", {"url": bad})),
        json_reply(submit()[1]),  # turn cap of 2 reached: structured final submission
    ])
    session = AgentSession("researcher", "p")
    r = Researcher(session, make_llm(fake, trace=trace, task_max_model_turns=2), make_settings(task_max_model_turns=2),
                   tools.search, tools.open)
    r.run_tasks([ResearchTask(id="T1", question="What are standing charges?", evidence_required="e")])
    records = read_trace(tmp_path / "t.jsonl")
    names = events(records)
    for expected in ("task_dispatched", "search_performed", "search_blocked", "page_opened", "page_failed",
                     "source_registered", "cap_reached", "findings_extracted", "forced_finish", "task_returned"):
        assert expected in names, expected
    search = next(r for r in records if r["event"] == "search_performed")
    assert search["metadata"]["results"] == 1 and search["metadata"]["urls"] == [URL]
    assert search["agent"] == "researcher" and search["session_id"] == session.session_id and search["task_id"] == "T1"
    failed = next(r for r in records if r["event"] == "page_failed")
    assert failed["metadata"]["http_status"] == 404 and failed["error"]["message"] == "HTTP 404"
    assert next(r for r in records if r["event"] == "cap_reached")["metadata"]["limit"] == "turn limit"
    extracted = next(r for r in records if r["event"] == "findings_extracted")
    assert extracted["metadata"]["source_ids"] == ["S1"]
    llm = [r for r in records if r["event"] == "llm_call"]
    assert [r["metadata"]["mode"] for r in llm] == ["tools", "tools", "structured_output"]


# ---- CLI ----

def test_cli_missing_config_is_a_clear_error(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # no .env here
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert cli.main(["run", "--topic", "x"]) == 2
    err = capsys.readouterr().err
    assert "Missing required environment variables" in err and "ANTHROPIC_API_KEY" in err


def test_cli_progress_lines():
    record = {"event": "review_decision", "metadata": {"round": 1, "outcome": "ACCEPT", "rationale": "Good coverage."},
              "error": None, "task_id": None}
    assert cli.progress_line(record) == "Review round 1: ACCEPT - Good coverage."
    assert cli.progress_line({"event": "llm_call", "metadata": {}, "error": None, "task_id": None}) is None
