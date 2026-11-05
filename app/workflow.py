"""LangGraph workflow: plan -> research -> checks -> review -> (research more | create | stop).

Graph state holds coordination data and validated hand-off objects only. Agent conversation
histories live in each agent's own session and never enter the state. Every step and hand-off
is recorded in the run's JSONL trace.
"""

import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.agents.creator import Creator
from app.agents.orchestrator import Orchestrator
from app.agents.researcher import Researcher
from app.checks import check_research, merge_results
from app.config import Settings
from app.schemas import (
    ApprovedResearch,
    ExcludedTask,
    ReportDocument,
    ResearchPlan,
    ResearchResult,
    ResearchTask,
    RetrievalStatus,
    ReviewDecision,
    ReviewOutcome,
    validate_citations,
)
from app.tools.pdf import render_html, verify_pdf
from app.trace import Trace, null_trace

log = logging.getLogger(__name__)


class RunState(TypedDict, total=False):
    run_id: str
    topic: str
    status: str  # running | completed | stopped | failed
    error: str
    round: int
    plan: ResearchPlan
    tasks: list[ResearchTask]  # every task issued so far (plan + follow-ups)
    pending_tasks: list[ResearchTask]  # tasks for the next research round
    research: ResearchResult  # merged across rounds
    checks: list[str]
    checks_passed: bool
    failing_tasks: dict[str, str]  # task id -> reason, from the latest checks
    no_sources_read: bool
    excluded_tasks: list[ExcludedTask]  # tasks dropped from the report by the drop rule
    decisions: list[ReviewDecision]
    approved: ApprovedResearch
    report: ReportDocument
    html_path: str
    pdf_path: str


class Workflow:
    def __init__(
        self,
        settings: Settings,
        orchestrator: Orchestrator,
        researcher: Researcher,
        creator: Creator,
        pdf_writer: Callable[[str, Path], None],
        run_dir: Path,
        trace: Trace | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.settings = settings
        self.orchestrator = orchestrator
        self.researcher = researcher
        self.creator = creator
        self.pdf_writer = pdf_writer
        self.run_dir = run_dir
        self.trace = trace or null_trace()
        self.clock = clock
        self._deadline = 0.0
        self._started = 0.0
        self.graph = self._build()

    def run(self, run_id: str, topic: str) -> RunState:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._started = self.clock()
        self._deadline = self._started + self.settings.run_timeout_s
        s = self.settings
        self.trace.emit(
            "run_started", topic=topic, model=s.anthropic_model, depth=s.depth, run_dir=str(self.run_dir),
            limits={"plan_tasks": f"{s.min_plan_tasks}-{s.max_plan_tasks}", "max_research_rounds": s.max_research_rounds,
                    "max_follow_up_tasks": s.max_follow_up_tasks, "task_max_searches": s.task_max_searches,
                    "task_max_page_opens": s.task_max_page_opens, "source_target": s.source_target},
        )
        initial: RunState = {"run_id": run_id, "topic": topic, "status": "running", "round": 0, "decisions": []}
        state = cast(RunState, self.graph.invoke(initial))  # invoke() is typed as returning a plain dict
        summary = summarise(state, self.trace.records, (self.clock() - self._started), depth=s.depth)
        final_event = {"completed": "run_completed", "stopped": "run_stopped"}.get(state.get("status", ""), "run_failed")
        self.trace.emit(
            final_event,
            status={"run_completed": "ok", "run_stopped": "stopped"}.get(final_event, "error"),
            duration_ms=summary["duration_s"] * 1000,
            error=state.get("error"),
            **{k: summary[k] for k in ("research_rounds", "findings", "sources_read", "sources_failed", "tokens", "pdf")},
        )
        self._save("run_summary.json", summary)
        return state

    # ---- graph ----

    def _build(self):
        g = StateGraph(RunState)
        steps = ("plan", "research", "checks", "review", "create", "validate_citations",
                 "render_html", "generate_pdf", "verify_pdf", "stop")
        for name in steps:
            g.add_node(name, self._guarded(name, getattr(self, f"_{name}")))
        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", self._ok_then("research"))
        g.add_conditional_edges("research", self._ok_then("checks"))
        g.add_conditional_edges("checks", self._ok_then("review"))
        g.add_conditional_edges("review", self._route_review)
        g.add_conditional_edges("create", self._ok_then("validate_citations"))
        g.add_conditional_edges("validate_citations", self._ok_then("render_html"))
        g.add_conditional_edges("render_html", self._ok_then("generate_pdf"))
        g.add_conditional_edges("generate_pdf", self._ok_then("verify_pdf"))
        g.add_edge("verify_pdf", END)
        g.add_edge("stop", END)
        return g.compile()

    def _guarded(self, name: str, fn):
        """Any exception or timeout marks the run failed with the reason; nothing continues silently."""

        def node(state: RunState) -> dict:
            if self.clock() > self._deadline:
                reason = f"run timed out before step '{name}'"
                self.trace.emit("step_failed", status="error", step=name, error=reason)
                return {"status": "failed", "error": reason}
            started = self.clock()
            try:
                update = fn(state)
            except Exception as exc:
                log.exception("step %s failed", name)
                self.trace.emit("step_failed", status="error", step=name, error=exc,
                                duration_ms=(self.clock() - started) * 1000)
                return {"status": "failed", "error": f"{name}: {type(exc).__name__}: {exc}"}
            self.trace.emit("step_finished", step=name, duration_ms=(self.clock() - started) * 1000)
            return update

        return node

    @staticmethod
    def _ok_then(next_node: str):
        return lambda state: END if state.get("status") == "failed" else next_node

    @staticmethod
    def _route_review(state: RunState) -> str:
        if state.get("status") == "failed":
            return END
        outcome = state["decisions"][-1].outcome
        return {
            ReviewOutcome.ACCEPT: "create",
            ReviewOutcome.RESEARCH_MORE: "research",
            ReviewOutcome.STOP_INSUFFICIENT: "stop",
        }[outcome]

    def _agent(self, name: str) -> dict:
        session = getattr(self, name).session
        return {"agent": session.agent_name, "session_id": session.session_id}

    # ---- nodes ----

    def _plan(self, state: RunState) -> dict:
        plan = self.orchestrator.create_plan(state["topic"])
        self._save("plan.json", plan)
        self.trace.emit("plan_created", **self._agent("orchestrator"), handoff_id=self.trace.new_handoff("plan"),
                        tasks=[{"id": t.id, "question": t.question, "min_sources": t.min_sources} for t in plan.tasks])
        return {"plan": plan, "tasks": list(plan.tasks), "pending_tasks": list(plan.tasks)}

    def _research(self, state: RunState) -> dict:
        round_no = state["round"] + 1
        tasks = state["pending_tasks"]
        self.trace.emit("research_round_started", **self._agent("researcher"), round=round_no,
                        task_ids=[t.id for t in tasks], follow_up=round_no > 1)
        result = self.researcher.run_tasks(tasks)
        self._save(f"research_round_{round_no}.json", result)
        self.trace.emit(
            "research_returned", **self._agent("researcher"), handoff_id=self.trace.new_handoff(f"research-round-{round_no}"),
            status="warning" if result.incomplete else "ok", round=round_no, **source_counts(result),
            findings=len(result.findings), failed_task_ids=result.failed_task_ids, incomplete=result.incomplete,
        )
        return {"round": round_no, "research": merge_results(state.get("research"), result), "pending_tasks": []}

    def _checks(self, state: RunState) -> dict:
        report = check_research(state["tasks"], state["research"])
        self._save(f"checks_round_{state['round']}.json", report)
        self.trace.emit("checks_completed", status="ok" if report.passed else "warning", round=state["round"],
                        passed=report.passed, blocking_issues=report.problems, notes=report.notes)
        return {"checks": report.lines(), "checks_passed": report.passed, "failing_tasks": report.failing_tasks,
                "no_sources_read": report.no_sources_read}

    def _review(self, state: RunState) -> dict:
        more_allowed = state["round"] < self.settings.max_research_rounds and self.settings.max_follow_up_tasks > 0
        decision = self.orchestrator.review(state["research"], state["checks"], more_allowed)
        if decision.outcome is ReviewOutcome.ACCEPT and not state["checks_passed"] and more_allowed:
            # Blocking problems and research still possible: refuse the ACCEPT once.
            problems = [line.removeprefix("PROBLEM: ") for line in state["checks"] if line.startswith("PROBLEM: ")]
            self.trace.emit("accept_refused", **self._agent("orchestrator"), status="warning", round=state["round"],
                            blocking_issues=problems)
            decision = self.orchestrator.reconsider_accept(problems)
        decision, excluded = self._enforce_rules(state, decision, more_allowed)
        self._save(f"review_round_{state['round']}.json", decision)
        self.trace.emit(
            "review_decision", **self._agent("orchestrator"), handoff_id=self.trace.new_handoff(f"review-round-{state['round']}"),
            round=state["round"], outcome=decision.outcome.value, rationale=decision.rationale[:600],
            follow_up_task_ids=[t.id for t in decision.follow_up_tasks], more_research_allowed=more_allowed,
        )
        update: dict = {"decisions": state["decisions"] + [decision], "excluded_tasks": excluded}
        if decision.outcome is ReviewOutcome.RESEARCH_MORE:
            known = {t.id for t in state["tasks"]}
            follow_ups = [t for t in decision.follow_up_tasks if t.id not in known]
            if not follow_ups:
                raise ValueError("RESEARCH_MORE gave no follow-up tasks with new IDs")
            limit = self.settings.max_follow_up_tasks
            if len(follow_ups) > limit:
                self.trace.emit("followup_trimmed", status="warning", requested=[t.id for t in follow_ups],
                                kept=[t.id for t in follow_ups[:limit]], limit=limit)
                follow_ups = follow_ups[:limit]
            # `addresses` must name an existing task; anything else is cleared (logged), not fatal.
            for i, t in enumerate(follow_ups):
                if t.addresses is not None and t.addresses not in known:
                    self.trace.emit("followup_addresses_invalid", status="warning", task_id=t.id, addresses=t.addresses)
                    follow_ups[i] = t.model_copy(update={"addresses": None})
            self.trace.emit("research_more_requested", **self._agent("orchestrator"),
                            tasks=[{"id": t.id, "question": t.question, "addresses": t.addresses} for t in follow_ups])
            update |= {"tasks": state["tasks"] + follow_ups, "pending_tasks": follow_ups}
        return update

    def _enforce_rules(
        self, state: RunState, decision: ReviewDecision, more_allowed: bool
    ) -> tuple[ReviewDecision, list[ExcludedTask]]:
        """Python, not the model, enforces the round limit and the minimum evidence for ACCEPT.

        Returns the final decision and any tasks excluded from the report by the drop rule.
        """
        excluded: list[ExcludedTask] = []
        override = None
        if decision.outcome is ReviewOutcome.RESEARCH_MORE and not more_allowed:
            if state["checks_passed"]:
                override = ReviewOutcome.ACCEPT, "round limit reached and deterministic checks passed"
            else:
                outcome, why, excluded = self._drop_failing_tasks(state)
                override = outcome, f"round limit reached and deterministic checks failed: {why}"
        elif decision.outcome is ReviewOutcome.ACCEPT and not state["research"].findings:
            override = ReviewOutcome.STOP_INSUFFICIENT, "there are no findings"
        elif decision.outcome is ReviewOutcome.ACCEPT and not state["checks_passed"]:
            # Reached on the last round, or after the one repair still returned ACCEPT.
            outcome, why, excluded = self._drop_failing_tasks(state)
            override = outcome, f"ACCEPT with blocking problems: {why}"
        if override is None:
            return decision, []
        outcome, why = override
        self.trace.emit("round_limit_override" if why.startswith("round limit") else "decision_override",
                        status="warning", original_outcome=decision.outcome.value, new_outcome=outcome.value,
                        reason=why, excluded_task_ids=[e.task_id for e in excluded])
        new = ReviewDecision(outcome=outcome, rationale=f"[Overridden from {decision.outcome.value}: {why}] {decision.rationale}",
                             caveats=decision.caveats if outcome is ReviewOutcome.ACCEPT else [])
        return new, excluded

    @staticmethod
    def _drop_failing_tasks(state: RunState) -> tuple[ReviewOutcome, str, list[ExcludedTask]]:
        """Drop rule: report only the tasks that passed the checks, or stop if none did."""
        if state["no_sources_read"]:
            return ReviewOutcome.STOP_INSUFFICIENT, "no source pages were read successfully", []
        failing = state["failing_tasks"]
        if all(t.id in failing for t in state["tasks"]):
            return ReviewOutcome.STOP_INSUFFICIENT, "no task passed the deterministic checks", []
        excluded = [ExcludedTask(task_id=t.id, question=t.question, reason=failing[t.id])
                    for t in state["tasks"] if t.id in failing]
        return ReviewOutcome.ACCEPT, f"excluded {', '.join(e.task_id for e in excluded)} from the report", excluded

    def _create(self, state: RunState) -> dict:
        approved = ApprovedResearch.from_result(state["topic"], state["research"], state.get("excluded_tasks", []),
                                                caveats=state["decisions"][-1].caveats)
        self._save("approved_research.json", approved)
        handoff = self.trace.new_handoff("approved-research")
        self.trace.emit("research_approved", handoff_id=handoff, findings=len(approved.findings),
                        sources=[s.source_id for s in approved.sources],
                        excluded_task_ids=[e.task_id for e in approved.excluded_tasks],
                        reviewer_caveats=len(approved.reviewer_caveats))
        self.trace.emit("creator_invoked", **self._agent("creator"), handoff_id=handoff)
        report = self.creator.write_report(approved)
        self._save("report_content.json", report)
        self.trace.emit(
            "report_returned", **self._agent("creator"), handoff_id=self.trace.new_handoff("report-content"),
            title=report.title, sections=len(report.sections),
            paragraphs=sum(len(s.paragraphs) for s in report.sections),
            cited_source_ids=sorted(report.cited_source_ids(), key=lambda s: int(s[1:])),
        )
        return {"approved": approved, "report": report}

    def _validate_citations(self, state: RunState) -> dict:
        # The creator already had one repair attempt; this is the application's own final check.
        problems = validate_citations(state["report"], state["approved"].sources)
        self.trace.emit("citation_validation", status="error" if problems else "ok", problems=problems,
                        cited=len(state["report"].cited_source_ids()))
        if problems:
            raise ValueError("citation validation failed: " + "; ".join(problems))
        return {}

    def _render_html(self, state: RunState) -> dict:
        html = render_html(state["report"], state["approved"].sources, datetime.now(timezone.utc), state["run_id"],
                           excluded_tasks=state["approved"].excluded_tasks)
        path = self.run_dir / "report.html"
        path.write_text(html, encoding="utf-8")
        self.trace.emit("html_rendered", path=str(path), bytes=len(html.encode("utf-8")))
        return {"html_path": str(path)}

    def _generate_pdf(self, state: RunState) -> dict:
        path = self.run_dir / "report.pdf"
        self.pdf_writer(Path(state["html_path"]).read_text(encoding="utf-8"), path)
        self.trace.emit("pdf_generated", path=str(path), bytes=path.stat().st_size if path.exists() else 0)
        return {"pdf_path": str(path)}

    def _verify_pdf(self, state: RunState) -> dict:
        problems = verify_pdf(Path(state["pdf_path"]), state["report"])
        self.trace.emit("pdf_verified", status="error" if problems else "ok", problems=problems, path=state["pdf_path"])
        if problems:
            raise ValueError("PDF verification failed: " + "; ".join(problems))
        return {"status": "completed"}

    def _stop(self, state: RunState) -> dict:
        return {"status": "stopped", "error": f"STOP_INSUFFICIENT: {state['decisions'][-1].rationale}"}

    # ---- artefacts ----

    def _save(self, name: str, obj) -> None:
        data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
        (self.run_dir / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def source_counts(result: ResearchResult) -> dict:
    by_status = defaultdict(int)
    for s in result.sources:
        by_status[s.retrieval_status] += 1
    return {
        "sources_read": by_status[RetrievalStatus.READ],
        "sources_failed": by_status[RetrievalStatus.FAILED],
        "sources_irrelevant": by_status[RetrievalStatus.IRRELEVANT],
    }


def summarise(state: RunState, records: list[dict] | None = None, duration_s: float = 0.0, depth: str | None = None) -> dict:
    """Run totals for run_summary.json; the token and timing figures come from the trace."""
    records = records or []
    research = state.get("research")
    # The API reports uncached input, cache writes and cache reads separately; all three are input
    # the model processed, so total_input_tokens adds them up.
    fields = ("input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens")
    tokens: dict[str, dict] = defaultdict(lambda: {"calls": 0, **{f: 0 for f in fields}, "total_input_tokens": 0})
    stages: dict[str, float] = defaultdict(float)
    for r in records:
        m = r["metadata"]
        if r["event"] == "llm_call":
            t = tokens[r["agent"]]
            t["calls"] += 1
            for f in fields:
                t[f] += m.get(f) or 0
            t["total_input_tokens"] = t["input_tokens"] + t["cache_write_tokens"] + t["cache_read_tokens"]
        elif r["event"] in ("step_finished", "step_failed") and r["duration_ms"] is not None:
            stages[m["step"]] += r["duration_ms"] / 1000
    totals = {k: sum(t[k] for t in tokens.values()) for k in ("calls", *fields, "total_input_tokens")}
    return {
        "run_id": state.get("run_id"),
        "topic": state.get("topic"),
        "depth": depth,
        "status": state.get("status"),
        "error": state.get("error"),
        "duration_s": round(duration_s, 1),
        "research_rounds": state.get("round", 0),
        "decisions": [d.outcome.value for d in state.get("decisions", [])],
        "tasks": len(state.get("tasks", [])),
        "findings": len(research.findings) if research else 0,
        **(source_counts(research) if research else {"sources_read": 0, "sources_failed": 0, "sources_irrelevant": 0}),
        "failed_task_ids": research.failed_task_ids if research else [],
        "tokens": {"total": totals, "by_agent": dict(tokens)},
        "stage_seconds": {k: round(v, 1) for k, v in stages.items()},
        "decision_overrides": sum(r["event"] in ("round_limit_override", "decision_override") for r in records),
        "accept_refusals": sum(r["event"] == "accept_refused" for r in records),
        "excluded_tasks": [e.model_dump() for e in state.get("excluded_tasks", [])],
        "pdf": state.get("pdf_path") if state.get("status") == "completed" else None,
    }
