"""Command line entry point: python -m app.cli run --topic "..." """

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone

from app.agents.creator import Creator
from app.agents.orchestrator import Orchestrator
from app.agents.researcher import Researcher
from app.config import DEPTH_PRESETS, ConfigError, load_settings
from app.llm import LLMClient
from app.sessions import SessionStore
from app.tools.browser import Browser
from app.tools.pdf import write_pdf
from app.tools.search import TavilySearch
from app.trace import Trace
from app.workflow import Workflow


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def progress_line(record: dict) -> str | None:
    """A short human-readable line for the major events; None for the rest."""
    m, e = record["metadata"], record["event"]
    lines = {
        "run_started": lambda: f"Run started ({m.get('depth', 'standard')} depth): {m['topic']}",
        "plan_created": lambda: f"Plan: {len(m['tasks'])} research tasks ({', '.join(t['id'] for t in m['tasks'])})",
        "research_round_started": lambda: f"Research round {m['round']}: tasks {', '.join(m['task_ids'])}",
        "task_dispatched": lambda: f"  {record['task_id']}: {m['question'][:90]}",
        "search_performed": lambda: f"    search: {m.get('query', '')!r} -> {m.get('results', 'error')} results",
        "page_opened": lambda: f"    read: {m['url']} ({m['chars']} chars)",
        "page_failed": lambda: f"    failed: {m['url']} ({(record['error'] or {}).get('message', '')})",
        "cap_reached": lambda: f"    limit reached: {m['limit']}",
        "task_returned": lambda: f"  {record['task_id']} done: {m['findings']} findings{' (incomplete)' if m['incomplete'] else ''}",
        "checks_completed": lambda: f"Checks: {'passed' if m['passed'] else '; '.join(m['blocking_issues'])}",
        "review_decision": lambda: f"Review round {m['round']}: {m['outcome']} - {m['rationale'][:160]}",
        "round_limit_override": lambda: f"Override: {m['original_outcome']} -> {m['new_outcome']} ({m['reason']})",
        "creator_invoked": lambda: "Writing report content",
        "citation_validation": lambda: f"Citations: {'valid' if not m['problems'] else '; '.join(m['problems'])}",
        "pdf_verified": lambda: f"PDF verified: {m['path']}" if not m["problems"] else f"PDF invalid: {m['problems']}",
        "step_failed": lambda: f"FAILED at {m['step']}: {(record['error'] or {}).get('message', '')}",
    }
    return lines[e]() if e in lines else None


def run(topic: str, depth: str | None = None) -> int:
    try:
        settings = load_settings(depth=depth)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    run_id = new_run_id()
    run_dir = settings.runs_dir / run_id
    started = time.monotonic()

    def echo(record: dict) -> None:
        line = progress_line(record)
        if line:
            print(f"[{time.monotonic() - started:6.0f}s] {line}", flush=True)

    trace = Trace(run_dir / "trace.jsonl", run_id, secrets=(settings.anthropic_api_key, settings.tavily_api_key), echo=echo)
    store = SessionStore(run_id)
    for session in store.all():
        trace.emit("session_created", agent=session.agent_name, session_id=session.session_id)
    llm = LLMClient(settings, trace=trace)
    search = TavilySearch(settings.tavily_api_key, settings.search_max_results)

    try:
        with Browser(settings) as browser:
            workflow = Workflow(
                settings,
                Orchestrator(store.get("orchestrator"), llm, settings.min_plan_tasks, settings.max_plan_tasks,
                             settings.source_target, settings.max_follow_up_tasks),
                Researcher(store.get("researcher"), llm, settings, search, browser.open_page),
                Creator(store.get("creator"), llm),
                lambda html, path: write_pdf(browser.context, html, path),
                run_dir,
                trace=trace,
            )
            state = workflow.run(run_id, topic)
    except Exception as exc:  # e.g. Chromium missing; the workflow itself never raises
        trace.emit("run_failed", status="error", error=exc, stage="startup")
        print(f"Run failed before the workflow could start: {exc}", file=sys.stderr)
        return 1

    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    print()
    print(f"Status: {summary['status'].upper()} (depth: {summary['depth']})"
          + (f" - {summary['error']}" if summary["error"] else ""))
    print(f"Rounds: {summary['research_rounds']} | findings: {summary['findings']} | sources read: "
          f"{summary['sources_read']}, failed: {summary['sources_failed']} | duration: {summary['duration_s']}s | "
          f"tokens in/out: {summary['tokens']['total']['total_input_tokens']}/{summary['tokens']['total']['output_tokens']} "
          f"({summary['tokens']['total']['cache_read_tokens']} input tokens read from cache)")
    if summary["pdf"]:
        print(f"PDF:   {summary['pdf']}")
    print(f"Trace: {run_dir / 'trace.jsonl'}")
    print(f"Run folder: {run_dir}")
    return 0 if state.get("status") == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Multi-agent research report generator")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run the full research workflow for a topic")
    run_parser.add_argument("--topic", required=True)
    run_parser.add_argument(
        "--depth", choices=list(DEPTH_PRESETS),
        help="research depth preset (default: RESEARCH_DEPTH, else standard): "
             "quick = 1 round, 3-4 tasks; standard = up to 2 rounds, 4-5 tasks; deep = up to 2 rounds, 6-8 tasks",
    )
    run_parser.add_argument("--verbose", action="store_true", help="also show library log messages")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args.topic, args.depth)


if __name__ == "__main__":
    sys.exit(main())
