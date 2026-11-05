"""Deterministic research checks run by Python before the orchestrator reviews a round."""

from pydantic import BaseModel

from app.schemas import ResearchResult, ResearchTask, RetrievalStatus


class CheckReport(BaseModel):
    problems: list[str] = []  # blocking: an ACCEPT is refused or narrowed while any exist
    notes: list[str] = []  # informational: failed pages, researcher caveats
    failing_tasks: dict[str, str] = {}  # task id -> reason, for the tasks behind the problems
    no_sources_read: bool = False

    @property
    def passed(self) -> bool:
        return not self.problems

    def lines(self) -> list[str]:
        return [f"PROBLEM: {p}" for p in self.problems] + [f"NOTE: {n}" for n in self.notes]


def check_research(tasks: list[ResearchTask], result: ResearchResult) -> CheckReport:
    """Structural checks; schema validation has already rejected unknown or unread source IDs."""
    report = CheckReport()
    status = {s.source_id: s.retrieval_status for s in result.sources}
    read = [sid for sid, st in status.items() if st is RetrievalStatus.READ]
    if not read:
        report.no_sources_read = True
        report.problems.append("No source pages were read successfully")

    # Follow-up attribution: a follow-up's findings also count towards the original task it
    # `addresses`, so a gap filled in round 2 is not reported against the original task. Originals
    # are judged on their own findings plus those of their follow-ups. A follow-up is judged on its
    # own findings, except that a source shortfall is forgiven once the task it addresses passes;
    # a follow-up with no findings at all always fails.
    ids = {t.id for t in tasks}
    originals = [t for t in tasks if t.addresses not in ids]
    follow_ups = [t for t in tasks if t.addresses in ids]

    def evaluate(task: ResearchTask, contributor_ids: set[str]) -> str | None:
        findings = [f for f in result.findings if f.task_id in contributor_ids]
        sources = {sid for f in findings for sid in f.source_ids}
        if not findings:
            return "no findings"
        if len(sources) < task.min_sources:
            return f"supported by {len(sources)} source(s); the plan asked for at least {task.min_sources}"
        return None

    for task in originals:
        reason = evaluate(task, {task.id} | {f.id for f in follow_ups if f.addresses == task.id})
        if reason:
            report.failing_tasks[task.id] = reason
    for task in follow_ups:
        reason = evaluate(task, {task.id})
        if reason and reason != "no findings" and task.addresses not in report.failing_tasks:
            reason = None
        if reason:
            report.failing_tasks[task.id] = reason
    for task in tasks:  # keep the problems in plan order
        reason = report.failing_tasks.get(task.id)
        if reason:
            report.problems.append(f"Task {task.id} has {reason}" if reason == "no findings" else f"Task {task.id} is {reason}")

    for s in result.sources:
        if s.retrieval_status is RetrievalStatus.FAILED:
            report.notes.append(f"{s.source_id} ({s.url}) could not be read: {s.error}")
    for task_id in result.failed_task_ids:
        report.notes.append(f"Task {task_id} failed: the researcher gave no valid submission")
    if result.incomplete:
        report.notes.append("The researcher marked at least one task incomplete")
    if result.notes:
        report.notes.append(f"Researcher notes: {result.notes}")
    return report


def merge_results(previous: ResearchResult | None, new: ResearchResult) -> ResearchResult:
    """Combine rounds. Source IDs are unique for the run, so a later copy of a source wins."""
    if previous is None:
        return new
    sources = {s.source_id: s for s in previous.sources}
    sources.update({s.source_id: s for s in new.sources})
    return ResearchResult(
        task_ids=previous.task_ids + [t for t in new.task_ids if t not in previous.task_ids],
        sources=sorted(sources.values(), key=lambda s: int(s.source_id[1:])),
        findings=previous.findings + new.findings,
        incomplete=previous.incomplete or new.incomplete,
        failed_task_ids=previous.failed_task_ids + new.failed_task_ids,
        notes="\n".join(n for n in (previous.notes, new.notes) if n),
    )
