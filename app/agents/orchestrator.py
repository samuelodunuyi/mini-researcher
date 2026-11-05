"""Orchestrator agent: plans research tasks and reviews research results."""

from app.llm import LLMClient
from app.schemas import ResearchPlan, ResearchResult, ReviewDecision
from app.sessions import AgentSession


class Orchestrator:
    """One session is used for both planning and every review in a run.

    The task range, source target and follow-up limit come from the run's depth preset.
    """

    def __init__(
        self,
        session: AgentSession,
        llm: LLMClient,
        min_plan_tasks: int = 4,
        max_plan_tasks: int = 5,
        source_target: str = "8-15",
        max_follow_up_tasks: int = 3,
    ):
        self.session = session
        self.llm = llm
        self.min_plan_tasks = min_plan_tasks
        self.max_plan_tasks = max_plan_tasks
        self.source_target = source_target
        self.max_follow_up_tasks = max_follow_up_tasks

    def create_plan(self, topic: str) -> ResearchPlan:
        low, high = self.min_plan_tasks, self.max_plan_tasks

        def out_of_range(plan: ResearchPlan) -> list[str]:
            n = len(plan.tasks)
            return [] if low <= n <= high else [f"the plan has {n} tasks; it must have between {low} and {high}"]

        plan = self.llm.call_structured(
            self.session,
            f"Create a research plan for this topic with {low} to {high} tasks. Across the whole run, aim for "
            f"about {self.source_target} strong sources, reused across tasks where they fit.\nTopic: {topic}",
            ResearchPlan,
            check=out_of_range,
        )
        return plan.model_copy(update={"topic": topic})

    def review(self, result: ResearchResult, check_report: list[str], research_more_allowed: bool) -> ReviewDecision:
        allowance = (
            "More research is allowed: you may choose RESEARCH_MORE with at most "
            f"{self.max_follow_up_tasks} targeted follow-up tasks."
            if research_more_allowed
            else "No more research rounds are allowed: choose ACCEPT or STOP_INSUFFICIENT."
        )
        checks = "\n".join(f"- {p}" for p in check_report) or "- No problems found."
        return self.llm.call_structured(
            self.session,
            "Review these research results against your plan.\n\n"
            f"Automated checks:\n{checks}\n\n{allowance}\n\n"
            f"Research results (page text omitted):\n{summarise_result(result)}",
            ReviewDecision,
        )


    def reconsider_accept(self, problems: list[str]) -> ReviewDecision:
        """One repair after the application refuses an ACCEPT while blocking problems remain."""
        listed = "\n".join(f"- {p}" for p in problems)
        return self.llm.call_structured(
            self.session,
            "The application cannot accept these results while these automated checks fail:\n"
            f"{listed}\n\nChoose RESEARCH_MORE with at most {self.max_follow_up_tasks} follow-up tasks aimed at the "
            "failing tasks, or STOP_INSUFFICIENT if more research won't fix them.",
            ReviewDecision,
        )


def summarise_result(result: ResearchResult) -> str:
    """The orchestrator sees findings and source metadata, not full page text."""
    return result.model_dump_json(
        indent=1, exclude={"sources": {"__all__": {"extracted_content"}}}
    )
