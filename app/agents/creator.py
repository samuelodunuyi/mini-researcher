"""Creator agent: turns approved findings into structured report content."""

import json

from app.llm import LLMClient
from app.schemas import ApprovedResearch, ReportDocument, validate_citations
from app.sessions import AgentSession


class Creator:
    def __init__(self, session: AgentSession, llm: LLMClient):
        self.session = session
        self.llm = llm

    def write_report(self, approved: ApprovedResearch) -> ReportDocument:
        source_list = [
            {
                "source_id": s.source_id,
                "title": s.title,
                "publisher": s.publisher,
                "url": str(s.url),
                "dates": s.published_date,
            }
            for s in approved.sources
        ]
        findings_json = json.dumps([f.model_dump() for f in approved.findings], indent=1)
        excluded = ""
        if approved.excluded_tasks:
            questions = "\n".join(f"- {t.question}" for t in approved.excluded_tasks)
            excluded = (
                "These questions were excluded because the evidence was insufficient. Do not write about them; "
                f"the application adds its own limitations section listing them:\n{questions}\n\n"
            )
        caveats = ""
        if approved.reviewer_caveats:
            listed = "\n".join(f"- {c}" for c in approved.reviewer_caveats)
            caveats = (
                "The reviewer accepted this research with these caveats. Follow them: state each gap or uncertainty "
                "as a limitation in the relevant section, citing the sources it concerns; where one finding supersedes "
                f"another, use the newer one:\n{listed}\n\n"
            )
        content = (
            f"Topic: {approved.topic}\n\n"
            f"Approved findings:\n{findings_json}\n\n"
            f"Sources you may cite:\n{json.dumps(source_list, indent=1)}\n\n"
            f"{excluded}{caveats}Write the report content."
        )
        doc = self.llm.call_structured(
            self.session,
            content,
            ReportDocument,
            check=lambda d: validate_citations(d, approved.sources),
        )
        return doc.model_copy(update={"topic": approved.topic})
