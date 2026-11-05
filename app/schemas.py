"""Pydantic models for every hand-off between agents."""

import re
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

SOURCE_ID_PATTERN = r"^S\d+$"
CITATION_RE = re.compile(r"\[(S\d+)\]")


class ResearchTask(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    evidence_required: str = Field(min_length=1)
    min_sources: int = Field(default=1, ge=1)
    preferred_source_types: list[str] = Field(default_factory=list)
    # Follow-up tasks only: the ID of the original task this one fills in.
    addresses: str | None = None


class ResearchPlan(BaseModel):
    topic: str = Field(min_length=1)
    tasks: list[ResearchTask] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_task_ids(self) -> "ResearchPlan":
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task IDs must be unique")
        return self


class RetrievalStatus(str, Enum):
    READ = "read"
    FAILED = "failed"
    IRRELEVANT = "irrelevant"


class ResearchSource(BaseModel):
    source_id: str = Field(pattern=SOURCE_ID_PATTERN)
    title: str = Field(min_length=1)
    publisher: str | None = None
    url: HttpUrl
    published_date: str | None = None
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retrieval_status: RetrievalStatus
    http_status: int | None = None
    extracted_content: str = ""
    # True when extracted_content is a selection from a longer page; full_chars is the page's full length.
    truncated: bool = False
    full_chars: int | None = None
    error: str | None = None

    @field_validator("url")
    @classmethod
    def http_or_https(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme not in ("http", "https"):
            raise ValueError("URL must use http or https")
        return v

    @model_validator(mode="after")
    def read_sources_have_content(self) -> "ResearchSource":
        if self.retrieval_status is RetrievalStatus.READ and not self.extracted_content.strip():
            raise ValueError(f"{self.source_id} is marked read but has no extracted content")
        return self


class SearchHit(BaseModel):
    """A search-engine candidate. Never evidence by itself."""

    title: str
    url: str
    snippet: str = ""
    score: float | None = None


class PageContent(BaseModel):
    """What the browser returned for one URL."""

    url: str
    ok: bool
    final_url: str | None = None
    content_type: str = "html"  # html | pdf
    title: str = ""
    text: str = ""
    http_status: int | None = None
    published_date: str | None = None
    publisher: str | None = None
    error: str | None = None


class ResearchFinding(BaseModel):
    id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    limitations: str = ""


class ResearchResult(BaseModel):
    task_ids: list[str] = Field(min_length=1)
    sources: list[ResearchSource] = Field(default_factory=list)
    findings: list[ResearchFinding] = Field(default_factory=list)
    incomplete: bool = False
    failed_task_ids: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def findings_cite_read_sources(self) -> "ResearchResult":
        source_ids = [s.source_id for s in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must be unique")
        status = {s.source_id: s.retrieval_status for s in self.sources}
        for f in self.findings:
            if f.task_id not in self.task_ids:
                raise ValueError(f"finding {f.id} references unknown task {f.task_id}")
            for sid in f.source_ids:
                if sid not in status:
                    raise ValueError(f"finding {f.id} references unknown source {sid}")
                if status[sid] is not RetrievalStatus.READ:
                    raise ValueError(f"finding {f.id} cites {sid}, which was not successfully read")
        return self

    def read_sources(self) -> list[ResearchSource]:
        return [s for s in self.sources if s.retrieval_status is RetrievalStatus.READ]


class ExcludedTask(BaseModel):
    """A planned task left out of the report because it failed the deterministic checks."""

    task_id: str
    question: str
    reason: str


class ApprovedResearch(BaseModel):
    """The creator's only input: approved findings and the read sources they cite."""

    topic: str = Field(min_length=1)
    findings: list[ResearchFinding] = Field(min_length=1)
    sources: list[ResearchSource] = Field(min_length=1)
    # Written by the application, not the LLM; rendered as a fixed limitations section.
    excluded_tasks: list[ExcludedTask] = Field(default_factory=list)
    # The orchestrator's caveats from its ACCEPT: instructions for the creator, not printed verbatim.
    reviewer_caveats: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def findings_cite_included_sources(self) -> "ApprovedResearch":
        status = {s.source_id: s.retrieval_status for s in self.sources}
        for f in self.findings:
            for sid in f.source_ids:
                if status.get(sid) is not RetrievalStatus.READ:
                    raise ValueError(f"approved finding {f.id} cites {sid}, which is not an included read source")
        return self

    @classmethod
    def from_result(
        cls,
        topic: str,
        result: "ResearchResult",
        excluded: list[ExcludedTask] | None = None,
        caveats: list[str] | None = None,
    ) -> "ApprovedResearch":
        """Approved findings minus excluded tasks; only the read sources those findings cite are kept."""
        excluded = excluded or []
        dropped = {e.task_id for e in excluded}
        findings = [f for f in result.findings if f.task_id not in dropped]
        cited = {sid for f in findings for sid in f.source_ids}
        return cls(
            topic=topic,
            findings=findings,
            sources=[s for s in result.sources if s.source_id in cited],
            excluded_tasks=excluded,
            reviewer_caveats=caveats or [],
        )


class ReviewOutcome(str, Enum):
    ACCEPT = "ACCEPT"
    RESEARCH_MORE = "RESEARCH_MORE"
    STOP_INSUFFICIENT = "STOP_INSUFFICIENT"


class ReviewDecision(BaseModel):
    outcome: ReviewOutcome
    rationale: str = Field(min_length=1)
    follow_up_tasks: list[ResearchTask] = Field(default_factory=list)
    # On ACCEPT: gaps the report must state as limitations, and findings that supersede others.
    caveats: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def follow_up_matches_outcome(self) -> "ReviewDecision":
        if self.outcome is ReviewOutcome.RESEARCH_MORE and not self.follow_up_tasks:
            raise ValueError("RESEARCH_MORE requires at least one follow-up task")
        if self.outcome is not ReviewOutcome.RESEARCH_MORE and self.follow_up_tasks:
            raise ValueError(f"{self.outcome.value} must not include follow-up tasks")
        return self


class ReportParagraph(BaseModel):
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def inline_citations_declared(self) -> "ReportParagraph":
        undeclared = set(CITATION_RE.findall(self.text)) - set(self.source_ids)
        if undeclared:
            raise ValueError(f"inline citations not listed in source_ids: {sorted(undeclared)}")
        return self


class ReportSection(BaseModel):
    heading: str = Field(min_length=1)
    paragraphs: list[ReportParagraph] = Field(min_length=1)


class ReportDocument(BaseModel):
    title: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    introduction: str = Field(min_length=1)
    sections: list[ReportSection] = Field(min_length=1)

    def cited_source_ids(self) -> set[str]:
        cited = set(CITATION_RE.findall(self.introduction))
        for section in self.sections:
            for p in section.paragraphs:
                cited.update(p.source_ids)
        return cited


def validate_citations(document: ReportDocument, sources: list[ResearchSource]) -> list[str]:
    """Return a list of citation problems; an empty list means every citation is valid."""
    status = {s.source_id: s.retrieval_status for s in sources}
    problems = []
    for section in document.sections:
        for i, p in enumerate(section.paragraphs, start=1):
            unmarked = [sid for sid in p.source_ids if f"[{sid}]" not in p.text]
            if unmarked:
                problems.append(
                    f"paragraph {i} of '{section.heading}' lists {', '.join(unmarked)} in source_ids "
                    "but has no matching inline marker"
                )
    for sid in sorted(document.cited_source_ids(), key=lambda s: int(s[1:])):
        if sid not in status:
            problems.append(f"{sid} is cited but does not exist")
        elif status[sid] is not RetrievalStatus.READ:
            problems.append(f"{sid} is cited but was not successfully read")
    if problems:
        return problems

    # Every figure in a paragraph must be stated by a page that paragraph cites.
    by_id = {s.source_id: s for s in sources}
    # An uncited introduction is checked against no sources, so it may contain no figures at all.
    intro_ids = CITATION_RE.findall(document.introduction)
    paragraphs = [("the introduction", document.introduction, intro_ids)] + [
        (f"paragraph {i} of '{section.heading}'", p.text, p.source_ids)
        for section in document.sections
        for i, p in enumerate(section.paragraphs, start=1)
    ]
    for where, text, ids in paragraphs:
        missing = unsupported_numbers(text, [by_id[sid] for sid in ids if sid in by_id])
        if missing:
            problems.append(f"{where} states {', '.join(missing)}, which its cited pages do not contain")
    return problems


NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _normalise_figures(text: str) -> str:
    """Make figures comparable: drop thousands separators, unify dashes and spacing."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return " ".join(text.replace("–", "-").replace("—", "-").split())


def unsupported_numbers(text: str, sources: list[ResearchSource]) -> list[str]:
    """Numbers (2+ digits, amounts, percentages, years, decimals) in `text` that none of the sources state.

    Single digits are ignored: they are too common to check meaningfully. Currency signs and
    % are ignored by matching the bare figure; each source's dates count as part of its text.
    """
    pool = _normalise_figures(" ".join(f"{s.extracted_content} {s.published_date or ''}" for s in sources))
    found = set(NUMBER_RE.findall(pool))
    claimed = NUMBER_RE.findall(_normalise_figures(CITATION_RE.sub("", text)))
    return sorted({n for n in claimed if (len(n) > 1 or "." in n) and n not in found}, key=claimed.index)
