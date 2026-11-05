# Researcher
 
This is a small multi-agent research system. Give it a topic and three separate agents will plan the research, read real web pages in a browser and turn what they find into a referenced PDF report. The application also records each step in a JSONL execution trace.

```bash
python -m app.cli run --topic "What a UK business should know before signing a new electricity contract"
```

The sample submission is a real run of that command. You can find the PDF, execution trace and every agent hand-off in [`artifacts/sample_run/`](artifacts/sample_run/). There is also a shorter run using `--depth quick` in [`artifacts/sample_run_quick/`](artifacts/sample_run_quick/). The [`DESIGN_NOTE.md`](DESIGN_NOTE.md) explains the main design decisions, how the application handles failures and what the runs revealed about cost and performance.

## How it works

```text
CLI ─► LangGraph workflow (app/workflow.py)
        │
        ├─ Orchestrator: plan ──► Researcher: Tavily search + Playwright page reading ──► Python: checks
        │        ▲                        ▲                                                    │
        │        │                        └──── RESEARCH_MORE (2 rounds at most) ◄─────────────┤
        │        └─────────────────────────────── Orchestrator: review ◄────────────────────────┘
        │                                               │ ACCEPT                 │ STOP_INSUFFICIENT ─► stopped
        ▼                                               ▼
   Creator: write report ─► Python: citation check ─► Jinja2 HTML ─► Chromium PDF ─► Python: verify ─► completed
```

LangGraph keeps the information needed to coordinate the workflow, along with the validated objects passed between agents. It does not store their conversations. Each agent's conversation history stays separate.

### The three agents

| Agent                                           | What it does                                                                                                                                                                                                                                       | What it doesn't do                                                                                                                     |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Orchestrator** (`app/agents/orchestrator.py`) | Breaks the topic into research tasks. After each round, it reviews the findings and the results of the Python checks. It then decides whether to `ACCEPT`, request targeted follow-up work with `RESEARCH_MORE`, or stop with `STOP_INSUFFICIENT`. | It doesn't browse the web, write the report or make the PDF. It sees the findings and source details, but not the original page text.  |
| **Researcher** (`app/agents/researcher.py`)     | Runs a tool-use loop for each task. It searches with Tavily, opens pages in Playwright Chromium and returns findings with supporting evidence and source IDs.                                                                                      | It can't cite a page it hasn't opened. Python assigns source IDs only after the browser has successfully read a page.                  |
| **Creator** (`app/agents/creator.py`)           | Turns the approved findings into a structured report with an introduction and sections. Every paragraph in those sections refers to its sources using `[S#]` citations.                                                                            | It doesn't browse the web or add facts. It has no tools and only receives the approved findings and the sources it is allowed to cite. |

**How the sessions are kept separate.** Each run creates three sessions in `app/sessions.py`. Every session has its own ID, system prompt (in `app/prompts/*.md`) and conversation history. Agents pass validated Pydantic objects to one another, never conversation transcripts:

```text
ResearchPlan ─► ResearchTask ─► ResearchResult (findings + sources) ─► ReviewDecision ─► ApprovedResearch ─► ReportDocument
```

All three agents use the same Anthropic client. The client itself doesn't hold conversation history: each model call receives only the history for the agent making that call. The execution trace shows three different session IDs, and every model call is tagged with the ID of its agent.

### What the application enforces in code

Some decisions are made by the orchestrator, but the application also enforces rules in Python so the workflow doesn't depend on the model following every instruction perfectly.

- **No more than two research rounds,** including the first one.
- **Python checks the results after every round** (`app/checks.py`). It flags cases where:
  - no page was read
  - a task came back without any findings
  - a task has fewer distinct sources than the research plan requested
  Findings from a follow-up task count towards the original task it addresses.
- **The orchestrator can't accept results while these checks are failing.** If another round is available, it gets one chance to request repairs. On the final round, tasks that still fail are dropped and listed in a **Limitations of this report** section written by the application. If nothing useful remains, the run stops.
- **The creator receives the reviewer's caveats.** When the orchestrator accepts the research, it also identifies gaps that the report needs to mention and findings that have been superseded. The creator must include these in cited paragraphs, which still go through the citation and number checks.
- **Sources are checked against a registry.** A finding can only cite a page that Playwright actually opened and read. Search snippets help find pages, but they are never treated as evidence.
- **Figures are checked.** Every number in a finding or report paragraph must appear on at least one of the pages cited for it.
- **The amount of work is limited.** The application caps model turns, searches, page opens and time for each task. The full run also has a timeout.
- **Citations and the PDF are verified.** Every `[S#]` must refer to a source that was read and approved, and every paragraph must have a citation. The PDF must exist, open successfully and contain the report title, at least one citation and the References heading.

A run is only marked **completed** after the PDF passes verification. If an exception, timeout or invalid model response remains after the single repair attempt, the run is marked **failed** and the reason is recorded. The workflow doesn't continue silently.

## Technology

| What it needs to do | What I used and why                                                                                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Orchestration       | **LangGraph**: makes the workflow explicit, handles the conditional paths for the three review outcomes and keeps workflow state separate from agent histories. |
| LLM                 | **Anthropic Claude Sonnet** (`claude-sonnet-5-5`): the same model is used by all three agents, with prompt caching enabled.                                     |
| Search              | **Tavily**: used to find candidate URLs, not as evidence in its own right.                                                                                      |
| Browser and PDF     | **Playwright Chromium**: reads the original web pages and then generates the PDF. Each run uses one browser.                                                    |
| Validation          | **Pydantic v2**: validates every hand-off between agents.                                                                                                       |
| Report              | **Jinja2** for the HTML template and **pypdf** to check the finished PDF.                                                                                       |
| Logging             | JSONL execution trace, implemented in `app/trace.py`.                                                                                                           |
| Tests               | **pytest** for the test suite and **Pyright** for type checking.                                                                                                |

## Setup

You'll need Python 3.11 or newer, an Anthropic API key and a Tavily API key.

```bash
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m playwright install chromium
copy .env.example .env                 # macOS/Linux: cp .env.example .env
```

| Variable            | Required? | What it's for                                                                                                      |
| ------------------- | --------- | ------------------------------------------------------------------------------------------------------------------ |
| `ANTHROPIC_API_KEY` | yes       | Your Anthropic API key.                                                                                            |
| `ANTHROPIC_MODEL`   | yes       | The model used by all three agents, for example `claude-sonnet-5-5`.                                               |
| `TAVILY_API_KEY`    | yes       | Your Tavily API key.                                                                                               |
| `RESEARCH_DEPTH`    | no        | Sets the depth to `quick`, `standard` (the default) or `deep`. The `--depth` command-line option takes precedence. |

You can override any numeric limit in `app/config.py` by setting an environment variable with the same name in capitals. For example, `TASK_MAX_PAGE_OPENS=4` or `PDF_CHAR_LIMIT=25000`. These overrides take precedence over the depth preset.

## Running

```bash
python -m app.cli run --topic "What a UK business should know before signing a new electricity contract"
python -m app.cli run --depth quick --topic "..."
python -m app.cli run --verbose --topic "..."   # also show library log messages
```

The three depth settings control how much research the application carries out:

| Depth      | Plan tasks | Rounds  | Follow-ups | Per task                 | Source target | Measured on the demo topic     |
| ---------- | ---------- | ------- | ---------- | ------------------------ | ------------- | ------------------------------ |
| `quick`    | 3–4        | 1       | 0          | 2 searches, 2 page opens | about 4–6     | 152 s, 8 sources, about $0.34  |
| `standard` | 4–5        | up to 2 | up to 3    | 4 searches, 3 page opens | about 8–15    | 422 s, 20 sources, about $1.17 |
| `deep`     | 6–8        | up to 2 | up to 5    | 5 searches, 5 page opens | about 20–30   | not run live                   |

The cost figures are estimates based on the logged token counts and Sonnet 5.5 list prices ($2 per million input tokens, $10 per million output tokens, $2.50 per million cache-write tokens and $0.10 per million cache-read tokens). They are not billing figures. Results can vary between runs at the same depth, so the source targets are guidance rather than hard limits.

**Exit codes:** `0` means the run completed; `1` means it stopped with `STOP_INSUFFICIENT` or failed; `2` means there was a configuration problem, such as a missing environment variable or an unknown depth setting.

## Output

Each run gets its own directory at `artifacts/runs/<run_id>/`. These run directories are ignored by Git.

| File                                                                                                                                | What it contains                                                                                                                                                                                    |
| ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `report.pdf` / `report.html`                                                                                                        | The report, including its title, topic, date, introduction, sections with `[S#]` citations, any limitations, references (title, publisher, URL, publication date and access date) and page numbers. |
| `trace.jsonl`                                                                                                                       | The execution trace, described below.                                                                                                                                                               |
| `run_summary.json`                                                                                                                  | The run status and depth, total duration, time spent in each stage, tokens used by each agent (cached and uncached), source counts, configuration overrides, excluded tasks and the PDF path.       |
| `plan.json`, `research_round_N.json`, `checks_round_N.json`, `review_round_N.json`, `approved_research.json`, `report_content.json` | The validated JSON passed between the agents and workflow stages.                                                                                                                                   |

**Committed sample runs:**

- `artifacts/sample_run/`: the submission sample, using standard depth. Run ID: `20261104-164855-59366f`.
- `artifacts/sample_run_quick/`: the quick-depth example. Run ID: `20261104-165617-c44dcd`.

Both samples were generated using the final code.

- `artifacts/earlier_run/`: an earlier standard-depth run (`20261104-155918-bc8ed4`), made before the reviewer caveats were added. It's kept for the consistency comparison in the design note.

### Reading `trace.jsonl`

Every line in the trace has the same fields: `timestamp`, `run_id`, `agent`, `session_id`, `event`, `task_id`, `handoff_id`, `status`, `duration_ms`, `metadata` and `error`. API keys are replaced with `[REDACTED]` before anything is written to the file. Hand-off IDs follow the order of execution, starting with `H1-plan`, `H2-task-T1` and so on.

The main event groups are:

- **Run:** `run_started`, `session_created`, and `run_completed` / `run_stopped` / `run_failed`, including run totals.
- **Planning and review:** `plan_created`, `checks_completed`, `review_decision`, `accept_refused`, `research_more_requested`, `round_limit_override` and `decision_override`.
- **Research:** `task_dispatched` / `task_returned`, `search_performed`, `page_opened` / `page_failed`, `source_registered`, `number_check`, `findings_extracted`, `cap_reached` and `forced_finish`.
- **Report and PDF:** `research_approved`, `creator_invoked`, `report_returned`, `citation_validation`, `html_rendered`, `pdf_generated` and `pdf_verified`.
- **Model calls:** `llm_call`, which records token usage, cache use, retries and duration.

## Tests

Run the test suite and type checker with:

```bash
pytest              # 160 tests; no network or API keys needed
python -m pyright   # type check
```

The model and search calls are mocked, so the tests don't need API keys or internet access. The browser and PDF tests use real headless Chromium, but run against a local HTTP server.

The tests cover:

- independent agent sessions
- valid and malformed schemas
- missing, unknown and unread sources, and invalid URLs
- browser failures, including 404s, 5xx retries, timeouts, refused connections, empty pages, non-HTML content and PDFs
- research limits and the final structured submission
- review outcomes, the round limit, the acceptance gate and the rule for dropping weak tasks
- citation and number checks
- PDF generation and verification
- successful and failed complete runs
- trace redaction, depth presets and the CLI

## Limitations

There are a few things the current implementation does not fully check yet:

- **It doesn't check the wording of claims against their sources.** A claim without numbers could still be stronger than the source says, such as using "must" when the page says "may". At the moment, this is covered by the prompts and the orchestrator's review.
- **The number check has some gaps.** It doesn't catch numbers written as words or single digits.
- **Source targets are guidance, not limits.** The quick sample cited 8 sources against a target of 4–6, while the standard sample cited 20 against a target of 8–15.
- **The `deep` setting hasn't been run live.** It has been tested, but its actual time and cost haven't been measured.
- **The prompts are designed for UK, English-language topics.** All three agents use the same model.
- **Results depend on live websites.** If a page won't load, the application records the failure and doesn't cite it. The task may therefore finish with less evidence than planned.
- **There is no evaluation harness yet.** So far, I've assessed quality by inspecting the live runs.

## How this was built

I developed the project plan from the brief, defining what the system needed to achieve, how the three agents should work, and the architecture that would bring them together. I also set the implementation priorities and guardrails, including the responsibilities of each agent, how their sessions and information should be kept separate, the checks required to protect the quality of the output, and what was outside the scope of the project.

I used **Claude Code (Anthropic)** to help manage and streamline the implementation. Working through the project in stages, I used it to assist with writing and refining the code, running tests, carrying out live executions and inspecting the results. I directed the work throughout, reviewed the implementation and test results, and made the decisions on what to keep, change or improve.

## Read the code in this order

You should be able to follow the main parts of the code in about fifteen minutes. I'd start here:

1. **`app/workflow.py`**: Shows the complete run as a LangGraph graph. Each stage is a short method. `_review` and `_enforce_rules` contain the main decision rules.
2. **`app/agents/`**:
   - `orchestrator.py`: Handles planning, review and the single repair attempt.
   - `researcher.py`: Handles the browser and search tool-use loop. Start with `run_tasks`; the tools it uses are below it.
   - `creator.py`: Makes one structured model call to produce the report content.
3. **`app/schemas.py`**: Contains the models used for agent hand-offs, followed by `validate_citations` and `unsupported_numbers` near the end.
4. **`app/checks.py`**: Contains the deterministic checks that run before each review.
5. **`app/tools/`**: Contains `search.py`, `browser.py` and `pdf.py`, which handle the external services and PDF generation.

The remaining files, including `llm.py`, `sessions.py`, `trace.py`, `config.py` and `cli.py`, handle the supporting setup and plumbing. You can skim those once you've followed the main workflow.

## Project layout

```
app/
  cli.py        command line entry point        config.py    settings, depth presets, .env loading
  schemas.py    hand-off models and checks       sessions.py  isolated agent sessions
  llm.py        Anthropic wrapper, retries        workflow.py  LangGraph workflow and enforced rules
  checks.py     deterministic research checks     trace.py     JSONL trace with redaction
  agents/       orchestrator.py, researcher.py, creator.py
  tools/        search.py (Tavily), browser.py (Playwright), pdf.py (Jinja2 + Chromium PDF)
  prompts/      one instruction file per agent    templates/   report.html
tests/          pytest suite
artifacts/      sample_run/, sample_run_quick/, earlier_run/ (committed); runs/ (git-ignored)
```
