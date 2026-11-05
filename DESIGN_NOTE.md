# Design note

## How the system works

The application starts from the CLI and runs through LangGraph. The orchestrator creates a research plan, the researcher carries out the research, and the orchestrator reviews the findings before deciding what happens next. It can request more research (two rounds at most), stop if the evidence is insufficient, or approve the findings and pass them to the creator.

The researcher uses Tavily to find pages and Playwright with Chromium to read them, then returns findings linked to their sources. The creator turns the approved findings into structured report content. The agents only exchange validated Pydantic objects, and Python handles the rendering, citation checks and PDF verification.

## How I approached the brief

I took the requirement for the orchestrator to decide when the job is done to mean it should own the judgement calls: the plan, whether the findings are good enough, what needs further research, and when to accept or stop. It also passes any caveats on to the creator.

LangGraph handles the workflow, while Python enforces the rules around research rounds, sources and validation. A model can make useful judgements, but it should not be trusted to enforce every rule, especially when the brief asks for weak results to be caught.

I interpreted separate agent sessions as separate context, not separate processes. Each agent keeps its own history and system prompt, and although they share a stateless API client, every model call receives only its own agent's session.

## Reliability

The most important addition would be an **exact-quote support check**. The application verifies sources and numerical claims, but a statement can still be stronger than its source: a page might say something *may* happen, while the report says it *must*. I would require each finding to include a short direct quotation and check that it appears on the cited page. I would also check that the creator includes the orchestrator's caveats, which today it is only instructed to do.

Second, I would add **checkpointing and resume support** using LangGraph's checkpointer, so a failure while writing the report or generating the PDF does not lose the completed research. Third, I would enforce the **run deadline during execution**, since the 30-minute limit is only checked between steps. Finally, I would add a **warning for older sources**.

Several failure cases are already handled. The source registry and numerical check corrected four findings and one report paragraph in the sample run. Through the acceptance gate and drop rule, the application lists unsupported work as a limitation itself, and the creator stated the reviewer's 19 caveats as cited limitations. Pages that fail to load are never cited, and an out-of-credit error stops the run immediately. At the tool-call limit, the application finishes with structured output, because `claude-sonnet-5-5` returns HTTP 400 for forced tool calls.

## Consistency

The main thing I would improve is report structure: the creator currently organises the findings itself, so runs on the same topic differ. The two most recent standard runs, `20261104-164855-59366f` and `20261104-155918-bc8ed4` (an earlier run, in `artifacts/earlier_run/`), cited 20 and 17 sources, produced 30 and 33 findings and used different headings. A **fixed report outline with one section per planned task** would make reports easier to compare and gaps more obvious.

I would then build an **evaluation harness**: a fixed set of topics, each run twice, scored by code (citations, numerical support, share of official sources, coverage per task) and by an LLM judge (support and completeness). Run on every prompt or model change, it would measure whether the change helped instead of judging from one PDF. Already in place: Pydantic schemas, a fixed prompt per agent, one Jinja2 template and references written by code.

## Optimisation

Costs are estimates from logged tokens at Sonnet 5.5 list prices, not billing figures; the cost without caching is in brackets.

| Run | Depth | Time | Input tokens (cached) | Output tokens | Sources | Estimated cost |
|---|---|---:|---:|---:|---:|---:|
| `20261104-164855-59366f` | Standard | 422 s | 1,946,784 (90%) | 52,161 | 20 | $1.17 ($4.42) |
| `20261104-165617-c44dcd` | Quick | 152 s | 264,686 (79%) | 17,680 | 8 | $0.34 ($0.71) |

Research took 67% of the standard run (281 of 422 seconds). The tasks within a round are independent, so I would run them in parallel. I kept them sequential so the trace reads as one sequence and one browser suffices.

The researcher also accounts for 94% of input tokens, because its session resends the pages it has already read. Summarising earlier tasks instead would cut context size and time further. I would test that, and a cheaper model for extraction while keeping Sonnet for planning and review, once the evaluation harness can show that quality holds.

Already in place: prompt caching (an estimated 74% saving), fewer and broader tasks with source reuse, passage selection for long pages, and `--depth quick` at roughly a third of the time and cost.

## What I would do with one more week

1. The exact-quote support check, with the evaluation harness alongside it.
2. A fixed report outline, one section per research task.
3. Checkpointing and resume, and a deadline enforced during execution.
4. Parallel research tasks and a lighter researcher session.

## What I would do differently if I started again

* Let the orchestrator handle delegation itself. Currently a fixed LangGraph graph moves work between agents, and the orchestrator only reviews at set points. Starting again, I would consider making the orchestrator a tool-using agent, with tools such as `assign_research`, `request_follow_up` and `commission_report`, with Python still validating every call. This would follow the brief more closely, where the orchestrator sends work out and decides when the job is done, and would let the plan change as the research develops. The trade-off is a less predictable workflow, so the safeguards would move from the graph into tool validation.
* Make the claim itself the unit of evidence, not the paragraph. Findings are currently free text, checked afterwards for citations and numbers. Starting again, I would link every claim to an exact quotation from a stored snapshot of its source page, and the creator would build paragraphs from claim IDs, so every fact would be traceable by design. The exact-quote check under Reliability is the incremental step towards that.
