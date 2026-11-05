# Sample run

This is the submission sample: one genuine end-to-end run of the demo topic on the final code. It used the live Anthropic API (`claude-sonnet-5-5`), Tavily search and Playwright Chromium, and is copied unchanged from `artifacts/runs/20261104-164855-59366f/`.

```
python -m app.cli run --topic "What a UK business should know before signing a new electricity contract"
```

| | |
|---|---|
| Status | completed (standard depth) |
| Duration | 421.5 s: research 281.4 s, report writing 92.2 s, review 36.0 s, plan 11.0 s, PDF under 1 s |
| Research rounds | 2. The first review asked for 3 follow-ups, each linked to the task it fills. The second review accepted, with 19 caveats for the report |
| Tasks | 8 (5 in the plan and 3 follow-ups) |
| Findings | 30 |
| Sources | 20, all read successfully and all cited; 0 failed |
| Decision overrides | none: no refused ACCEPT and no excluded tasks |
| Checks that fired | The number check caught unsupported figures in 4 findings, all corrected by the researcher. The creator needed one repair, for a figure its cited pages don't contain |
| Model calls | 37 (orchestrator 3, researcher 32, creator 2) |
| Input tokens | 1,946,784 in total: 1,759,165 read from cache (90%), 187,471 written to cache and 148 uncached |
| Output tokens | 52,161 |
| Estimated cost | about $1.17 at Sonnet 5.5 list prices ($2 per million input tokens, $10 output, $2.50 cache write, $0.10 cache read), against about $4.42 without caching. These are estimates, not billing data |
| Report | `report.pdf`, 8 pages, 10 sections. The introduction describes scope only. The reviewer's caveats appear as cited limitation statements, for example that the back-billing cap was not confirmed and is "not stated here as fact". Citations were validated and the PDF verified |

## Files

| File | Contents |
|---|---|
| `report.pdf` / `report.html` | The final report and the HTML it was printed from |
| `trace.jsonl` | Every event and hand-off (`H1-plan` to `H15-report-content`): 171 records |
| `run_summary.json` | Totals: status, timings per stage, tokens per agent, overrides |
| `plan.json` | The orchestrator's research plan |
| `research_round_N.json` | The researcher's results for each round, including the page text it read |
| `checks_round_N.json` | The deterministic checks before each review |
| `review_round_N.json` | The orchestrator's decision for each round, including the final review's caveats |
| `approved_research.json` | The creator's only input: approved findings, the sources they cite and the reviewer's caveats |
| `report_content.json` | The creator's structured report, before rendering |
