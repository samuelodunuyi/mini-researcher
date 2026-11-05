# Sample run: quick depth

This is a second measured example, run with `--depth quick` on the same final code as the main sample. It's copied unchanged from `artifacts/runs/20261104-165617-c44dcd/`. The main submission sample is [`../sample_run/`](../sample_run/), which uses standard depth.

```
python -m app.cli run --depth quick --topic "What a UK business should know before signing a new electricity contract"
```

| | quick (this run) | standard (`sample_run`) |
|---|---|---|
| Run | `20261104-165617-c44dcd` | `20261104-164855-59366f` |
| Status | completed | completed |
| Duration | 151.8 s | 421.5 s |
| Research rounds | 1 (quick never runs a second) | 2 |
| Tasks | 4 | 8 (5 planned and 3 follow-ups) |
| Findings | 14 | 30 |
| Sources cited (all read, none failed) | 8 | 20 |
| Reviewer caveats passed to the creator | 17 | 19 |
| Model calls | 16 | 37 |
| Input tokens (total / from cache) | 264,686 / 209,371 | 1,946,784 / 1,759,165 |
| Output tokens | 17,680 | 52,161 |
| Estimated cost | about $0.34 | about $1.17 |
| Report | 5 pages, 9 sections | 8 pages, 10 sections |

Costs are estimates from the logged token counts at Sonnet 5.5 list prices ($2 per million input tokens, $10 output, $2.50 cache write, $0.10 cache read), not billing data.

All eight sources are official: Ofgem, the Energy Ombudsman and GOV.UK. The number check caught one finding with four figures not in its cited page, and the researcher corrected it before submission. The introduction describes scope only.

The files have the same layout as `sample_run/`. With quick depth there is only one round of research, checks and review.
