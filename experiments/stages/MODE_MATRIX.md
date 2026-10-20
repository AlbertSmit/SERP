# The mode boundary: ship/kill a product policy, not a router

A3 asks a policy question, not a modelling one: can `default = Porter BM25`,
`deep = EBR`, `routed = the frozen per-query router` and `graph = Porter + code
graph` coexist without silently degrading ordinary search? No new scorer was
written, no router rule or RRF weight was tuned, and every constant is the one
already frozen in `HOTPOT.md` / `EBR_ROUTER.md` / `CODE_GRAPH.md`.

**Verdict:** Porter BM25 is the only safe default. `deep` and `graph` are defensible
specialists but must be explicit or corpus-declared. The per-query router is not
ship-worthy: it reduces EBR's SciFact damage from −0.044 to −0.005, but that
residual is still a significant nDCG loss and the bound cannot be certified. Exact
identifiers are safe in every shipped path, and the mandatory identifier gate
removes even the residual variant noise on identifier queries.

## Read this before the numbers

- The code corpus is a **synthetic capability test**: 250 graph-generated queries
  over 14 files / 289 chunks with graph-closure qrels, not human relevance.
- **Touche has 49 queries.** Its CIs cannot resolve a ±0.02 nDCG effect; quoted
  Touche nDCG is a direction, not a confirmation.
- **NFCorpus dev is a diagnostic**, never confirmation.
- `deep` and `routed` are reported on single-hop sets to measure the damage a
  policy would do, not because they are candidates there.
- HotpotQA 6,005 is the held-out test split; SciFact/Touche are test; NFCorpus dev.
- Paired bootstraps are 1,000 resamples, seed 1729. Safety is judged against the
  pre-registered margin **−0.005**, not the sign of a delta.

## Frozen modes

| mode | definition | allowed workloads |
|---|---|---|
| `default` | Porter BM25, k1=1.2, b=0.75 | all |
| `deep` | EBR: Porter + untyped bridge + per-entity, RRF k=60, 20 seeds, 20 bridges, 5% df cap | explicit, or declared `multi_hop_qa` (HotpotQA) |
| `routed` | `lab/router.py`; deep when its rules fire, else default | measured as an automatic baseline only |
| `graph` | Porter + `lab/code_graph.py` graph channel, RRF k=60, code corpus only | declared `code_graph` (SERP code) |

The mandatory **exact-identifier gate** is part of the policy: a digit/underscore
token (`hba1c`, `batch2`, `sql_v2`) forces the lexical default and disables bridge,
entity and graph expansion. Bare numbers (`1990`) are not identifiers and do not
fire the gate.

## Mode matrix

Primary metric per set, with Δ vs Porter BM25 (nDCG for text, nDCG@10 for code).

| workload | n | default | deep | routed | graph | router fire | ID gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| hotpot/test (nDCG@10) | 6,005 | 0.7331 | **0.7556** | 0.7479 | — | 65.2% | 3.1% |
| hotpot/test (both@10) | 6,005 | 0.6113 | **0.7166** | 0.6816 | — | | |
| scifact/test (nDCG) | 300 | **0.6838** | 0.6398 | 0.6788 | — | 7.0% | 25.7% |
| touche/test (nDCG) | 49 | 0.3262 | 0.3479 | 0.3262 | — | 0.0% | 0.0% |
| nfcorpus/dev (nDCG) | 324 | **0.2918** | 0.2823 | 0.2918 | — | 0.0% | 0.3% |
| serp_code/250 (nDCG@10) | 250 | 0.4191 | — | — | **0.5552** | — | 3.6% |

Paired deltas vs Porter (95% CI, 1,000 resamples, seed 1729):

| set | mode | ΔnDCG | 95% CI | safety at −0.005 |
|---|---|---|---:|---|
| hotpot/test | deep | **+0.0225** | [+0.0184,+0.0266] | pass |
| hotpot/test | routed | **+0.0148** | [+0.0113,+0.0181] | pass |
| scifact/test | deep | **−0.0440** | [−0.0652,−0.0254] | fail |
| scifact/test | routed | **−0.0049** | [−0.0110,−0.0007] | **inconclusive** |
| touche/test | deep | +0.0217 | [−0.0125,+0.0544] | inconclusive |
| touche/test | routed | 0.0000 | [0.0000,0.0000] | pass (0 fires) |
| nfcorpus/dev | deep | −0.0095 | [−0.0215,+0.0028] | inconclusive |
| nfcorpus/dev | routed | 0.0000 | [0.0000,0.0000] | pass (0 fires) |
| serp_code/250 | graph | **+0.1361** | [+0.1129,+0.1575] | pass |

Secondary metrics that change the reading:

- **HotpotQA deep**: both@10 **+0.1052** [+0.0939,+0.1152], both@5 +0.0673, R@10
  +0.0526. Routed keeps two thirds: both@10 +0.0703 [+0.0611,+0.0793].
- **SciFact deep**: R@100 −0.0043 (CI spans zero), MRR −0.0387 [−0.0630,−0.0162].
  The loss is ordering, not candidate coverage.
- **Touche deep**: nDCG +0.0217 but **R@100 −0.0368** [−0.0616,−0.0104]. Shallow
  order bought with deep recall; neither is a win.
- **NFCorpus deep**: nDCG −0.0095 (ns) with R@100 +0.0165 [+0.0042,+0.0290].
- **Code graph**: both@10 +0.2680 [+0.2080,+0.3240], R@10 +0.1580. By type,
  `two_hop` 0.12→0.34 and `export_import` 0.00→0.82; `calls_two` 0.50→0.52. The
  headline remains a graph-closure upper bound.

## Exact-identifier invariants

| suite | system | hit@1 | hit@10 | MRR | near-miss before exact | unknown activation |
|---|---|---:|---:|---:|---:|---:|
| JS engine, 100 pos | engine | 1.000 | 1.000 | 1.00* | 0.000 | 0 / 82 probes |
| SciFact, 60 pos / 40 neg | default | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| SciFact | deep | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| SciFact | deep + gate | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Code terms, 41 pos / 40 neg | default | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Code terms | graph | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Code terms | graph + gate | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

\* JS engine reports mean exact rank 1.00, not MRR. Its activation counters are 0
for single, underscore, mixed and prose probes.

The invariant holds: exact identifiers outrank variants everywhere, and unknown
identifiers never activate bridge, entity or graph expansion (0/40 in both Python
suites). The residual cost is visibility, not order: ungated deep places variants
in the top-10 for 6.7% of identifier queries (graph: 14.6%), never above the exact
hit. The gate removes it entirely.

## Latency and cost

| workload | docs | default p50/p95 ms | specialist p50/p95 ms | build | entity index / peak RSS |
|---|---:|---:|---:|---:|---|
| hotpot/test | 66,581 | 0.43 / 0.65 | deep 13.75 / 17.56 | 8.9 s | 427k entities, 811k postings |
| scifact/test | 5,183 | 0.12 / 0.20 | deep 1.99 / 2.29 | 0.6 s | 33.6k entities |
| touche/test | 382,545 | 1.02 / 3.10 | deep 140.2 / 161.9 | 121.1 s | 1.0M entities, 5.5M postings, 11.7 GB peak |
| nfcorpus/dev | 3,633 | 0.09 / 0.15 | deep 0.56 / 1.76 | 0.7 s | 18.8k entities |
| serp_code/250 | 289 chunks | 0.07 / 0.08 | graph 1.04 / 1.23 | 0.4 s | 132 symbols, 397 calls |

Peak RSS is process-wide; the 11.7 GB figure is the Touche run. Deep costs ~20–30×
the default's latency on text and becomes an offline-index, not per-query, luxury
on a 382k-document corpus. Graph mode costs ~15× a 0.07 ms lexical query, which is
still negligible in absolute terms.

## Acceptance gate

| # | invariant | result |
|---|---|---|
| 1 | `default` reproduces Porter within numerical tolerance | **PASS** — max abs diff 0.0 vs `results/ebr_beir.json` and `results/code_graph.json` |
| 2 | `deep` only by explicit mode or declared corpus/profile | **PASS** — automatic deep only on `multi_hop_qa` (hotpot); code uses the `code_graph` profile; no universal router shipped |
| 3 | no automatic policy below −0.005 on single-hop | **FAIL (strict)** — routed on SciFact −0.0049 [−0.0110,−0.0007]; the CI does not certify the −0.005 bound. Literal reading (CI entirely below the margin) passes; Touche/NFCorpus are identical because the router never fires |
| 4 | exact IDs outrank variants; unknown IDs do not expand | **PASS** — see table; 0 near-miss-before-exact, 0 unknown activation |
| 5 | fallback if any invariant fails | **APPLIED** — BM25-only default, specialists behind explicit flags |

Gate 3 is the key result and it is uncomfortable: **routed still loses on SciFact**,
significantly (1 win / 294 ties / 5 losses, MRR −0.0047 [−0.0100,−0.0011]). It is
not a benefit predictor; the fired subset is simply SciFact prose with relative
clauses, where Porter is already strong. `routed` gives back one third of the
HotpotQA gain (−0.035 both@10 vs always-deep) and still cannot be certified on
single-hop text. There is no threshold that fixes both.

## Recommended product policy

1. **Default search: Porter BM25**, exact-identifier gate on by default, no
   automatic expansion. Deterministic, cheapest, and reproduced exactly.
2. **`deep` is an explicit mode / declared multi-hop profile.** Do not ship the
   per-query router. On HotpotQA the declared profile keeps +0.105 both@10; routed
   would keep only +0.070 and would leak a loss onto single-hop text.
3. **`graph` is a code-corpus profile**, not a general channel, and is a capability
   demo until human/issue qrels exist.
4. **Exact identifiers are a hard gate, not a scoring feature.** It costs nothing on
   the frozen suites and removes variant visibility on identifier queries.
5. **Cost is part of the boundary.** Deep's 382k-document index is 1.0M entities,
   5.5M postings, 121 s build and ~11.7 GB peak; that belongs behind a workload
   flag, not in the default path.

## Reproduce

```sh
cd /Users/asmit/Personal/algo-serp-stages
.venv/bin/python experiments/stages/run_mode_matrix.py
```

New files: `lab/mode_matrix.py` (policy, gate, safety arithmetic, fixture),
`run_mode_matrix.py`, `results/mode_matrix.json`,
`data/mode_queries.jsonl` (frozen workload/query-hash manifest). Existing
implementations (`hotpot.py`, `ebr_beir.py`, `router.py`, `code_graph.py`,
`exact_ids.py`) are imported read-only. Runtime ~6 min, dominated by the Touche
index. Result JSON: all per-set summaries, paired CIs, W/T/L, latency percentiles,
cost, exact suites and gate verdicts.
