# Research findings

[Back to the README](../README.md) · [Mode guide](practical-guide.md) · [API](api.md) · [P4 history](history-p4.md)

This project began as an attempt to build one no-embedding scorer that beats
BM25 everywhere. **That search is closed: no such scorer was found.** What the
research supports is narrower. BM25 is the default ranker. Entity-bridge
retrieval (EBR) is a measured multi-hop recall mode. Constraint-driven
evidence-set selection (CESR-U) is a packing objective that works only when the
query's constraints determine the answer. The contribution is the measured
result and the way these known techniques are composed, not a new retrieval
algorithm. Entity-based multi-hop retrieval, RRF, RM3, DPH and cross-encoder
reranking are all prior art.

## Modes and research prototypes

Only **Default** is shipped. The others are measured research prototypes, not
supported product modes.

| Mode | What it is | Where it runs today |
| --- | --- | --- |
| Default | BM25 / FTS5, fixed RRF with an in-word 4-gram channel, truncation packing | JS engine: `packHybrid`, `retrieveRanked`, `packRanked` |
| Deep | EBR: Porter BM25 first hop, untyped bridge entities, per-entity coverage, fixed RRF (`k = 60`) | Python lab only (`experiments/stages/lab/hotpot.py`) |
| Structured | CESR-U: exact constraints, universal/identity coverage, minimal evidence set under a budget | Python lab only (`experiments/stages/lab/cesr_u.py`, `cesr_u_auto.py`) |
| Optional | RM3 expansion as a per-workload profile; a cross-encoder as a post-fusion reranker | Python lab only |

Only the default mode is in the shipped engine. The MCP `pack_context` tool now
defaults to the BM25 + 4-gram RRF + truncation path. Pass `mode: "legacy"` (or
enable `use_bridges` / `temporal_mode`) for the experimental P4 path. The library
method `packContext` remains the legacy P4 path for compatibility; use
`packHybrid` for the default path.

### Default: BM25 / FTS5

BM25 is the safest general baseline measured. Every lexical alternative tried
lost to it on at least one labelled BEIR set (see
[Benchmarks](#benchmarks)). The JS default adds a gated in-word character-4-gram
channel, fused with RRF. Exact identifiers always rank first, and unknown
identifiers cannot activate either channel. On Touche (n = 49), RRF(Porter, char4)
gains nDCG@10 +0.029 [+0.013, +0.046]. On SciFact train (n = 809) it is neutral:
−0.004 [−0.012, +0.004]. Treat the gram channel as a recall layer, not a
better ranker.

### Deep: EBR multi-hop

EBR adds a second hop: it retrieves evidence that shares an entity with the
first-hop results, not with the query. It retrieves complete evidence sets more
often than BM25 on three multi-hop datasets. It is **not** a general ranker:

- It hurts single-hop retrieval. SciFact nDCG@10 −0.044 [−0.065, −0.025];
  Touche R@100 −0.037 [−0.063, −0.011].
- Use it only when the caller knows the query is multi-hop. No query-level
  router was found (see below).
- Against a neural reranker, EBR reaches about **92% of the cross-encoder's
  both-facts gain, but only about 27% of its nDCG gain**. The both@10 difference
  is statistically indistinguishable from zero (EBR − CE(Porter) −0.010
  [−0.037, +0.017]); the nDCG difference is not (−0.072 [−0.085, −0.058]). EBR
  does not replace a reranker. It improves a reranker's candidate pool: rerank
  EBR's own candidates and the cross-encoder is significantly ahead on both@10
  too (−0.047 [−0.074, −0.021]), so EBR's both-facts reach is a candidate-set
  result rather than an ordering result.
- On a like-for-like CPU sample (n = 100), the cross-encoder costs 874 ms/query
  including the BM25 top-100 it needs, against 12.7 ms for EBR end to end (69×).
  The earlier 22× figure mixed a GPU rerank with a CPU pipeline and understated
  the cross-encoder's cost.

### Structured: CESR-U evidence-set selection

CESR-U chooses the smallest set of chunks that satisfies explicit constraints
("the definition of X and every caller of X") within a token budget. It is a
**packing/selection objective**. It becomes retrieval only when the constraints
fully determine the evidence set.

It has been validated on synthetic code queries with graph-generated qrels. With
a fixed text parser it plans **77.6%** of the queries, is oracle-identical on every
query it can plan, and on those 149 parsed text-unique queries beats RRF on
`all@10` by `+0.436 / +0.215 / +0.134` at 500/1000/2000 tokens. End to end with
an RRF fallback for unplannable queries, the gain over RRF at 2,000 tokens is
+0.106 [+0.059, +0.160] on all 188 text-unique queries, and +0.052 [+0.004, +0.096]
on all 250. It is 0.112 behind the oracle, and on conflicting-text queries it
loses to RRF (−0.113 [−0.226, −0.016]). The unplannable 22.4% is mostly `two_hop` (the bridge name
is not in the text) plus a few imported-callee cases. The cue phrases are the
generator's own templates, so this validates the plumbing, not paraphrase
robustness. On real PR-derived code search, CESR did not beat BM25.

### Optional: RM3 and a neural reranker

- **RM3** is a workload option, never a default. It significantly improves
  NFCorpus, SciDocs and TREC-COVID, and significantly hurts ArguAna and Quora.
- **A cross-encoder** (MiniLM-L-6 in the experiments) is an optional reranker
  after fusion. It is the reference that EBR is measured against, not part of
  the no-embedding core.

### Packing

Truncation is the default: take the fused ranks in order and skip chunks that
do not fit. P4, residual and greedy packers stay available through
`packRanked({ packer })` as **experimental** options. On the chunk benchmark
(SciFact test, 300 queries, paired bootstrap 95% CIs), current P4 and residual
packing use about 46–48 of 250 tokens, so their lower recall (0.49 vs 0.60) is
mostly budget underfill: at matched token spend truncation is not significantly
better than P4 on document recall (+0.010 [−0.022, +0.039]), and against
residual it is a recall tie with a small document-precision gain (+0.037
[+0.010, +0.064]). P4 never beats truncation at equal tokens. At the nominal
budget it is a trade-off, not a pure loss: P4's short packs have significantly
higher document precision (250 tokens: 0.372 vs 0.206, truncation − P4 −0.167
[−0.202, −0.133]; residual 0.338). The coverage gate
trades recall for precision: at 250 tokens, gated truncation is 0.601 document
recall / 0.206 document precision, while no-gate truncation is 0.688 / 0.101;
both differences are significant. The gate is a precision control, not a free
win.

## What sqlite-context is not

- **Not a universal BM25 replacement.** A single formula (P4) failed its packing
  gate. A uniform lexical portfolio and a learned per-query selector each failed
  the non-inferiority gate.
- **Not a RAG replacement.** EBR improves complete-evidence recall, but closes
  only about 27% of a cross-encoder's nDCG gain. It is statistically
  indistinguishable from a reranker over BM25's pool on both@10, and behind one
  over its own pool.
- **Not code-graph PR search.** On real PRs the code graph and CESR did not beat
  BM25. The synthetic gains came from graph-generated qrels.

## Research status (frozen)

The research program is frozen as of this release. No open experiment changes
the product decision.

- **Structured mode (CESR-U):** stays in the lab. The only test that would change
  a product decision is about 50 hand-written queries that name symbols in real
  users' words, not the generator's templates. Run it only if someone will
  actually call an explicit "definition of X and every caller of X" tool.
- **Deep mode (EBR):** stays in Python. Port it to JavaScript only when a real
  multi-hop workload appears.
- **Code search and the universal no-embedding scorer:** both are closed as
  negative results.
- **Settled:** BM25 remains the default; truncation remains the default packer;
  the coverage gate is a precision control with a measured recall cost.

## Benchmarks

All numbers are paired deltas with 95% bootstrap CIs, read from
`experiments/stages/results/*.json`. "n" is the number of queries.

### Real labelled data

| Claim | Data | Result | Source |
| --- | --- | --- | --- |
| EBR vs Porter BM25 | HotpotQA distractor paragraphs, held-out 80% of dev, n = 6,005 | both@10 0.611 → 0.717, **+0.105 [+0.094, +0.115]**; nDCG@10 +0.023 [+0.018, +0.027] | `hotpot.json` |
| EBR transfer | 2WikiMultiHopQA dev, n = 12,576 | all@10 0.364 → 0.488, **+0.124 [+0.118, +0.130]**; nDCG@10 +0.057 [+0.055, +0.059]; R@100 +0.133 | `multihop_transfer.json` |
| EBR transfer | MuSiQue dev, paragraph identity, n = 2,417 | all@10 0.145 → 0.210, **+0.065 [+0.052, +0.080]**; nDCG@10 +0.033 [+0.027, +0.040]; R@100 +0.083 | `multihop_transfer.json` |
| EBR vs cross-encoder | HotpotQA test subset, n = 1,000 | both@10: EBR 0.707, CE(Porter) 0.717, Porter 0.599. Paired EBR − CE: both@10 −0.010 [−0.037, +0.017] (ns); nDCG@10 −0.072 [−0.085, −0.058]. CPU p50: EBR 12.7 ms vs CE end to end 874 ms | `hotpot_neural_reference_paired.json` |
| EBR as reranker pool | same subset | CE(EBR pool) vs CE(Porter pool): nDCG +0.010 [+0.006, +0.013], both@10 +0.037 [+0.025, +0.050] | `hotpot_neural_reference.json` |
| EBR on single-hop | SciFact test (n = 300); Touche (n = 49) | **harmful**: SciFact nDCG −0.044 [−0.065, −0.025]; Touche R@100 −0.037 [−0.063, −0.011] | `ebr_beir.json` |
| Typed bridges | HotpotQA test | **negative**: both@10 −0.080 [−0.089, −0.072] vs untyped bridges; second-hop reach 0.162 vs 0.686 | `hotpot_typed.json` |
| Text-only EBR router | HotpotQA test; SciFact test | **conditional at best**: separates corpora, not queries. Fired and silent HotpotQA queries gain alike (+0.108 vs +0.101). SciFact −0.005 [−0.011, −0.001] fails non-inferiority. Not shipped | `ebr_beir.json`, `mode_matrix.json` |
| Uniform lexical portfolio (RRF) | 7 BEIR test sets | **failed**: macro nDCG@10 0.4615 vs BM25 0.4675; 0/7 sets positive; Quora −0.018 [−0.020, −0.015] | `lexical_portfolio.json` |
| Learned per-query selector | same 7 sets | **failed**: macro 0.4620 vs 0.4675; ArguAna −0.014 [−0.021, −0.007], Quora −0.023 [−0.025, −0.020]. A per-query oracle would gain +0.034, but that headroom was not learnable | `selective_lexical.json` |
| RM3 vs BM25 | same 7 sets | NFCorpus +0.013, SciDocs +0.005, TREC-COVID +0.028 (CIs above 0); ArguAna −0.040 [−0.051, −0.030], Quora −0.017 [−0.019, −0.015] | `lexical_portfolio.json` |
| Code graph / CESR on real PRs | zustand, n = 92 merged PRs, edited-file qrels | **did not beat BM25**. R@10 / nDCG@10: BM25 0.828 / 0.685; graph RRF 0.787 / 0.621 (nDCG −0.064 [−0.123, −0.006]); CESR 0.842 / 0.703 (nDCG +0.018 [−0.028, +0.066], 90/92 ties). Gate failed | `real_repo.json` |
| Packing (P4/residual vs truncation) | SciFact test chunks, n = 300, budget 250 | doc recall: truncation 0.601, P4 0.491, residual 0.486, using 166 vs 47 vs 48 tokens. Nominal-budget Δ vs P4 +0.110 [+0.077, +0.147]; at matched tokens +0.010 [−0.022, +0.039] (ns), so the gap is mostly underfill | `stage-benchmark.json` |

### Synthetic and capability results (circular; not relevance evidence)

| Claim | Setup | Result | Caveat |
| --- | --- | --- | --- |
| CESR prose packing | 2Wiki / MuSiQue dev, budget 500 | all@10 vs EBR truncation: 0.628 vs 0.366 (+0.262 [+0.255, +0.271]); 0.341 vs 0.139 (+0.201 [+0.182, +0.220]); uses 50–56% of EBR's tokens | **Oracle constraints** from gold evidence triples and decompositions. Same candidate pool, so this is a packing gain. Pool ceiling 0.73 / 0.58 |
| Code-graph EBR | 250 synthetic queries, this project's own code | both@10 +0.320 [+0.260, +0.376] vs Porter | **Graph-generated queries and qrels.** Did not reproduce on real PRs |
| CESR-code | same 250 queries | all@10 vs RRF truncation +0.220 [+0.152, +0.284] at 500 tokens | Oracle constraints and graph qrels. Full coverage implies complete evidence only 72.5% of the time |
| CESR-U | 185 text-unique synthetic queries | all@10 0.919–0.924 vs CESR-code 0.849–0.854 and RRF 0.692–0.768 (vs CESR-code +0.070 [+0.022, +0.124]) | Oracle constraints, graph qrels. `two_hop` constraints use the hidden bridge name that also defines the qrels. 16% of queries exceed the 500-token budget |
| CESR-U, parsed constraints | 250 synthetic queries, fixed text parser | plans 194/250; all@10 vs RRF on the 149 parsed text-unique queries +0.436 / +0.215 / +0.134 at 500/1000/2000 tokens; end to end +0.052 [+0.004, +0.096] on all 250 at 2,000 | Parser cues are the generator's templates; graph qrels; `two_hop` unplannable |
| TS 7 AST graph | this project's code | call-edge precision 20/20 vs regex 18/20; fused both@10 0.606 vs 0.608 | A **correctness and chunk-segmentation** result, not a retrieval win. No CIs |
| Exact identifiers | JS engine probe suite | 0 of 82 unknown-identifier probes activate retrieval | Synthetic invariant, not relevance |

### Historical P4 tables

The SciFact tables in [history-p4.md](history-p4.md) predate the stage
protocol. They used a combined two-query "budget recall" metric and were never
rerun. They are superseded by the matched chunk benchmark above, where current
P4 does not beat truncation. Do not cite them as evidence that P4 beats BM25.

## Limitations and non-goals

- EBR, CESR-U, RM3 and the cross-encoder are Python research code. They are not
  in the JS engine or the MCP server.
- `packContext` (library) remains the legacy P4 path, which loses nominal-budget
  recall to truncation on the chunk benchmark by underfilling the budget. The MCP
  tool defaults to `packHybrid` + truncation; legacy P4 is available via
  `mode: "legacy"`.
- The HotpotQA "test" split is a held-out 80% of the official dev set, retrieved
  from its 66,581 distractor paragraphs, not full Wikipedia. The 2Wiki and
  MuSiQue results are on dev splits with frozen constants. The router was
  designed after reading HotpotQA dev and SciFact train.
- CESR and CESR-code gains use oracle constraints. CESR-U also has a fixed
  text parser, but its cue phrases are the query generator's own templates, so
  constraint extraction from free text is unproven.
- Real-PR qrels are the files each PR edited, scored against a HEAD snapshot.
  45% of those files are tests, and temporal leakage is measured, not removed.
- SciFact was used in earlier research and is a sanity check only. Touche has
  49 queries.
- The neural reference is one model (MiniLM-L-6) on 1,000 queries. Its latency
  ratios come from a 100-query CPU sample on one machine; the original 22×
  compared a GPU rerank with a CPU pipeline.
- Packing CIs are paired over 300 SciFact queries. The nominal-budget gap to
  P4/residual is mostly budget underfill: at matched token spend, document-recall
  differences against P4 are not significant.
- Non-goals: another weight search, tuning on test sets, a universal scorer.

## Reproduce

Lab setup and data paths are in [experiments/stages/README.md](../experiments/stages/README.md).
Each runner writes its JSON to `experiments/stages/results/`.

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install snowballstemmer

PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_hotpot.py
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_multihop_transfer.py
.venv/bin/python experiments/stages/run_hotpot_neural.py      # needs a cross-encoder
.venv/bin/python experiments/stages/run_hotpot_neural_paired.py  # paired test + CPU latency
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_hotpot_typed.py
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_ebr_beir.py
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_mode_matrix.py
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_lexical_portfolio.py
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_selective_lexical.py
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_cesr.py
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_code_graph.py
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_cesr_code.py
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_cesr_u.py
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_cesr_u_auto.py
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_real_repo.py
node scripts/stage-benchmark.js                                # needs mcp_memory.sqlite
```

## Research docs

- Plan and decision record: [PLAN.md](../PLAN.md)
- Protocol and gates: [PROTOCOL.md](../experiments/stages/PROTOCOL.md),
  [ARGON_PLAN.md](../experiments/stages/ARGON_PLAN.md)
- Multi-hop: [HOTPOT.md](../experiments/stages/HOTPOT.md),
  [MULTIHOP_TRANSFER.md](../experiments/stages/MULTIHOP_TRANSFER.md),
  [NEURAL_REFERENCE.md](../experiments/stages/NEURAL_REFERENCE.md),
  [TYPED_EBR.md](../experiments/stages/TYPED_EBR.md),
  [EBR_ROUTER.md](../experiments/stages/EBR_ROUTER.md),
  [MODE_MATRIX.md](../experiments/stages/MODE_MATRIX.md)
- Universal scorer search: [LEXICAL_PORTFOLIO.md](../experiments/stages/LEXICAL_PORTFOLIO.md),
  [SELECTIVE_LEXICAL.md](../experiments/stages/SELECTIVE_LEXICAL.md)
- Evidence sets: [CESR.md](../experiments/stages/CESR.md),
  [CESR_CODE.md](../experiments/stages/CESR_CODE.md),
  [CESR_U.md](../experiments/stages/CESR_U.md),
  [CESR_U_AUTO.md](../experiments/stages/CESR_U_AUTO.md)
- Code: [CODE_GRAPH.md](../experiments/stages/CODE_GRAPH.md),
  [TS7_GRAPH.md](../experiments/stages/TS7_GRAPH.md),
  [REAL_REPO.md](../experiments/stages/REAL_REPO.md)
- Packing: [JS_BENCHMARK.md](../experiments/stages/JS_BENCHMARK.md)
