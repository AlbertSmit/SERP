# SERP stage experiment: plan

Worktree: `/Users/asmit/Personal/algo-serp-stages`
Branch: `experiment/serp-stages`
Frozen protocol: `experiments/stages/PROTOCOL.md`

## Premise under test

SERP should not claim to replace BM25. It should be:

1. **BM25 as the ranker.** Porter-stemmed BM25 is the fair baseline.
2. **A separate lexical recall channel.** Char4 grams and corpus-vetted concepts
   (VC), not a new hand-designed scoring formula.
3. **Fixed RRF.** Reciprocal-rank fusion, `k = 60`, no weight tuning.
4. **Packing after retrieval.** Residual packing only when a real token budget
   requires selection. The coverage gate moves here.
5. **Stage-wise evidence.** Did the channel find relevant documents BM25 missed?
   Did fusion rank them usefully? Did packing preserve them?

## Phases

### Phase 0 — Isolation (done)

- Worktree created from the SERP repo.
- Current uncommitted engine state carried in as the baseline.
- Python lab vendored from `first_principles` with a data-root override.
- New corpus downloaded: BEIR Webis-Touche2020 (382,545 docs, 49 queries).
- Protocol and decision rules frozen before results.

### Phase 1 — Retrieval channels (running)

Deliverable: `experiments/stages/results/retrieval.json`

Systems: word BM25, Porter BM25, char4, VC, VC+grams, and fixed RRF combinations.
Datasets: `touche` (independent) and `scifact` (sanity only).

Gate:

- `new_relevant > lost_relevant` at depth 100 on Touche.
- RRF nDCG@10 non-inferior to Porter within `-0.005`.
- R@100 does not fall.

If the gate fails, stop. Do not tune against SciFact or the SERP fixtures.

### Phase 2 — Exact identifiers and negatives (running)

Deliverable: `experiments/stages/results/exact_ids.json`

Invariants:

- An exact identifier outranks every spelling variant.
- An unknown identifier does not activate the gram channel.
- A near-miss variant does not appear above the exact identifier.

Expected finding from the smoke run: char4 alone activates on unknown identifiers
(`negative_activation ≈ 0.975`), so the gate is mandatory.

### Phase 3 — Packing (running)

Deliverables:

- `experiments/stages/results/packing.json` (document-level sanity check)
- `experiments/stages/results/stage-benchmark.json` (chunk-level SERP benchmark)

Controls: truncation, greedy coverage, residual, P4, current `packContext`.

Gate:

- A packer is kept only if it preserves at least as many relevant documents as
  truncation at the same budget.
- It must not increase exact-ID near-miss activation.

If truncation wins, P4 is not the differentiator. Keep it behind a flag and describe
SERP as budgeted evidence selection, not a superior scorer.

### Phase 4 — Port only the winners

- Keep the new stage API (`retrieveRanked`, `fuseRrf`, `packRanked`, `packHybrid`).
- Keep the gram index and exact-ID gate if Phase 1 passes.
- Keep the packer only if Phase 3 passes.
- Do not change the existing `packContext` default until the regression fixture and
  the new benchmark agree.

### Phase 5 — Independent check

- One frozen run on Touche.
- One frozen run on the chunk benchmark.
- A second reader checks that the reported numbers match the JSON.
- No post-hoc tuning.

## File ownership

| Area | Owner | Files |
|---|---|---|
| Python retrieval | subagent A | `lab/channels.py`, `lab/fusion.py`, `lab/metrics.py`, `run_retrieval.py` |
| Python exact/packing | subagent B | `lab/exact_ids.py`, `lab/pack.py`, `run_exact_ids.py`, `run_packing.py` |
| JS stage API | subagent C | `src/engine/engine.js`, `src/engine/tokenize.js`, `test/engine.test.js` |
| JS benchmark | parent / next subagent | `scripts/stage-benchmark.js`, `experiments/stages/JS_BENCHMARK.md` |
| Protocol | parent (frozen) | `experiments/stages/PROTOCOL.md` |

## Risks

- **Touche is large.** 382k long documents may strain memory. If the full channel
  stack cannot run in one process, split by channel and report the limitation; do
  not silently subset the corpus.
- **Touche has only 49 queries.** CIs will be wide. A null result is a valid stop.
- **SciFact has been seen before.** It is a sanity check, not new evidence.
- **The exact-ID suite is synthetic.** It tests invariants, not relevance.
- **Document-level packing is not chunk-level packing.** The JS benchmark is the
  real packing test; the Python packer is a sanity check.
- **The current SERP benchmark is circular.** Its ground truth is "chunks with the
  rarest query tokens", which rewards P4 by construction. Do not use it as evidence.

## New track: Entity-Bridge Retrieval (EBR)

The fusion-only framing was too conservative. The revised research direction is
multi-hop evidence retrieval without embeddings. See `experiments/stages/HOTPOT.md`.

Method: Porter BM25 first hop -> high-surprisal bridge entities from the seeds ->
second-hop entity postings -> per-entity coverage channel -> fixed RRF.

HotpotQA held-out test (6,005 questions, 66,581 paragraphs):

| system | nDCG@10 | both@10 | both@5 | R@10 |
|---|---:|---:|---:|---:|
| Porter BM25 | 0.733 | 0.611 | 0.463 | 0.797 |
| RRF(Porter, bridge, entity) | **0.756** | **0.717** | **0.531** | **0.849** |

Paired vs Porter: nDCG `+0.0225` [+0.0184,+0.0266], both@10 `+0.1052`
[+0.0939,+0.1152], R@10 `+0.0526` [+0.0468,+0.0583]. The standard 10-paragraph
pool also improves (`both@5 +0.1152`).

This is the strongest evidence so far for a no-embedding method that does
something BM25 structurally cannot: retrieve both supporting facts.

## Decision record

| Question | Result | Decision |
|---|---|---|
| Does char4/VC find relevant docs BM25 misses on Touche? | yes; three-way RRF +0.0459 nDCG, +0.0215 R@100, 25/6 new/lost | keep as a lexical recall layer |
| Does fixed RRF stay non-inferior on nDCG? | yes on Touche; SciFact within margin | keep RRF |
| Does the gram channel activate on unknown identifiers? | yes ungated; fixed with exact-term gate | gate mandatory |
| Does residual/P4 packing beat truncation at a budget? | no; residual/P4 underfill and lose recall | demote P4 behind a flag |
| Is P4 the product differentiator? | no evidence; P4 scoring is tied with truncation at matched tokens | reposition SERP as budgeted evidence selection |
| Can a no-embedding method beat BM25 on multi-hop? | yes; EBR +0.105 both@10, +0.023 nDCG on HotpotQA test | continue EBR, gate by query type |
| Does EBR transfer beyond HotpotQA? | **yes**: 2Wiki dev all@10 +0.124, R@100 +0.133, nDCG +0.057; MuSiQue paragraph-level all@10 +0.065, R@100 +0.083, nDCG +0.033; all CIs above zero | keep the multi-hop direction; EBR is not Hotpot-specific |
| Is the MuSiQue gain a title-leakage artifact? | no; rebuilt at paragraph identity, leak cases 0; evidence gain holds, nDCG/MRR shrink | corrected numbers are the record |
| How close is EBR to a neural reranker? | 92% of both@10 gain at 22x less latency; only 27% of nDCG gain | position EBR as recall/candidate generation, not a ranker replacement |
| Does EBR improve a neural reranker? | yes; CE(EBR pool) +0.0095 nDCG, +0.0370 both@10 over CE(Porter pool) | keep EBR as a first-stage candidate generator |
| Does code-graph EBR transfer to SERP's domain? | synthetic: +0.320 both@10 upper bound; +0.20 on the clean two-hop slice; +0.10 at 20% edge loss | keep as a relational channel, not a global ranker |
| Do typed relations improve the bridge channel? | **no, they hurt everywhere**: both@10 -0.080 vs untyped at fixed channel set; typed second-hop reach 0.162 vs untyped 0.686 | drop bridge typing; keep broad cross-seed consensus |
| Does EBR hurt single-hop retrieval? | **yes**: SciFact test nDCG -0.044; Touche R@100 -0.037; entity channel -0.48 | not a universal replacement |
| Can a text-only router fix that? | partially: 65% fire on HotpotQA, 7% on SciFact, 0% on Touche/NFCorpus; routed still -0.005 on SciFact and gives back 0.035 both@10 on HotpotQA | router is a safety valve, not a benefit predictor |
| Is EBR field-novel? | no; entity-based multi-hop IR is prior art | claim a measured project result, not novelty |
| Is TS 7's AST graph worth it? | yes for correctness, not retrieval: TS7 20/20 call precision vs regex 18/20; fused both@10 0.606 vs 0.608 | use TS7 AST behind `unstable/*` with a version pin; checker opt-in |
| Does a heterogeneous lexical portfolio beat BM25 everywhere? | **no**: macro nDCG 0.4615 vs 0.4675; Quora -0.0176, TREC-COVID -0.0120; 0/7 positive | do not ship uniform portfolio; only a selective/fallback selector could work |
| Is there oracle headroom for a selector? | yes: per-query oracle all systems +0.064 macro; BM25+RM3+DPH oracle +0.034; non-BM25 best on 24-90% of queries | test a QPP/feature selector with BM25 fallback, not another weighted fusion |
| Can a learned selector capture that headroom? | **no**: selector macro 0.4620 vs BM25 0.4675; ArguAna -0.0137, Quora -0.0225; training-holdout argmax 35% vs 83% for always-BM25 | stop the universal no-embedding scorer search |
| Final universal-scorer verdict | not in reach with these lexical scorers/features; oracle headroom is not learnable from query/list features | ship BM25 default + explicit RM3 workload option + multi-hop/code specialists |
| Is constrained evidence-set retrieval a real new algorithm? | **yes as a packing/selection objective**: with gold constraints, CESR all@10 .628 vs EBR .366 (2Wiki, 500 tokens), .341 vs .139 (MuSiQue); uses 15-56% of EBR tokens | keep CESR as the deep-mode packing objective |
| Why doesn't CESR solve retrieval? | coverage saturates; 2Wiki is limited by pool recall (73.2% ceiling), MuSiQue by constraint adequacy (all constraints covered -> complete evidence only 52%) | next: exact constraints for code/memory, not fuzzy prose constraints |
| What is SERP's new algorithm identity? | BM25/EBR candidate generation + constrained evidence-set selection under budget; exact code graph constraints are the deployable case | test CESR-code before claiming a universal retriever |
| Does CESR-code become a real retrieval algorithm? | **no; it is a real packing algorithm**: CI-backed all@10 gains, but full coverage implies complete evidence only 72.5%; 96.8% of missing gold satisfy an already-covered constraint | ship CESR as the packer; do not call it a retriever |
| What is the remaining hard problem? | existential coverage cannot express universally quantified evidence ("every caller of X") and identical alternatives need identity/tie-breaking; the query generator also has 65/250 text-ambiguous queries | CESR-U research only: universal/identity constraints + real-repo qrels |
| Does the code-graph/CESR direction transfer to real code search? | **no**: 92 PR-derived queries on zustand; BM25 file R@10 .828 / nDCG .685; graph_rrf .787 / .621 (significantly worse nDCG); CESR .842 / .703 but 90/92 ties and gate fails | do not ship code graph/CESR as retrieval; synthetic +0.32 did not reproduce |
| Why did the synthetic code graph look good? | graph-generated qrels and second-hop queries; real PR text names the answer and BM25 already ranks it; 45% of changed files are tests the query never names | treat synthetic graph benchmarks as capability tests only |
| Does CESR-U fix the universal/identity failure? | **yes on determined queries**: text-unique all@10 .919-.924 vs CESR-code .849-.854 and RRF .692-.768; full-coverage completeness 1.000 on 161 determined queries, 0.924 overall | CESR-U is a real algorithm when constraints determine the answer |
| What limits CESR-U? | remaining misses are 100% witness alternatives; 24/185 text-unique queries are ambiguous, and real PR text does not determine constraints | do not sell it as universal code search; use it for structured evidence queries |
