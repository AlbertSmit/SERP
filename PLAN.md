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

## Decision record

Fill this in as results arrive:

| Question | Result | Decision |
|---|---|---|
| Does char4/VC find relevant docs BM25 misses on Touche? | pending | pending |
| Does fixed RRF stay non-inferior on nDCG? | pending | pending |
| Does the gram channel activate on unknown identifiers? | smoke: yes | gate before fusion |
| Does residual/P4 packing beat truncation at a budget? | smoke: no (doc level) | pending chunk level |
| Is P4 the product differentiator? | pending | pending |
