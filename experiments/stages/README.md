# SERP stage lab

Self-contained experiment for the revised SERP premise:

> BM25 is the ranker. A separate lexical channel recovers what BM25 misses. Fixed
> RRF combines the lists. Packing runs only after retrieval.

The protocol is frozen in `PROTOCOL.md`. Results are written to `results/`.

## Setup

From the worktree root:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install snowballstemmer
```

The original research data lives at `/Users/asmit/Personal/algo`. Override with
`ALGO_DATA_ROOT` if needed. The new corpus is at `experiments/stages/data/webis-touche2020`.

## Run

```sh
# Stage 1+2: channels and RRF on the new corpus and the SciFact sanity check
.venv/bin/python experiments/stages/run_retrieval.py --datasets touche,scifact

# Stage 3: packing controls on document-level budgets
.venv/bin/python experiments/stages/run_packing.py --datasets scifact,touche --budgets 500,1000,2000

# Exact identifiers and negative queries
.venv/bin/python experiments/stages/run_exact_ids.py --datasets scifact
```

## Files

- `lab/harness.py`: copied from `first_principles`, with a data-root override and the `touche` entry
- `lab/channels.py`: word BM25, Porter BM25, char4, VC, VC+grams, one-stream
- `lab/fusion.py`: fixed RRF
- `lab/metrics.py`: retrieval metrics and stage transitions
- `lab/pack.py`: truncation, greedy coverage, residual packing
- `lab/exact_ids.py`: synthetic exact-identifier and negative-query suite
- `run_retrieval.py`, `run_packing.py`, `run_exact_ids.py`: runners

## Why document-level packing is only a sanity check

BEIR labels are document-level. The product packs chunks, so the real packing
test belongs in `scripts/stage-benchmark.js` against the chunked SERP database.
The Python packer tests the selection idea on the same corpus and labels; it is
not the final packing evidence.
