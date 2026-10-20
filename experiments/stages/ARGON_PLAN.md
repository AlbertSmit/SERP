# ARGON: retrieval strategy and falsification plan

> **For DeepSeek v4.1 Flash subagents:** execute the experiments independently, one frozen configuration per experiment, and write the result JSON before interpreting it. Do not tune on test queries.

**Goal:** decide whether SERP should ship as a lexical default plus explicit relational specialists, rather than continue searching for a universal BM25 replacement.

**Architecture:** Porter BM25 remains the default ranker. Separate no-embedding channels add candidates only when the workload has a defensible structural reason to need them: entity bridges for multi-hop QA, exact code graphs for code/memory, and optionally an external relation/alias index. Fixed RRF and truncation remain the baseline controls; any neural reranker is a separately labelled reference or optional final stage.

**Tech Stack:** SQLite FTS5, existing Python experiment harness, fixed RRF (`k=60`), paired bootstrap intervals, BEIR-style qrels, TS7 AST graphs where applicable.

---

## Executive verdict

SERP has demonstrated a **capability**, not a general relevance victory. On held-out HotpotQA, untyped EBR retrieves both supporting paragraphs much more often than Porter BM25 (`both@10` +0.1052, CI [+0.0939,+0.1152]) and improves ordinary nDCG by +0.0225. That is a real multi-hop recall result. The mechanism is evidence-chain expansion: first-hop lexical seeds, broad cross-seed entity agreement, sparse postings, and fixed fusion.

It has not demonstrated that EBR is a better ranker. The cross-encoder gains +0.0995 nDCG over Porter on the neural subset while EBR captures only 27% of that gain. The “92% of neural gain” statement is valid only for HotpotQA `both@10` (`+0.108` versus `+0.118`), not for relevance generally. Also, the neural comparison is a 1,000-query subset, one model, and a candidate-pool comparison—not a claim that EBR matches RAG.

Code-graph EBR is promising in the narrower sense that exact relations can recover chains BM25 cannot. But the headline +0.320 `both@10` is a graph-derived synthetic upper bound, partly circular, on 14 files and 289 chunks. The clean non-circular-ish two-hop slice is +0.20, and 20% edge loss reduces the overall gain to about +0.10. TS7 improves parser correctness and the BM25 chunk baseline; it does not improve the fused retrieval result. That is an engineering-quality win, not evidence of a new ranking algorithm.

Therefore: **ship a specialist architecture, not a universal SERP scorer.** Keep BM25 as the safe default; expose explicit deep/multi-hop mode; use exact code/memory graphs behind domain-aware controls; keep a reranker optional; use truncation for packing unless a future experiment proves otherwise.

## What is actually supported

### Capability claims supported by the evidence

1. **Lexical retrieval can recover structurally missing evidence without embeddings.** EBR finds second-hop HotpotQA support outside the Porter pool; CE(EBR pool) improves over CE(Porter pool) by +0.0095 nDCG and +0.037 `both@10` on the same 1,000-query subset.
2. **Broad, redundant bridge agreement is useful on HotpotQA.** Untyped bridges work; relation-typed filtering removes the useful bridge candidates. Typed EBR has second-hop reach 0.162 versus 0.686 for untyped, and loses by 0.080 `both@10` at the fixed channel set.
3. **Exact structural graphs can execute code evidence chains.** The graph channel helps two-hop and import/export questions, and fusion beats Porter on the synthetic benchmark. This establishes executability under a graph, not real-world relevance.
4. **Parser quality and chunk boundaries matter.** TS7 has 20/20 sampled call precision versus regex 18/20 and removes obvious false edges. Its largest observed retrieval effect is improving the lexical baseline through chunk segmentation.
5. **Packing is evidence selection, not a demonstrated scoring improvement.** Existing results support truncation as the default; residual/P4 underfill or lose recall.

### Relevance/product claims not supported

1. EBR is not a universal BM25 replacement: SciFact nDCG is −0.044 and Touche R@100 is −0.037. A router limits exposure but does not identify which individual queries benefit; it gives back about one third of the Hotpot gain.
2. Relation typing is not currently a precision improvement. The attempted lexicon attacks the answer relation, while the useful bridge often belongs to a different hop.
3. The code graph is not validated on human or issue-derived relevance judgments. Its qrels are generated from the same graph used for retrieval, so the result must not be marketed as code-search quality.
4. No evidence supports “competes with RAG” without qualification. SERP competes on a selected multi-hop recall capability at low latency; it loses on ordinary relevance ordering to the reference cross-encoder.
5. No experiment supports a universal learned or corpus-statistical semantic rewrite. PPMI, WordNet, feedback, compression, and untyped co-occurrence mostly failed; sparse translation was contaminated by familiar training documents and still trailed the lexical control.

## Pursue versus kill, ranked by expected value per engineering week

| Rank | Work | Decision value | Recommendation |
|---|---|---|---|
| 1 | External multi-hop transfer with frozen EBR | Determines whether Hotpot is a workload-specific win | Do next |
| 2 | Real-repository graph benchmark with human/issue qrels | Determines whether code graph is product-worthy | Do next |
| 3 | Workload-mode safety and exact-ID invariants | Determines whether specialists can ship without hurting default search | Do in parallel |
| 4 | Relation-backed sparse alias/rewriting probe | Tests the only credible unexplored no-embedding semantic direction | One cheap falsification only |
| 5 | Optional reranker and packing integration | Product measurement after recall layer is stable | Do after 1–3 |

Kill now: universal EBR; relation-typed bridge filtering as implemented; router-as-benefit-predictor; another PPMI/WordNet/feedback/compression rerun; residual/P4 as the default packer; “TS7 changes retrieval quality” as a premise. Do not spend weeks tuning RRF weights or bridge counts before external transfer and real qrels exist.

## Concrete experiment plan

Each experiment has a separate runner and result file. Existing source files should be imported read-only unless a listed modification is unavoidable. Constants below are frozen before test execution.

### A1 — Multi-hop transfer, not another Hotpot run

**Goal:** test whether EBR is a general multi-hop capability or merely a HotpotQA artifact.

**Files:** create `experiments/stages/lab/multihop_transfer.py`, `experiments/stages/run_multihop_transfer.py`, and `experiments/stages/results/multihop_transfer.json`. Modify nothing in the existing EBR implementation; factor a read-only adapter if needed.

**Datasets:** 2WikiMultiHopQA and MuSiQue, using their official train/dev/test evidence annotations. Use dev only for wiring checks; report test once. If downloading both is too costly, run 2Wiki first and record the stop.

**Fixed constants:** Porter BM25 `k1=1.2,b=0.75`; 20 first-hop seeds; 20 bridge entities; 5% entity df cap; RRF `k=60`; depth 1,000; same capitalized/title extraction; no typed filtering; no router; no parameter search.

**Metrics:** supporting-evidence `both@5`, `both@10`, evidence recall@10/100, nDCG@10, MRR, bridge reach, and paired bootstrap CI versus Porter. Also report the fraction of qrels already in Porter top-100 and the extra EBR candidates. Run the same systems on a single-hop control subset if the dataset provides one.

**Acceptance gate:** keep the direction only if EBR has positive CI on the primary evidence-completeness metric on at least one external dataset and does not lose evidence recall@100 on the other; a statistically significant nDCG loss larger than −0.005 on both external sets kills it as a general mode. A result that improves only Hotpot remains a Hotpot specialist.

**Expected outcome:** positive evidence recall, smaller and less stable than Hotpot; nDCG likely flat or negative. **Falsifier:** no evidence-recall gain, or gains disappear when qrels are not graph-like/constructed around named entities.

### A2 — Real code search with graph edges and non-circular qrels

**Goal:** determine whether exact code graphs improve retrieval on real developer information needs rather than graph-generated questions.

**Files:** create `experiments/stages/lab/real_code_graph.py`, `experiments/stages/run_real_code_graph.py`, `experiments/stages/results/real_code_graph.json`, and a checked-in query/qrel fixture `experiments/stages/data/real_code_queries.jsonl`. Use TS7 graph extraction behind a pinned version; do not change the existing graph benchmark.

**Corpus:** start with 3 diverse JavaScript/TypeScript repositories (SERP plus two unrelated permissively licensed repos), frozen at commit hashes. Build 100–150 queries from issue titles, bug reports, commit messages, or existing code-search benchmark queries. Two human reviewers label relevant chunks; retain only queries with agreement or adjudicate disagreements. No generated qrels.

**Fixed constants:** TS7 `7.0.2`; AST-only graph plus checker-enabled graph as two predeclared variants; Porter top-10 graph seeds; two hops; existing edge weights and RRF `k=60`; top-100 evaluation; no query-specific routing learned from labels.

**Metrics:** nDCG@10, MRR, Recall@100, answer/chunk recall, paired deltas, per-repo and per-query-type slices, parser edge precision on a 50-edge hand audit, graph build time, index size, and 0/20/40% synthetic edge-drop robustness as a diagnostic only.

**Acceptance gate:** product graph mode must improve mean nDCG or Recall@100 with CI above zero on at least two repositories, have no repository with nDCG loss below −0.02, and retain ≥90% of Porter Recall@100. If it fails, keep the graph as an opt-in diagnostic/index feature, not a retrieval channel.

**Expected outcome:** modest gain concentrated in symbol/relationship queries; little or negative gain for plain lexical search. **Falsifier:** human qrels show no gain over Porter, or graph gains vanish on unrelated repositories despite high parser precision.

### A3 — Ship/kill the mode boundary and safety invariants

**Goal:** test a product policy, not a clever router: BM25 default, explicit `deep` mode, and optional graph mode must not silently degrade ordinary search.

**Files:** create `experiments/stages/lab/mode_matrix.py`, `experiments/stages/run_mode_matrix.py`, `experiments/stages/results/mode_matrix.json`, and `experiments/stages/data/mode_queries.jsonl`. Do not modify the existing router; treat it as a baseline row.

**Datasets:** HotpotQA held-out test, SciFact test, Touche test, NFCorpus dev (diagnostic), the exact-ID fixture, and A2's real code queries if available. Partition by corpus/workload, not by post-hoc query benefit.

**Fixed modes:** `default=Porter`; `deep=EBR`; `routed=existing frozen router`; `graph=Porter+graph` only for code corpus; no mixed global router training. Exact identifier gate is mandatory for digit/underscore queries. Keep RRF `k=60`.

**Metrics:** Hotpot `both@10`/`both@5`, BEIR nDCG@10/R@100/MRR, code Recall@100, exact hit/MRR, near-miss-before-exact, unknown-identifier negative activation, mode fire rate, p50/p95 latency, and index build/storage cost.

**Acceptance gate:** default mode must reproduce Porter within numerical tolerance; deep mode must be explicitly selected or selected by a corpus/workload configuration; on single-hop sets no automatic policy may have nDCG delta below −0.005 with CI excluding the margin; exact identifiers must always outrank variants and unknown identifiers must not activate gram/graph expansion. If these cannot be met, ship BM25 only and expose experiments behind a flag.

**Expected outcome:** explicit mode passes; universal per-query routing fails or is unnecessarily conservative. **Falsifier:** users cannot reliably identify when deep mode applies, or deep mode's latency/index cost is unacceptable relative to its evidence-recall gain.

### A4 — Cheap falsification of relation-backed sparse rewriting

**Goal:** test the one genuinely plausible no-embedding direction not yet adequately tested: **trusted, typed, revocable relations** (redirects, aliases, headings, definitions, or explicit knowledge-base edges), rather than corpus co-occurrence.

**Files:** create `experiments/stages/lab/relation_alias_probe.py`, `experiments/stages/run_relation_alias_probe.py`, `experiments/stages/results/relation_alias_probe.json`, and a small frozen `experiments/stages/data/relation_alias_fixture.jsonl` containing relation source, scope, polarity, query, target, and decoy.

**Data:** first use 100–200 naturally occurring Wikipedia redirect/title/anchor relations or an equivalent licensed alias source, with held-out target documents and hard co-occurrence decoys. Include the existing nonce bridge control, but do not count it as evidence of real-world relevance. No PPMI, WordNet, feedback, compression, or LLM extraction.

**Fixed constants:** exact relation postings only; one-hop expansion; source support must be present; no more than 8 aliases per term; Porter baseline and Porter+alias RRF `k=60`; no tuning on qrels; polarity/conflicting-edge rule fixed before evaluation.

**Metrics:** relation precision on a hand-audited sample, candidate recall@100, nDCG@10, false-match rate on decoys, unknown-term activation, source-removal revocation test, and latency/index size.

**Acceptance gate:** positive candidate recall with no nDCG loss beyond −0.005 on the held-out collection, false-match rate ≤5% on decoys, and removal of a source must remove its derived postings. If not, kill the semantic-rewrite direction rather than adding more extraction rules.

**Expected outcome:** improves lexical-gap recall where relations are explicit, but may not improve ranking because decoys remain. **Falsifier:** relation precision below 95%, or recall gains come with material false-match/nDCG regression. If it passes, it becomes a separate evidence channel, not a replacement scorer.

### A5 — Candidate pool, reranker, and packing product measurement

**Goal:** measure the useful composition: no-embedding recall first, optional neural reranker second, truncation packing last.

**Files:** create `experiments/stages/lab/pool_rerank_pack.py`, `experiments/stages/run_pool_rerank_pack.py`, and `experiments/stages/results/pool_rerank_pack.json`. Reuse existing frozen neural reference and packing controls read-only.

**Datasets:** the 1,000 fixed Hotpot neural subset, A2 real code queries if available, and the chunk-level JS benchmark. Do not expand the neural test set after inspecting results.

**Fixed constants:** cross-encoder exactly `cross-encoder/ms-marco-MiniLM-L-6-v2`, MPS, max length 512, batch 32, Porter/EBR union top-100; compare `Porter→CE` versus `EBR→CE`; packing budgets fixed to the existing benchmark values; truncation is control, residual/P4 unchanged.

**Metrics:** nDCG@10, both@10/both@5, Recall@100, CE pair count and ms/query, pool recall, packing recall/retention, selected relevant documents, exact-ID invariants, and total end-to-end latency.

**Acceptance gate:** EBR must improve pool recall and either improve CE quality or reduce the CE depth needed for the same `both@10`; truncation remains default unless another packer preserves at least as many relevant chunks at the same budget without exact-ID regressions. If EBR adds no pool value outside Hotpot, keep it only as a Hotpot/deep specialist.

**Expected outcome:** EBR is a valuable candidate generator, CE restores relevance ordering, and truncation wins packing. **Falsifier:** CE cannot exploit the extra EBR candidates, or EBR's latency/index cost dominates the quality gain.

## Product shape

1. **Default search:** Porter BM25, exact-ID safeguards, deterministic results, and ordinary truncation packing. Lexical char4/VC may remain a recall channel only where the independent Touche gate continues to pass; do not sell it as a better final ranker.
2. **Explicit deep/multi-hop mode:** EBR is opt-in or selected by a declared workload/corpus profile (“multi-hop QA”, “investigate related evidence”), not by a claim that a few question cues predict per-query benefit. Return provenance for bridge entities and supporting paths so users can inspect why a document was added.
3. **Code/memory graph mode:** index exact `CALLS`, `IMPORTS`, `EXPORTS`, `DEFINED_IN`, and containment edges. Use TS7 AST at build time with an exact version pin and readiness checks; checker resolution is opt-in. Fuse graph expansion with BM25 and gate it to relational/code tasks. Treat graph completeness as a published health metric.
4. **Optional reranker:** neural cross-encoder is a post-fusion reference/optional accelerator, clearly labelled as neural and outside SERP core. EBR's strongest role is to improve its candidate pool; it is not a substitute for relevance scoring.
5. **Packing:** retrieve broadly, then truncate in fused rank order under the token budget. Keep coverage-aware packers experimental until they beat truncation on real chunk qrels without underfill or identifier regressions.

## Top three falsifiers and cheapest tests

1. **The Hotpot gain does not transfer.** Run A1 on 2Wiki dev/test with constants unchanged. If evidence recall does not improve, narrow the claim to HotpotQA and likely stop general EBR investment.
2. **Graph gains are circular or parser-dependent.** Run A2 on one unrelated repository with 50 human-labeled issue queries; compare Porter, graph fusion, and 20% edge-drop. No human-qrel gain kills graph retrieval as a product feature.
3. **The safety/cost envelope fails.** Run A3 on SciFact/Touche plus a 1,000-query latency/index-cost sample. Any automatic policy with a significant nDCG loss, exact-ID violation, or unacceptable deep-mode cost means BM25-only default and specialist features behind explicit flags.

## Is there a better no-embedding direction?

Yes, but only one is worth a cheap test: **trusted relation-backed sparse retrieval**, where aliases/redirects/definitions/structured facts are explicit, typed, scoped, and revocable. This is materially different from PPMI or feedback because the bridge has an external support statement and can be audited or removed. It is also not magic: relation precision and decoy rejection are the bottleneck. A4 is deliberately a small falsification, not a new multi-week program. If it fails the precision and held-out ranking gates, the honest answer becomes “no better general no-embedding direction has been demonstrated”; focus on workload-specific graphs and candidate generation.

## Evidence and protocol notes

The requested `experiments/stages/PLAN.md` path is absent in this worktree; the plan used here is the repository-root `PLAN.md`. The first-principles and semantic-lab evidence lives in `/Users/asmit/Personal/algo/first_principles/FINDINGS.md` and `/Users/asmit/Personal/algo/semantic_lab/RESULTS.md`. Several existing results are explicitly exploratory or synthetic; this plan treats their limits as decision boundaries rather than trying to convert them into broad claims.
