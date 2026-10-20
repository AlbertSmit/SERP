# Relation-typed EBR on HotpotQA

**Verdict: typing hurts.** A fixed relation lexicon applied to the bridge channel
is markedly worse than the untyped bridge on every configuration, on every metric,
on the held-out test split. It does not only lose recall: on the queries where it
fires, the typed bridge is a *worse* set of entities than a random subset of the
untyped bridge at the same size. The null/negative result is reported rather than
tuned away. The one positive number (a both@10 gain for `ebr_typed_plus_coverage`
over Porter) is smaller than the untyped system's, and lives entirely in the
`relation_unknown` fallback slice.

Files:

- `lab/hotpot_typed.py` - relation lexicon, extractor, question detector, typed
  bridge channel. `lab/hotpot.py` is imported read-only and unchanged.
- `run_hotpot_typed.py` - the paired evaluation.
- `results/hotpot_typed.json` - every table below, plus per-relation slices.

## What was built

* **Lexicon.** 82 cue phrases over 20 relation types: `directed_by`, `written_by`,
  `starring`, `born_in`, `located_in`, `member_of`, `part_of`, `founded_by`,
  `owned_by`, `produced_by`, `composed_by`, `educated_at`, `capital_of`,
  `country`, `city`, `author`, `publisher`, `award`, `occupation`, `nationality`.
  Regex over raw paragraph text. The object of a cue is the capitalized span(s)
  after it, cut at the next clause or participial phrase; the subject is the
  title-matching span of the clause, else the first one.
* **Question detector.** 20 regexes, most-specific first, max two types per
  question. Fires on **2,366 / 6,005 test questions (39.4%)**.
* **Typed bridge.** The untyped candidate set is *filtered*, never re-scored:
  same `idf(entity)/(seed_rank+1)` weight, same 20 seeds, same top-20, same 5% df
  cap. Three rules:
  * `argument` - entity is the subject or object of a detected-type relation in
    the seed paragraph.
  * `link` - a detected-type relation in the seed paragraph connects a query
    entity to the entity.
  * `union` (= pre-registered `ebr_typed`) - `argument` or `link`.
  * `anchor` (diagnostic) - like `link` but for *any* relation type, not only the
    one the question asks about.
* **Fallback.** No detected relation means the untyped bridge is used unchanged,
  so typed and untyped systems differ only on the detected slice.

Constants are inherited and untouched: RRF k=60, depth 1000, 20 seeds, 20
bridges, 5% df cap, 1,000 bootstrap resamples. Dev was a sanity check only; every
delta below has the same sign and similar magnitude on dev (e.g. `ebr_typed`
both@10 0.5636 dev / 0.5722 test, second-hop reach on the detected slice 0.127
dev / 0.162 test).

## Relation coverage (66,581 paragraphs)

| relation | paragraphs | instances | | relation | paragraphs | instances |
|---|---:|---:|---|---|---:|---:|
| nationality | 15,467 | 15,183 | | country | 1,609 | 2,087 |
| city | 6,493 | 6,217 | | occupation | 1,582 | 1,720 |
| starring | 5,684 | 5,991 | | born_in | 1,524 | 966 |
| located_in | 5,556 | 5,914 | | author | 1,218 | 1,664 |
| directed_by | 4,520 | 4,438 | | owned_by | 1,010 | 852 |
| part_of | 4,030 | 4,244 | | educated_at | 749 | 663 |
| written_by | 3,207 | 3,241 | | composed_by | 584 | 511 |
| member_of | 3,048 | 3,184 | | capital_of | 317 | 346 |
| produced_by | 2,373 | 2,367 | | | | |
| award | 2,097 | 2,613 | | | | |
| founded_by | 1,852 | 1,908 | | | | |
| publisher | 1,723 | 2,069 | | | | |

39,124 / 66,581 paragraphs (58.8%) contain at least one relation instance.
`nationality` and `occupation` are the two types the capitalized-span design
cannot use: their fillers are adjectives and lower-case role nouns, and
nationality adjectives sit above the 5% df cap anyway.

## Results: global pool, held-out test (6,005 questions)

| system | nDCG@10 | both@10 | both@5 | R@10 |
|---|---:|---:|---:|---:|
| Porter BM25 | 0.7331 | 0.6113 | 0.4633 | 0.7967 |
| **`ebr_untyped`** (RRF Porter+bridge+coverage) | **0.7556** | **0.7166** | **0.5306** | **0.8493** |
| RRF(Porter, bridge_untyped) | 0.7202 | 0.6525 | 0.5019 | 0.8073 |
| bridge_untyped alone | 0.6631 | 0.5319 | 0.4243 | 0.7320 |
| `ebr_typed` (RRF Porter+bridge_typed) | 0.6366 | 0.5722 | 0.4123 | 0.7538 |
| `ebr_typed_plus_coverage` | 0.7124 | 0.6511 | 0.4751 | 0.8137 |
| `ebr_typed_link_plus_coverage` | 0.7270 | 0.6641 | 0.4901 | 0.8218 |
| `ebr_typed_anchor_plus_coverage` | 0.7250 | 0.6633 | 0.4893 | 0.8213 |
| `typed_bridge_only` (ablation) | 0.4942 | 0.3794 | 0.3087 | 0.5460 |

`ebr_untyped` reproduces `results/hotpot.json` exactly (0.7556 / 0.7166 / 0.5306 /
0.8493), so the comparison is against the real frozen system.

### Paired bootstrap vs Porter BM25 (test, global)

| system | metric | Δ | 95% CI | W/T/L |
|---|---|---:|---|---:|
| `ebr_untyped` | nDCG@10 | +0.0225 | [+0.0184, +0.0266] | 2146/2415/1444 |
| | both@10 | +0.1052 | [+0.0939, +0.1152] | 901/4835/269 |
| | both@5 | +0.0673 | [+0.0563, +0.0793] | 777/4855/373 |
| | R@10 | +0.0526 | [+0.0468, +0.0583] | 953/4735/317 |
| `ebr_typed` | nDCG@10 | **−0.0965** | [−0.1020, −0.0910] | 1101/2266/2638 |
| | both@10 | **−0.0391** | [−0.0510, −0.0281] | 536/4698/771 |
| | both@5 | **−0.0510** | [−0.0618, −0.0401] | 464/4771/770 |
| | R@10 | −0.0429 | [−0.0495, −0.0361] | 552/4485/968 |
| `ebr_typed_plus_coverage` | nDCG@10 | −0.0208 | [−0.0259, −0.0156] | 1839/2173/1993 |
| | both@10 | +0.0398 | [+0.0296, +0.0506] | 696/4852/457 |
| | both@5 | +0.0118 | [−0.0000, +0.0235] | 675/4726/604 |
| | R@10 | +0.0171 | [+0.0112, +0.0231] | 747/4734/524 |
| `typed_bridge_only` | nDCG@10 | −0.2389 | [−0.2475, −0.2302] | 855/1479/3671 |
| | both@10 | −0.2320 | [−0.2458, −0.2170] | 493/3626/1886 |
| | both@5 | −0.1545 | [−0.1672, −0.1414] | 454/4169/1382 |
| | R@10 | −0.2507 | [−0.2610, −0.2407] | 499/2778/2728 |
| `ebr_typed_link_plus_coverage` | nDCG@10 | −0.0061 | [−0.0111, −0.0011] | 1863/2327/1815 |
| | both@10 | +0.0528 | [+0.0418, +0.0634] | 708/4906/391 |
| | both@5 | +0.0268 | [+0.0148, +0.0388] | 684/4798/523 |
| | R@10 | +0.0251 | [+0.0192, +0.0307] | 757/4803/445 |

### The typing delta at a fixed channel set

Comparing typed and untyped bridges inside the *same* fusion (Porter + one bridge
channel), which isolates typing from the coverage channel:

| comparison | metric | Δ | 95% CI |
|---|---|---:|---|
| `ebr_typed` vs RRF(Porter, bridge_untyped) | nDCG@10 | −0.0836 | [−0.0889, −0.0784] |
| | both@10 | −0.0803 | [−0.0888, −0.0718] |
| | both@5 | −0.0896 | [−0.0979, −0.0813] |
| `ebr_typed_link` (rule b only) vs same | nDCG@10 | −0.0231 | [−0.0266, −0.0194] |
| | both@10 | −0.0373 | [−0.0446, −0.0300] |
| `ebr_typed_anchor` (any relation, rule b) vs same | nDCG@10 | −0.0303 | [−0.0340, −0.0267] |
| | both@10 | −0.0405 | [−0.0480, −0.0325] |
| `ebr_typed_anchor_plus_coverage` vs `ebr_untyped` | nDCG@10 | −0.0306 | [−0.0338, −0.0272] |
| | both@10 | −0.0533 | [−0.0604, −0.0463] |

Every rule is negative. Restricting to query-linked bridges (rule b) is less bad
than admitting every relation argument (rule a), but it is still negative, and it
is empty for 61.5% of detected questions (mean 1.5 bridges vs 20 untyped).

## Slices: where the loss happens

| system | relation_detected (n=2,366) nDCG / both@10 / both@5 | relation_unknown (n=3,639) nDCG / both@10 / both@5 |
|---|---|---|
| Porter | 0.7062 / 0.5448 / 0.4100 | 0.7506 / 0.6546 / 0.4979 |
| `ebr_untyped` | 0.7376 / 0.6750 / 0.4894 | 0.7673 / 0.7436 / 0.5573 |
| `ebr_typed` | 0.4937 / 0.4260 / 0.2485 | 0.7295 / 0.6672 / 0.5188 |
| `ebr_typed_plus_coverage` | 0.6278 / 0.5089 / 0.3487 | 0.7673 / 0.7436 / 0.5573 |
| `typed_bridge_only` | 0.2212 / 0.1293 / 0.1019 | 0.6718 / 0.5419 / 0.4433 |

The `relation_unknown` column is the fallback: `ebr_typed` equals
RRF(Porter, bridge_untyped) and `ebr_typed_plus_coverage` equals `ebr_untyped`
to four decimals, confirming the fallback is exact. **All of the loss is on the
detected slice**, and there it is catastrophic: both@10 −0.1188 vs Porter
[−0.1365, −0.1010], nDCG −0.2125 [−0.2236, −0.2012]. The untyped system gains
+0.1302 both@10 on the same questions. Typing does not "trade precision for
recall"; it inverts the gain.

## Why it failed: the bridge hop is not the question's relation

For 859 gold second-hop documents on the detected test slice, does the bridge
channel retrieve them at all? These reach figures come from a separate diagnostic
run; they are not stored in `results/hotpot_typed.json`.

| bridge set | second-hop reach |
|---|---:|
| untyped, 20 bridges | 0.686 |
| untyped, top 11 (same budget, no typing) | 0.586 |
| random 11 of the untyped 20 | 0.446 |
| `ebr_typed` union (10.6 bridges) | **0.162** |
| `link` rule (1.5 bridges) | 0.075 |
| `anchor` rule (any relation) | 0.091 |

Reached by typed only: 2.0% of gold docs. Reached by untyped only: 54.4%.

A random subset of the same size reaches **2.8x more** second-hop evidence than
the relation-typed subset. The failure is selection, not sparsity, and three
mechanisms cause it:

1. **Wrong hop.** The relation a question asks about is usually the relation of
   the *last* hop, not the bridge hop. "What was the nationality of the actor
   playing a Russian hostage taker in *Hostage for a Day*?" detects `nationality`,
   so `John Candy` (the actor, and the gold bridge, reached by `starring`) is
   filtered out and only nationality adjectives survive. The `link` rule fails on
   the same questions for the same reason.
2. **Lexicon miss.** Corpus phrasing is far wider than question phrasing:
   "*Persian Surgery Dervishes*" says "an album by Terry Riley", "Ralph
   Hefferline" says "professor at Columbia University". Neither `album by` nor
   `professor at` is in the lexicon, so the correct bridge is never admitted.
3. **Answer-shaped arguments.** The arguments that do survive are places, award
   names and dates - the answer type, not the linking entity. They also skew to
   single-seed high-idf spans, destroying the cross-seed consensus that makes the
   untyped channel work.

## Per-relation: no slice is helped

Δ `ebr_typed` vs RRF(Porter, bridge_untyped), both@10; n = test questions whose
primary detected type is that relation.

| relation | n | Δ both@10 | Δ nDCG@10 | | relation | n | Δ both@10 | Δ nDCG@10 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| located_in | 316 | −0.288 | −0.290 | | publisher | 90 | −0.300 | −0.271 |
| starring | 308 | −0.205 | −0.159 | | award | 84 | −0.429 | −0.395 |
| born_in | 294 | −0.156 | −0.186 | | composed_by | 74 | −0.135 | −0.179 |
| founded_by | 238 | −0.139 | −0.181 | | member_of | 71 | −0.070 | −0.137 |
| directed_by | 166 | −0.181 | −0.191 | | nationality | 51 | −0.176 | −0.194 |
| city | 128 | −0.273 | −0.324 | | author | 44 | −0.136 | −0.140 |
| written_by | 108 | −0.241 | −0.182 | | owned_by | 44 | −0.227 | −0.208 |
| country | 104 | −0.048 | −0.055 | | educated_at | 27 | −0.222 | −0.220 |
| part_of | 102 | −0.216 | −0.266 | | capital_of | 13 | −0.462 | −0.433 |
| produced_by | 93 | −0.183 | −0.228 | | occupation | 11 | +0.091 | +0.061 |

`occupation` is the only non-negative cell, on 11 questions, and its nDCG is
still negative. `country` is the least-bad large cell, consistent with place
names being one thing the `city in` / `located in` cues genuinely capture.

## 10-paragraph pool (standard distractor setting)

`both@10` is saturated, so `both@5` is the informative metric.

| system | nDCG@10 | both@5 |
|---|---:|---:|
| Porter BM25 | 0.8228 | 0.5334 |
| `ebr_untyped` | **0.8363** | **0.6486** |
| `ebr_typed` | 0.7907 | 0.5769 |
| `ebr_typed_plus_coverage` | 0.8176 | 0.6083 |
| `typed_bridge_only` | 0.7401 | 0.5314 |

Paired: `ebr_typed_plus_coverage` vs Porter nDCG −0.0052 [−0.0088, −0.0016],
both@5 +0.0749 [+0.0633, +0.0863]; typed vs untyped bridge both@5 −0.0501
[−0.0580, −0.0428]. Same direction, smaller magnitude - the loss is not a global
candidate-pool artifact.

## Limits

- The typed channel is fused with the same fixed RRF weight as a full channel.
  A sparser channel placing 10 instead of 20 documents into the top ranks is
  structurally penalised. Re-weighting RRF by channel size would be a different,
  tuned system, so it was not tried.
- One dataset, one language, one lexicon. The cue list was fixed before the test
  run; the obvious missing cues (`album by`, `professor at`, `worked for`) were
  deliberately not added after seeing dev.
- Subjects come from a positional heuristic, so the `link` rule is noisier than
  the `argument` rule. That does not rescue typing: `link` is also negative.
- This establishes that *this* relation-typed bridge does not work on HotpotQA.
  It does not establish that relation typing cannot work; it localises the
  problem to hop alignment and lexicon coverage.

## What the negative result implies

- The gain in untyped EBR comes from **broad, redundant, cross-seed entity
  agreement**, not from per-entity precision. Pruning the candidate set attacks
  the mechanism that produces the gain.
- A relation index would have to be built for the **first hop** (how the seed
  mentions the next entity), not for the answer relation, and it would need
  open-ended cue coverage rather than a 82-phrase list.
- Precision filters must run *after* fusion (rerank or gate), where they can
  remove a document without deleting a channel's evidence.

## Reproduce

```sh
cd /Users/asmit/Personal/algo-serp-stages
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_hotpot_typed.py
```

First run builds `data/.cache_hotpot_typed.pkl` (~35 s, 179 MB); later runs load
it in ~2.5 s. Total runtime ~10 min for dev + test. A bug-fix rerun is the only
allowed rerun; no constant was changed after the test run.
