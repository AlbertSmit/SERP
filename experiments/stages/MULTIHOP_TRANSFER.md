# A1: frozen multi-hop transfer of EBR (2WikiMultiHopQA, MuSiQue)

Frozen transfer test, not a tuning run. The EBR composition is imported read-only
from `lab/hotpot.py` (via `lab/multihop_transfer.py`); no file in the existing
implementation was modified. Constants: Porter BM25 `k1=1.2, b=0.75`; 20 first-hop
seeds; 20 bridge entities; 5% entity df cap; fixed RRF `k=60`; depth 1,000; no
typed filtering, no router, no parameter search, no neural reranker.

Split: labeled **dev** for both datasets. 2Wiki `test.parquet` is unlabeled and was
not used. A 200-question wiring slice verified the loaders before each full run.

Revision note: MuSiQue is reported twice. `musique` is the original
**title-identity** run (documents keyed by title; kept for comparison only).
`musique_paragraph` is the corrected **paragraph-identity** run: documents keyed by
`title + "\x00" + paragraph_text`, qrels are the supporting paragraphs themselves.
Only the corrected run counts toward the gate. 2Wiki is unchanged; its Porter
numbers are bit-identical to the first report and its bridge/fused numbers moved by
≤3.2e-4 because the unmodified channels accumulate weights in `set` order (now
pinned with `PYTHONHASHSEED=0`, recorded in the JSON).

## Dataset fit

| dataset | document identity | questions | documents | collisions | leak cases |
|---|---|---:|---:|---:|---:|
| 2Wiki dev | title | 12,576 | 54,957 | 2,506 | n/a |
| MuSiQue dev (corrected) | title+text | 2,417 | 21,100 | 3,471 | 0 |
| MuSiQue dev (superseded) | title | 2,417 | 17,629 | 9,843 | 626 |

MuSiQue corrected detail: 48,315 paragraph slots → 21,100 unique paragraphs
(27,215 duplicate slots); 1,090 titles carry more than one distinct paragraph;
3,471 extra paragraphs under repeated titles. These are legitimate distinct
documents, not identity defects. 0 missing support paragraphs, 0 questions with
collapsed support paragraphs (the `support_1` artifact is gone). 626 questions still
contain 1,794 **same-title confuser** paragraphs, but all differ from the gold text
(0 identical), so they are now correctly non-relevant hard negatives rather than
creditable gold.

2Wiki detail unchanged: 10 context paragraphs per question, 2 or 4 gold titles
(9,825 / 2,751), 2,506 repeated titles with different text, 32 questions repeating a
support title twice, 0 missing support titles.

Entity extraction is still the capitalized-span proxy. It fits both datasets (it
recovers the second comparison side on 2Wiki: entity channel `all@10` 0.941 vs Porter
0.910 on `comparison`), but it is a large drag when used alone.

## 2Wiki dev (title identity, unchanged; n=12,576)

| system | all@5 | all@10 | R@10 | R@100 | nDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Porter | 0.297 | 0.364 | 0.678 | 0.750 | 0.686 | 0.917 |
| bridge alone | 0.367 | 0.463 | 0.714 | 0.803 | 0.673 | 0.848 |
| per-entity alone | 0.247 | 0.263 | 0.559 | 0.571 | 0.571 | 0.751 |
| **EBR (RRF)** | **0.377** | **0.488** | **0.759** | **0.883** | **0.743** | **0.936** |

EBR vs Porter: all@5 +0.0799 [+0.0746,+0.0849] (1149/11283/144); all@10 +0.1240
[+0.1183,+0.1301] (1648/10840/88); R@10 +0.0814 [+0.0782,+0.0845]; R@100 +0.1328
[+0.1294,+0.1364]; nDCG +0.0571 [+0.0550,+0.0593] (5252/6341/983); MRR +0.0187
[+0.0158,+0.0218].

## MuSiQue dev, **corrected paragraph identity** (n=2,417) — this counts

| system | all@5 | all@10 | R@10 | R@100 | nDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Porter | 0.096 | 0.145 | 0.488 | 0.679 | 0.469 | 0.685 |
| bridge alone | 0.133 | 0.194 | 0.457 | 0.584 | 0.428 | 0.608 |
| per-entity alone | 0.030 | 0.043 | 0.323 | 0.382 | 0.345 | 0.576 |
| **EBR (RRF)** | **0.125** | **0.210** | **0.545** | **0.762** | **0.503** | **0.700** |

| metric | Δ EBR−Porter | 95% CI | W/T/L |
|---|---:|---|---:|
| all@5 | +0.0298 | [+0.0190,+0.0414] | 127/2235/55 |
| all@10 | +0.0650 | [+0.0517,+0.0803] | 236/2102/79 |
| recall@10 | +0.0567 | [+0.0478,+0.0666] | 572/1622/223 |
| recall@100 | +0.0826 | [+0.0740,+0.0915] | 608/1684/125 |
| nDCG@10 | +0.0332 | [+0.0271,+0.0395] | 950/951/516 |
| MRR | +0.0151 | [+0.0045,+0.0250] | 577/1414/426 |

Bridge alone loses nDCG (−0.0411 [−0.0478,−0.0341]); entity alone loses heavily
(−0.1243). The fusion, not either channel, is the system.

## MuSiQue dev, title identity (superseded; shown for comparison)

| system | all@5 | all@10 | R@10 | R@100 | nDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Porter | 0.084 | 0.110 | 0.418 | 0.580 | 0.409 | 0.619 |
| bridge alone | 0.122 | 0.168 | 0.396 | 0.493 | 0.378 | 0.545 |
| per-entity alone | 0.041 | 0.052 | 0.317 | 0.356 | 0.346 | 0.581 |
| EBR (RRF) | 0.111 | 0.173 | 0.488 | 0.670 | 0.459 | 0.661 |

EBR vs Porter (title identity): all@10 +0.0629 [+0.0505,+0.0757], R@10 +0.0702
[+0.0607,+0.0800], R@100 +0.0893 [+0.0808,+0.0984], nDCG +0.0498
[+0.0437,+0.0560], MRR +0.0419 [+0.0324,+0.0511].

**The defect did not manufacture the gain.** all@10 delta is +0.0629 (title) vs
+0.0650 (paragraph); absolute EBR all@10 rose from 0.173 to 0.210 only because
Porter's rose from 0.110 to 0.145. nDCG@10 gain is smaller under correct labels
(+0.0332 vs +0.0498) and MRR gain is roughly a third of the title-level number
(+0.0151 vs +0.0419), which is the expected direction: some of the title-level
nDCG/MRR credit was coming from the wrong paragraph.

## Slices (all@10 / R@10 / nDCG@10)

2Wiki: `bridge_comparison` (n=2,751) Porter 0.009/0.506/0.579, bridge 0.031,
EBR 0.032/0.593/0.654. `comparison` (n=3,040) 0.910/0.954/0.893, bridge 0.590,
EBR 0.989/0.995/0.941. `compositional` (n=5,236) 0.227/0.606/0.631, bridge
0.570, EBR 0.414/0.705/0.683. `inference` (n=1,549) 0.383/0.684/0.658, bridge
0.617, EBR 0.560/0.776/0.719.

MuSiQue corrected, by gold paragraph count: `support_2` (n=1,252)
0.248/0.568/0.533, bridge 0.339, EBR 0.343/0.623/0.561. `support_3` (n=760)
0.050/0.461/0.453, bridge 0.057, EBR 0.096/0.529/0.495. `support_4` (n=405)
0.002/0.291/0.304, bridge 0.005, EBR 0.010/0.331/0.337.

The aggregate `all@10` gain still mixes regimes: 4-gold completeness is 1% (2Wiki
`bridge_comparison` 3.2%) after correction. The headline is the 2-gold regime.

## Bridge reach and candidate cost

| | 2Wiki | MuSiQue corrected | MuSiQue title (old) |
|---|---:|---:|---:|
| gold paragraphs | 30,654 | 6,404 | 6,287 |
| gold in Porter top-100 | 0.7168 | 0.6558 | 0.5551 |
| gold outside Porter top-20 | 9,944 | 3,065 | 3,520 |
| bridge retrieves at all | 0.6314 | 0.3732 | 0.2812 |
| bridge retrieves in top-10 | 0.3310 | 0.1240 | 0.0986 |
| extra EBR candidates / query | 54.0 | 45.5 | 44.9 |
| extra as fraction of EBR top-100 | 0.5398 | 0.4545 | 0.4493 |

Corrected MuSiQue reach is **higher** than the title-level number (0.373 vs 0.281),
because paragraph-level qrels no longer ask the bridge channel to find an entire
article when only one paragraph was needed.

Pooled over the two counting datasets (2Wiki + corrected MuSiQue, n=14,993):
Porter 0.264/0.328/0.647/0.739/0.651/0.880 → EBR 0.336/0.443/0.725/0.864/0.705/0.898.
EBR all@10 +0.1145 [+0.1088,+0.1199], R@100 +0.1247 [+0.1211,+0.1278],
nDCG +0.0532 [+0.0512,+0.0552].

## Latency (CPU, full dev)

2Wiki: Porter 4.81 ms/query, EBR 10.32 ms/query; index build 6.9 s; eval 192 s.
MuSiQue corrected: 1.99 / 4.52 ms/query; build 2.4 s; eval 16 s.

## Acceptance gate and verdict

Pre-registered rule: *keep the direction only if EBR has a positive CI on `all@10`
or `recall@10` on at least one external dataset AND no nDCG@10 loss larger than
−0.005 with a CI excluding that margin on either dataset.* Basis: `2wiki` +
`musique_paragraph` (the title-level block is excluded).

Applied mechanically: positive CIs on both `all@10` and `recall@10` on both
datasets (2Wiki +0.1240 / +0.0814; MuSiQue corrected +0.0650 / +0.0567); nDCG@10 CIs
are [+0.0550,+0.0593] and [+0.0271,+0.0395], so `ndcg_failures` is empty. recall@100
improves on both, satisfying the stricter plan wording.

**Gate PASSED → keep the direction.** MuSiQue still supports the transfer claim under
paragraph identity; the effect is smaller on nDCG/MRR but not on evidence
completeness. `all@k` remains primary, and the nDCG gain is not a claim of better
general relevance ranking (Porter nDCG is low here, 0.686 / 0.469).

## Files

- `experiments/stages/lab/multihop_transfer.py` (loaders, metrics, gate)
- `experiments/stages/run_multihop_transfer.py` (runner)
- `experiments/stages/results/multihop_transfer.json` (all three blocks, exact numbers)

Reproduce (bit-identical): `PYTHONHASHSEED=0 .venv/bin/python
experiments/stages/run_multihop_transfer.py` (`--limit 200` for the wiring slice).
