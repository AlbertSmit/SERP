# Algorithm notes and reported benchmarks

## SubConPack 2.0 retrieval rules

The JavaScript engine uses positive RSJ IDF and includes every substantive query
token in its information-demand denominator. Unknown tokens receive the fixed
prior `log1p(M / 2)`. English and Dutch stopwords do not contribute demand.

Chunk normalization uses the live corpus mean length. Exact header matches receive
a 1.4 multiplier; prefix, stem, and compound matches receive a 0.85 multiplier.
Candidate pooling uses `max(floor(0.4M), 25)`. Vaults of at most 100 chunks add a
trigram-Jaccard fallback at 0.45 and abstain below it.

Packing first requires weighted query coverage (default `min_coverage = 0.315`).
A shaped rare identifier may bypass that gate only when it contains `_` or a digit
and occurs in at most `max(2, floor(0.02M))` documents. Packing stops below 8%
residual demand, never admits a zero-gain candidate, and uses a first-marginal
relative dropoff of `0.26`. Both coverage and dropoff are exposed as snake-case
options for development-only tuning.

### Staged interfaces

The engine also exposes retrieval as separate stages:

- `retrieveRanked` returns `{ word, gram, fused }`.
- `rankWordBm25` uses the existing word FTS index; `rankGramBm25` uses a separate
  in-word character-4-gram index with word-boundary markers (`1` and `2`).
- `fuseRrf` combines ranked lists with reciprocal-rank fusion (`k = 60`). Chunks
  containing an exact identifier token (a query token with a letter and a digit
  or underscore that exists as a term) sort before variant-only matches,
  regardless of fused score. This is a hard rule, not a weight.
- `packRanked` applies the coverage gate and a budget with one of four packers:
  `truncate`, `greedy`, `residual`, or `p4`. All four see the same
  `poolDepth` candidate pool.
- `packHybrid` composes retrieve, fuse, and pack.

The gram channel filters per token: identifier-like tokens (containing `_` or a
digit) contribute grams only when they exist in `chunks_exact_terms`. The word
channel filters them too: unicode61 would split `figure_1` into `figure` + `1`
and match prose that never contained the identifier. The exact table stores
`tokenize()` output, so `figure_1` is known only when it was indexed as a token.
`buildGramIndex` and `buildExactIndex` backfill the gram and exact tables for
pre-existing databases; constructors do not backfill automatically.

[Back to the README](../README.md).

---

## Historical: the P4 track (superseded)

> **Status.** Everything below this line records the original P4 research and is
> kept for history. It is not current evidence. The staged protocol
> (`experiments/stages/`) superseded it:
>
> - On the matched chunk benchmark (`experiments/stages/results/stage-benchmark.json`,
>   SciFact test, 300 queries, budget 250), current P4 packing reaches doc recall
>   0.491 using about 47 tokens, against 0.601 for RRF truncation using 166. At
>   matched tokens, truncation still wins. P4 is an experimental packer, not the
>   default, and there is no evidence that P4 beats BM25.
> - The tables in section 4 use a combined two-query "budget recall" metric on
>   SciFact, which was also used for development. They were never rerun under the
>   stage protocol, and they have no CIs.
> - "What we invented" overstates it. The stretched-exponential saturation,
>   burstiness weighting and residual decay were not shown to beat BM25 on
>   independent labelled data. Bridge expansion is a variant of known entity-based
>   multi-hop retrieval. The measured multi-hop result is EBR (see the README).
> - No universal no-embedding scorer was found. See `LEXICAL_PORTFOLIO.md` and
>   `SELECTIVE_LEXICAL.md`.
>
> The current positioning, numbers and CIs are in the [README](../README.md#benchmarks).

## 1. Motivation: Why BM25 Fails in MCP Memory & Agentic RAG

Classical lexical retrieval (Okapi BM25, 1994) was designed to rank isolated documents for human web search (`Top-K` ranking). Modern LLM agents operating via **Model Context Protocol (MCP)** or **RAG** face a fundamentally different mathematical optimization problem:

1. **Strict Token Budgets (`BudgetRecall@B`):** An agent cannot afford to waste a 120–250 token context budget on 4 chunks that repeat the exact same keywords for the first half of a query while starving the second half.
2. **Boilerplate & Log Spam (`L / U` Concentration):** Agent memory stores and codebases contain JSON dumps, stack traces, and repetitive logs that game standard Term Frequency (`TF`) saturation.
3. **Uniform Saturation Fallacy:** BM25 uses a single static saturation constant (`k1 = 1.2`) for every word in the dictionary, treating globally clumpy/bursty words identically to rare Poisson events.
4. **Unshared Vocabulary Hops (1-Hop Bridges):** Multi-sentence evidence often requires hopping from a seed chunk (`Chunk A`) to an explanatory chunk (`Chunk B`) via a shared technical entity that was not present in the user's prompt.

---

## 2. How This Algorithm Was Discovered

Standard LLM prompting fails to invent new retrieval algorithms because keywords like _"query"_, _"document"_, _"TF"_, and _"IDF"_ trigger mode collapse toward textbook BM25 in the model's weights.

To discover a genuine successor, we used an **Evolutionary Verifier Loop (FunSearch / AlphaEvolve paradigm)** with a frontier reasoning model (`Inkling` / `Nemotron 3 Ultra`) under three strict constraints:

1. **Domain De-Contextualization:** Framed the problem as _Sparse Discrete Signal Detection_ over multi-channel event packets (`c_i`, `L`, `U`, `N_i`, `T_i`, `M`).
2. **Negative Algebraic Constraints:** Strictly banned the classical Michaelis-Menten / Hill rational saturation fraction `c / (c + K)` and linear length normalization `1 - b + b*(L / L_mean)`.
3. **Locked Empirical Ground-Truth Verifiers:** Evaluated mutations and tuned continuous parameters via `scipy.optimize` on a strict **120 Dev / 180 Holdout** split of the **BEIR SciFact** corpus (32,297 total chunks, including 1,500 injected repetitive log/spam noise chunks).

---

## 3. P4 design components (historical claims)

### Discovery 1: Unit-Anchored Stretched-Exponential Potential (`P4`)

Instead of BM25's rational `c / (c + k1)` curve, our evolutionary search discovered a **Kohlrausch / Weibull Stretched-Exponential Potential** with exponent $p = 0.45$, anchored to unity (`1.0`) at $c_{\text{eff}} = 1$ so single-occurrence matches preserve exact Shannon self-information:

$$\phi_i(c_i, L) = \frac{1 - \exp\left(-\beta_i \cdot c_{\text{eff}}^{0.45}\right)}{1 - \exp(-\beta_i)}, \quad c_{\text{eff}} = \frac{c_i}{(L / \bar{L})^{\delta_l}}$$

### Discovery 2: Zero-Cost Burstiness Adaptation via SQLite `fts5vocab`

Standard SQLite `bm25()` only uses document frequency ($N_i$) and ignores total corpus occurrences ($T_i$). By attaching SQLite's built-in `fts5vocab` virtual table (`row` mode), we extract both `doc` ($N_i$) and `cnt` ($T_i$) in $O(1)$ time to compute each word's **Global Conditional Burstiness** $\mu_i = T_i / N_i$, dynamically regulating each word's saturation rate:

$$\beta_i = \beta_0 \left(\frac{T_i}{N_i}\right)^{-\gamma_b}$$

Rare, non-bursty identifiers ($\mu_i \approx 1$) saturate immediately on their first occurrence, whereas clumpy domain terms ($\mu_i \gg 1$) saturate more gradually.

### Discovery 3: Rectified Asymmetric $L/U$ Boilerplate Filter

By storing unique token cardinality $U$ alongside token length $L$ at ingestion time, the engine computes intra-chunk token concentration $L/U$. A rectified hyperbolic tangent penalty stays dormant (`1.0`) on normal prose ($L/U \le \mu_{\text{conc}}$) and aggressively suppresses repetitive log/JSON spam:

$$\Psi_{\text{conc}}(L, U) = \exp\left(-\delta_c \tanh\left(\eta \cdot \max\left(0, \frac{L/U - \mu_{\text{conc}}}{\sigma_{\text{conc}}}\right)\right)\right)$$

### Discovery 4: Word-Level Submodular Residual Decay (`pack_context`)

To pack maximum distinct facts into a strict token budget $B$, the engine maintains a dynamic residual Shannon demand vector $r_w$ for each query term $w \in Q$:

$$r_w^{(0)} = q_w \ln\left(1 + \frac{M - N_w + 0.5}{N_w + 0.5}\right)$$

At each greedy step, the engine selects the chunk $P^*$ that maximizes **Marginal Information Gain per Effective Token Cost**:

$$\text{Gain}(P \mid r^{(t)}) = \frac{\Psi_{\text{doc}} \cdot \Psi_{\text{conc}} \cdot \sum_{w \in Q \cap P} r_w^{(t)} \cdot \phi_w(c_w, L)}{(L + 10)^{0.35} \cdot (L / U)^{0.15}}$$

Immediately after selecting $P^*$, the residual demand of every covered query word decays exponentially:

$$r_w^{(t+1)} = r_w^{(t)} \cdot \exp\left(-\lambda_{\text{decay}} \cdot \phi_w(c_w, L)\right) \quad (\lambda_{\text{decay}} = 2.2)$$

Because covered query words lose ~88% of their weight after a single strong match, subsequent picks are mathematically forced to pivot to uncovered aspects of the user's query.

### Discovery 5: Speculative 1-Hop Lexical Bridge Chaining (`expand_bridges`)

For multi-hop queries where required evidence chunks do not share keywords with the prompt, Stage 1 extracts non-query candidate bridge symbols $b \notin Q$ from the top-3 seed chunks and ranks them by **Anomalous Bridge Salience**:

```math
\Psi_{\text{bridge}}(b) = (\operatorname{seedHits}(b))^{1.5} \cdot \ln\left(\frac{M}{N_b}\right) \cdot \left(\frac{T_b}{N_b}\right)^{-0.4}
```

The top-5 high-surprise, low-clumpiness bridge entities are injected into the Stage-2 candidate pool after primary seed selection, recovering 1-hop evidence in milliseconds without vector embeddings or GraphRAG.

---

## 4. Reported P4 benchmark (historical, not rerun)

Evaluated across **300 queries** (**120 Dev / 180 Unseen Holdout**) over **32,297 chunks** (27,114 atomic sentence chunks, 5,183 full documents, and 1,500 synthetic log-spam chunks):

### Unseen Holdout Split (180 Queries)

| Method                                        | `BudgetRecall@120` (3–5 Chunks) | `BudgetRecall@250` (5–8 Chunks) | `BridgeRecall@120` (1-Hop Masked) | `TemporalF1@250` (Noise + Aging) | P50 Latency    |
| --------------------------------------------- | ------------------------------- | ------------------------------- | --------------------------------- | -------------------------------- | -------------- |
| **Method 1: Native SQLite `bm25()` Top-K**    | `0.1111`                        | `0.1333`                        | `0.0000`                          | `0.5276`                         | `177.9 ms`     |
| **Method 2: `bm25() / L` Length-Normalized**  | `0.0889`                        | `0.1889`                        | `0.0000`                          | `0.6216`                         | `224.2 ms`     |
| **Method 3: `P4` + Submodular Packer (Ours)** | **`0.1889` (+70.0%)**           | **`0.3111` (+133.4%)**          | `0.0000`                          | **`0.6482` (+22.9%)**            | **`127.2 ms`** |
| **Method 4: `P4` + 1-Hop Bridge + Temporal**  | `0.1444` (+30.0%)               | `0.1556` (+16.7%)               | **`0.0222` (Active)**             | `0.5098`                         | `278.8 ms`     |

### Dev Split (120 Queries)

| Method                                        | `BudgetRecall@120`    | `BudgetRecall@250`    | `BridgeRecall@120`    | `TemporalF1@250`      | P50 Latency    |
| --------------------------------------------- | --------------------- | --------------------- | --------------------- | --------------------- | -------------- |
| **Method 1: Native SQLite `bm25()` Top-K**    | `0.2833`              | `0.2500`              | `0.0000`              | `0.6027`              | `168.0 ms`     |
| **Method 2: `bm25() / L` Length-Normalized**  | `0.2333`              | `0.3000`              | `0.0000`              | `0.6201`              | `211.4 ms`     |
| **Method 3: `P4` + Submodular Packer (Ours)** | **`0.3833` (+35.3%)** | **`0.4333` (+73.3%)** | `0.0000`              | **`0.6803` (+12.9%)** | **`114.9 ms`** |
| **Method 4: `P4` + 1-Hop Bridge + Temporal**  | `0.3000` (+5.9%)      | `0.3167` (+26.7%)     | **`0.0333` (Active)** | `0.6048`              | `245.3 ms`     |

---

## 5. Live Input/Output Trace Proof

When given the compound biomedical query:

> `"ADAR1 binds to Dicer to cleave pre-miRNA. AIRE is expressed in some skin tumors."` (`Budget = 120 tokens`)

- **Native SQLite `bm25()` (115/120 tokens):** Spends all 5 picks repeating high-IDF `ADAR1 / Dicer / pre-miRNA` chunks (`5702790_sent3`, `5953485_sent2`, `5702790_sent1`, `5953485_sent3`, `946756_sent3`), completely starving the second question (**1/2 sub-queries answered**).
- **`SQLiteContextEngine` (`P4` Submodular, 118/120 tokens):**
  - **Pick #1 & #2:** Retrieves `5953485_sent2` (_"Most importantly, ADAR1 increases the maximum rate (Vmax) of pre-microRNA (miRNA) cleavage..."_) -> **Covers Sub-Query A**.
  - **Submodular Decay:** Decays residual weights of `adar1`, `dicer`, `cleave`, `mirna` by $e^{-2.2 \cdot \phi_w}$.
  - **Pick #3:** Automatically pivots to `12580014_sent0` (_"Keratin-dependent regulation of Aire and gene expression in skin tumor keratinocytes..."_) -> **Covers Sub-Query B (2/2 sub-queries answered)**.

---
