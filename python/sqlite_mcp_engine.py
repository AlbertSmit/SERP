#!/usr/bin/env python3
"""
sqlite_mcp_engine.py - Production SQLiteContextEngine implementing:
  1. Unit-Anchored P4 Base Scorer (chunks_vocab + meta)
  2. Submodular Residual-Decay Token-Budget Packer (pack_context)
  3. Zero-Embedding 1-Hop Lexical Bridge Expansion (expand_bridges)
  4. Entropy-Regulated Temporal Memory Decay (temporal_mode)
Uses the on-disk FTS5 DB (mcp_memory.sqlite) for all vocabulary and meta reads.
Target latency: < 30ms/query (Python overhead over SQLite is acceptable for benchmark).
"""
import sqlite3, math, re, time, os
import numpy as np
from collections import Counter

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'mcp_memory.sqlite')
TOKEN_RE = re.compile(r"\b[a-z0-9]{2,}\b")

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

class SQLiteContextEngine:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.load_stats()

    # ------------------------------------------------------------------
    # Global vocabulary statistics (N_i, T_i) from chunks_vocab
    # ------------------------------------------------------------------
    def load_stats(self):
        # Load N_i and T_i from chunks_vocab into memory dicts
        self.N_i = {}
        self.T_i = {}
        cursor = self.conn.execute("SELECT term, doc, cnt FROM chunks_vocab")
        for term, doc, cnt in cursor.fetchall():
            self.N_i[term] = doc
            self.T_i[term] = cnt
        self.M_ref = self.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        # Load meta stats into NumPy arrays indexed by rowid (fast vectorized lookup)
        meta_cursor = self.conn.execute("SELECT rowid, chunk_id, doc_id, L, U, l_over_u, intrinsic_surprisal, age_days, access_count FROM chunks_meta")
        meta_rows = meta_cursor.fetchall()
        self.M_ref = self.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        # Create NumPy arrays: index by integer rowid
        max_rowid = max(row[0] for row in meta_rows) if meta_rows else 0
        self.meta_L = np.zeros(max_rowid + 1, dtype=np.int32)
        self.meta_U = np.zeros(max_rowid + 1, dtype=np.int32)
        self.meta_l_over_u = np.zeros(max_rowid + 1, dtype=np.float32)
        self.meta_intrinsic = np.zeros(max_rowid + 1, dtype=np.float32)
        self.meta_age_days = np.zeros(max_rowid + 1, dtype=np.float32)
        self.meta_access_count = np.zeros(max_rowid + 1, dtype=np.int32)
        self.doc_stats = {}
        self.chunk_doc_id = {}
        for rowid, chunk_id, doc_id, L, U, lou, intr, age_days, access_count in meta_rows:
            rowid_i = int(rowid)
            self.meta_L[rowid_i] = int(L) if L else 0
            self.meta_U[rowid_i] = max(int(U), 1)
            self.meta_l_over_u[rowid_i] = float(lou)
            self.meta_intrinsic[rowid_i] = float(intr)
            self.meta_age_days[rowid_i] = float(age_days)
            self.meta_access_count[rowid_i] = int(access_count)
            self.doc_stats[chunk_id] = {
                'L': int(L) if L else 0, 'U': max(int(U), 1), 'l_over_u': float(lou),
                'intrinsic': float(intr), 'doc_id': doc_id,
                'age_days': float(age_days), 'access_count': int(access_count)
            }
            self.chunk_doc_id[chunk_id] = doc_id
        # Pre-load all chunk contents and token counts for speed
        self.chunk_content = {}
        self.chunk_counts = {}
        cursor = self.conn.execute("SELECT chunk_id, content FROM chunks_fts")
        for chunk_id, content in cursor.fetchall():
            self.chunk_content[chunk_id] = content
            tokens = tokenize(content)
            self.chunk_counts[chunk_id] = Counter(tokens)
        # Build NumPy sparse inverted index: term -> (rowids_array, counts_array)
        # Uses chunk_counts (cached) to vectorize exact P4 scoring
        self.inverted_rowids = {}
        self.inverted_counts = {}
        # Get all chunk IDs with integer mapping for array indexing
        chunk_ids_all = list(self.chunk_counts.keys())
        # For speed, build arrays per term by iterating chunks once
        term_to_entries = {}
        for cid in chunk_ids_all:
            c_counts = self.chunk_counts[cid]
            chunk_index = chunk_ids_all.index(cid)
            for term, cnt in c_counts.items():
                if term not in term_to_entries:
                    term_to_entries[term] = {'indices': [], 'counts': []}
                term_to_entries[term]['indices'].append(chunk_index)
                term_to_entries[term]['counts'].append(cnt)
        # Convert to NumPy arrays
        for term, data in term_to_entries.items():
            self.inverted_rowids[term] = np.array(data['indices'], dtype=np.int32)
            self.inverted_counts[term] = np.array(data['counts'], dtype=np.int32)
        # Pre-load chunk_id -> doc_id mapping from meta
        self.chunk_doc_id = {}
        for chunk_id, meta in self.doc_stats.items():
            self.chunk_doc_id[chunk_id] = meta['doc_id']

    # ------------------------------------------------------------------
    # Helper: read term frequency stats from chunks_vocab
    # ------------------------------------------------------------------
    def vocab_stats(self, term):
        if term in self.N_i:
            return self.N_i[term], self.T_i[term]
        # Try direct SQL lookup (in case not cached)
        row = self.conn.execute("SELECT doc, cnt FROM chunks_vocab WHERE term = ?", (term,)).fetchone()
        return (row[0], row[1]) if row else (self.M_ref, 1)

    # ------------------------------------------------------------------
    # Helper: get chunk text from chunks_fts by chunk_id
    # ------------------------------------------------------------------
    def get_content(self, chunk_id):
        return self.chunk_content.get(chunk_id, "")

    # ------------------------------------------------------------------
    # Helper: compute token counts for a chunk (cached)
    # ------------------------------------------------------------------
    def chunk_token_counts(self, chunk_id):
        return self.chunk_counts.get(chunk_id, Counter())

    # ------------------------------------------------------------------
    # Unit-Anchored P4 Base Scorer
    # ------------------------------------------------------------------
    def score_chunk_p4(self, chunk_id, query_text, params=None):
        """Compute P4 score for a single chunk against query_text."""
        if params is None:
            params = {
                'beta_0': 1.5, 'gamma_b': 0.35, 'delta_l': 0.35,
                'delta_c': 0.6, 'eta': 1.0, 'alpha_l': 0.35,
                'alpha_u': 0.15, 'lambda_decay': 0.5,
                'L_mean': 160.0, 'L_std': 300.0,
                'U_mean': 280.0, 'conc_mean': 1.5, 'conc_std': 0.8,
                'p_stretch': 0.45
            }
        # Get chunk meta
        meta = self.doc_stats.get(chunk_id, None)
        if meta is None:
            return 0.0
        L = meta['L']
        U = meta['U']
        conc = L / max(U, 1)
        # Compute token overlap with query
        q_tokens = tokenize(query_text)
        q_counts = Counter(q_tokens)
        c_counts = self.chunk_token_counts(chunk_id)
        # Filter query terms to vocab words (2+ chars)
        overlap_words = [w for w in q_counts if w in c_counts or w in self.N_i]
        # For scoring, only consider words that appear in chunk
        c_dict = {w: c_counts[w] for w in overlap_words if c_counts[w] > 0}
        if not c_dict:
            return 0.0

        # Read burst stats for each term from vocab
        L_mean = params.get('L_mean', 160.0)
        U_mean = params.get('U_mean', 280.0)
        M_ref = self.M_ref
        core_vals = []
        info_cover = 0.0
        total_info = 0.0

        for w in c_dict:
            N_i, T_i = self.vocab_stats(w)
            c_raw = c_dict[w]
            q = q_counts.get(w, 1)
            # Burstiness
            mu_i = T_i / max(N_i, 1)
            beta_i = params['beta_0'] * (mu_i ** (-params['gamma_b']))
            # Effective count scaled by L
            vol_scale_L = (L / max(L_mean, 1.0)) ** params['delta_l']
            c_eff = max(0.1, c_raw / max(vol_scale_L, 0.1))
            # Unit-anchored potential (stretched exponential)
            pot = (1.0 - math.exp(-beta_i * (max(c_eff, 0.01) ** params['p_stretch']))) / (1.0 - math.exp(-beta_i + 1e-9))
            # Asymmetric L/U boilerplate suppression
            conc_mean = params.get('conc_mean', 1.5)
            conc_std = params.get('conc_std', 0.8)
            deviation = (conc - conc_mean) / max(conc_std, 1e-3)
            entropy_pen = params['delta_c'] * math.tanh(params['eta'] * max(0.0, deviation))
            suppression = math.exp(-entropy_pen)
            info_w = math.log(max(M_ref, 1) / max(N_i, 1))
            core_vals.append(info_w * pot * suppression * q)
            info_cover += info_w * max(q, 1)
            total_info += info_w * max(q, 1)

        # Aggregate
        sum_pot = sum(core_vals) if core_vals else 0.0
        n_active = len(c_dict)
        sublinear = (1.0 + sum_pot) ** (1.0 - params.get('epsilon', 0.35) / max(n_active, 1))
        diversity_coupling = 1.0 + params.get('lambda_s', 0.5) * math.tanh(0.3 * n_active)
        synergy_coverage = 1.0
        if total_info > 0:
            coverage_fraction = min(1.0, info_cover / total_info)
            synergy_coverage = 1.0 + params.get('lambda_s', 0.5) * math.sqrt(coverage_fraction) * math.tanh(0.5 * n_active)
        score = float(sublinear * diversity_coupling * synergy_coverage * (sum_pot + 0.01))
        return score

    # ------------------------------------------------------------------
    # Submodular Residual-Decay Token-Budget Packer
    # ------------------------------------------------------------------
    def pack_context(self, query_text, budget=450, bridge_candidates=None, bridge_boost_dict=None,
                      lambda_decay=0.5, alpha_l=0.35, alpha_u=0.15, w_bridge=1.0, temporal_mode=False):
        """Greedy submodular packer with residual decay, optional bridge/temporal."""
        # Initialize optional inputs
        if bridge_candidates is None:
            bridge_candidates = []
        if bridge_boost_dict is None:
            bridge_boost_dict = {}
        # Stage 1: get candidates from FTS5
        stop_threshold = int(0.15 * self.M_ref)
        tokens = [w for w in tokenize(query_text) if len(w) >= 2 and (w not in self.N_i or self.N_i[w] <= stop_threshold)]
        if not tokens:
            # Even if no direct query tokens, include bridge candidates if any
            candidates = list(dict.fromkeys(bridge_candidates))
        else:
            match_expr = " OR ".join(f'"{w}"' for w in tokens if len(w) >= 2)
            cursor = self.conn.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 100",
                (match_expr,)
            )
            candidates = list(dict.fromkeys([r[0] for r in cursor.fetchall() if not r[0].endswith("_full")]))
        # Add bridge chunks to candidate pool
        for cid in bridge_candidates:
            if not cid.endswith("_full") and cid not in candidates:
                candidates.append(cid)
        # Residual importance weights (exact)
        q_counts = Counter(tokenize(query_text))
        r_weights = {}
        for w in q_counts:
            N_i, _ = self.vocab_stats(w)
            M_ref = self.M_ref
            r_weights[w] = math.log(1 + (M_ref - N_i + 0.5) / (N_i + 0.5)) * max(q_counts[w], 1)
        # Pre-compute chunk meta and counts
        chunk_scores = {}
        chunk_L = {}
        chunk_U = {}
        chunk_counts_local = {}
        for cid in candidates:
            meta = self.doc_stats.get(cid)
            if meta:
                chunk_L[cid] = meta['L']
                chunk_U[cid] = meta['U']
            else:
                chunk_L[cid] = 0
                chunk_U[cid] = 1
            chunk_counts_local[cid] = self.chunk_counts.get(cid, Counter())
        # Score all candidates using P4 (with alpha_l mapped to delta_c)
        p4_params = {
            'beta_0': 1.5, 'gamma_b': 0.35, 'delta_l': 0.35,
            'delta_c': alpha_l, 'eta': 1.0, 'alpha_l': alpha_l,
            'alpha_u': alpha_u, 'lambda_decay': lambda_decay,
            'L_mean': 160.0, 'L_std': 300.0,
            'U_mean': 280.0, 'conc_mean': 1.5, 'conc_std': 0.8,
            'p_stretch': 0.45
        }
        for cid in candidates:
            meta = self.doc_stats.get(cid)
            if meta:
                chunk_scores[cid] = self.score_chunk_p4(cid, query_text, params=p4_params)
            else:
                chunk_scores[cid] = 0.01
        # Greedy selection
        selected = []
        total_L = 0
        covered_docs = set()
        max_iter = 15
        init_r_sum = max(sum(r_weights.values()), 1e-6)
        used_bridge_docs = set()
        while total_L < budget and len(selected) < max_iter:
            best_cid = None
            best_marginal = -1.0
            for cid in candidates:
                if cid in selected:
                    continue
                L = chunk_L.get(cid, 0)
                if L == 0:
                    continue
                if total_L + L > budget:
                    # Skip without removing, so we can try smaller chunks first? Actually skip for now
                    continue
                c_counts = chunk_counts_local.get(cid, Counter())
                overlap_words = [w for w in r_weights if w in c_counts and c_counts[w] > 0]
                marginal_sum = 0.0
                for w in overlap_words:
                    N_i, T_i = self.vocab_stats(w)
                    c_raw = c_counts[w]
                    mu_i = T_i / max(N_i, 1)
                    beta_i = 1.5 * (mu_i ** (-0.35))
                    pot_i = (1.0 - math.exp(-beta_i * (max(c_raw, 0.01) ** 0.45))) / (1.0 - math.exp(-beta_i + 1e-9))
                    marginal_sum += r_weights[w] * pot_i
                U_val = chunk_U.get(cid, 1)
                cost_eff = max((L + 10) ** 0.35 * (L / max(U_val, 1)) ** 0.15, 0.1)
                parent_doc = self.chunk_doc_id.get(cid, cid.split('_')[0])
                synergy = 1.15 if parent_doc not in covered_docs else 0.85
                base_score = chunk_scores.get(cid, 0.0)
                rem_ratio = sum(r_weights.values()) / init_r_sum
                # Gate bridge boost: 0.0 on Pick #1-2; activates for final slots when primary demand decays
                bridge_gate = 1.0 if len(selected) >= 1 else 0.0
                bridge_boost_val = bridge_boost_dict.get(cid, 0.0) * bridge_gate * 0.35
                temporal_factor = 1.0
                lou_suppression = 1.0
                if temporal_mode:
                    meta_local = self.doc_stats.get(cid)
                    if meta_local:
                        lou = meta_local['l_over_u']
                        lou_suppression = math.exp(-alpha_u * max(0.0, lou - 2.2))
                marginal_gain = ((marginal_sum + (base_score * 0.05 * rem_ratio) + w_bridge * bridge_boost_val) * synergy * temporal_factor * lou_suppression) / max(cost_eff, 0.1)
                if marginal_gain > best_marginal:
                    best_marginal = marginal_gain
                    best_cid = cid
            if best_cid is None or best_marginal <= 0:
                # Fallback: pick remaining budget-fitting chunk with best raw score
                for cid in candidates:
                    if cid in selected:
                        continue
                    L = chunk_L.get(cid, 0)
                    if total_L + L > budget or L == 0:
                        continue
                    parent_doc = self.chunk_doc_id.get(cid, cid.split('_')[0])
                    base_score = chunk_scores.get(cid, 0.01)
                    coverage_synergy = 1.0 + 0.3 * (1.0 if parent_doc not in covered_docs else 0.0)
                    U_val = chunk_U.get(cid, 1)
                    cost_eff = max((L + 10) ** 0.35 * (L / max(U_val, 1)) ** 0.15, 0.1)
                    bridge_boost_val = bridge_boost_dict.get(cid, 0.0)
                    temporal_factor_fb = 1.0
                    lou_suppression_fb = 1.0
                    if temporal_mode:
                        meta_local_fb = self.doc_stats.get(cid)
                        if meta_local_fb:
                            temporal_factor_fb = self.temporal_factor(cid)
                            lou_fb = meta_local_fb['l_over_u']
                            lou_suppression_fb = math.exp(-alpha_u * max(0.0, lou_fb - 1.5))
                    marginal_gain_fb = (base_score * coverage_synergy + w_bridge * bridge_boost_val) * temporal_factor_fb * lou_suppression_fb / max(cost_eff, 0.1)
                    if marginal_gain_fb > best_marginal:
                        best_marginal = marginal_gain_fb
                        best_cid = cid
            if best_cid is None:
                break
            meta = self.doc_stats.get(best_cid)
            if meta is None:
                break
            L = meta['L']
            if total_L + L > budget:
                # Can't fit this chunk; skip it permanently for this selection
                if best_cid in candidates:
                    candidates.remove(best_cid)
                continue
            selected.append(best_cid)
            total_L += L
            # Decay residual weights using parameterized lambda_decay
            c_counts = chunk_counts_local.get(best_cid, Counter())
            for w in r_weights:
                if w in c_counts and c_counts[w] > 0:
                    N_i, T_i = self.vocab_stats(w)
                    mu_i = T_i / max(N_i, 1)
                    beta_i = 1.5 * (mu_i ** (-0.35))
                    pot_i = (1.0 - math.exp(-beta_i * (max(c_counts[w], 0.01) ** 0.45))) / (1.0 - math.exp(-beta_i + 1e-9))
                    r_weights[w] *= math.exp(-2.2 * pot_i)
            parent_doc_best = self.chunk_doc_id.get(best_cid, best_cid.split('_')[0])
            covered_docs.add(parent_doc_best)
            if bridge_boost_dict.get(best_cid, 0.0) > 0: used_bridge_docs.add(parent_doc_best)
            if best_cid in candidates:
                candidates.remove(best_cid)
        return selected

    # ------------------------------------------------------------------
    # Zero-Embedding 1-Hop Lexical Bridge Expansion
    # ------------------------------------------------------------------
    def expand_bridges(self, query_text, seed_chunks, budget=450):
        """From top-3 seed chunks, extract bridge tokens and find 1-hop evidence chunks."""
        if not seed_chunks:
            return []
        # Take top 3 seeds by score
        seeds = seed_chunks[:3]
        # Collect non-query terms from seeds
        q_words = set(tokenize(query_text))
        bridge_candidates = Counter()
        for cid in seeds:
            c_counts = self.chunk_token_counts(cid)
            for w, c in c_counts.items():
                if w not in q_words and len(w) >= 2:
                    # Compute bridge salience (weighted lower: 0.22 * Psi)
                    N_i, T_i = self.vocab_stats(w)
                    M_ref = self.M_ref
                    seed_hits = sum(1 for s in seeds if w in self.chunk_token_counts(s))
                    psi = (seed_hits ** 1.5) * math.log(max(M_ref, 1) / max(N_i, 1)) * ((max(T_i, 1) / max(N_i, 1)) ** (-0.4))
                    # Budget-gated: weight strictly lower than primary terms
                    bridge_candidates[w] = 0.22 * psi
        # Select top 5 bridge tokens
        top_bridges = [w for w, _ in bridge_candidates.most_common(5)]
        # Compute psi for top bridge terms (exact formula as spec)
        psi_dict = {}
        for b in top_bridges:
            seed_hits = sum(1 for s in seeds if b in self.chunk_token_counts(s))
            N_i, T_i = self.vocab_stats(b)
            M_ref_local = self.M_ref
            psi = (seed_hits ** 1.5) * math.log(max(M_ref_local, 1) / max(N_i, 1)) * ((max(T_i, 1) / max(N_i, 1)) ** (-0.4))
            psi_dict[b] = psi
        # Find evidence chunks via FTS5 MATCH on top bridge tokens
        bridge_evidence = []
        for b in top_bridges:
            cursor = self.conn.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 20",
                (f'"{b}"',)
            )
            for (cid,) in cursor.fetchall():
                if cid not in bridge_evidence and cid not in seeds:
                    bridge_evidence.append(cid)
        return bridge_evidence, psi_dict

    # ------------------------------------------------------------------
    # Entropy-Regulated Temporal Memory Decay
    # ------------------------------------------------------------------
    def temporal_factor(self, chunk_id):
        meta = self.doc_stats.get(chunk_id)
        if not meta:
            return 1.0
        intrinsic = meta['intrinsic']
        U = max(meta['U'], 1)
        L = meta['L']
        access_count = meta['access_count']
        decay_rate = 0.003 / (max(intrinsic, 1e-6) * max(U / L, 0.01) * math.log(1 + access_count))
        time_factor = math.exp(-decay_rate * meta['age_days'])
        return time_factor

    def apply_temporal_decay(self, chunk_ids):
        """Adjust chunk scores by temporal factor."""
        adjusted = {}
        for cid in chunk_ids:
            meta = self.doc_stats.get(cid)
            if not meta:
                adjusted[cid] = 0.0
                continue
            # Base score from meta or quick approximation
            base_score = self.score_chunk_p4(cid, "")
            tf = self.temporal_factor(cid)
            adjusted[cid] = base_score * tf
        return adjusted

    # ------------------------------------------------------------------
    # Full pipeline (for benchmark comparison)
    # ------------------------------------------------------------------
    def full_pipeline(self, query_text, budget=450, use_bridges=True, temporal_mode=False,
                       lambda_decay=0.5, alpha_l=0.35, alpha_u=0.15, w_bridge=1.0):
        # Step 1: base packer (without bridges) to get seed chunks
        base_selected = self.pack_context(query_text, budget=budget,
                                           lambda_decay=lambda_decay, alpha_l=alpha_l,
                                           alpha_u=alpha_u, w_bridge=0.0, temporal_mode=False)
        # Compute initial residual demand for gating bridge activation
        q_counts = Counter(tokenize(query_text))
        initial_residual = sum(
            math.log(1 + (self.M_ref - self.vocab_stats(w)[0] + 0.5) / (self.vocab_stats(w)[0] + 0.5)) * max(q_counts[w], 1)
            for w in q_counts if len(w) >= 2
        )
        # Step 2: bridge expansion (budget-gated, active when primary coverage > 30%)
        bridge_evidence = []
        bridge_boost_dict = {}
        if use_bridges and len(base_selected) >= 1 and initial_residual > 0:
            selected_counts = Counter()
            for c in base_selected:
                selected_counts.update(self.chunk_token_counts(c))
            covered_terms = sum(1 for w in q_counts if selected_counts[w] > 0)
            primary_coverage = covered_terms / max(len(q_counts), 1)
            # Gate: expand bridges when primary coverage > 30% (so Method 4 activates on real queries)
            if primary_coverage > 0.30:
                seed_chunks = base_selected[:3] if base_selected else []
                bridge_evidence, psi_dict = self.expand_bridges(query_text, seed_chunks, budget)
                # Build boost dict for each bridge chunk
                for cid in bridge_evidence:
                    meta_local = self.doc_stats.get(cid)
                    if meta_local:
                        c_counts_local = self.chunk_token_counts(cid)
                        max_psi = 0.0
                        for b in psi_dict:
                            if b in c_counts_local and c_counts_local[b] > 0:
                                max_psi = max(max_psi, psi_dict[b])
                        bridge_boost_dict[cid] = max_psi
        # Step 3: run full submodular packer with bridge candidates + temporal mode
        selected = self.pack_context(query_text, budget=budget,
                                     bridge_candidates=bridge_evidence,
                                     bridge_boost_dict=bridge_boost_dict,
                                     lambda_decay=lambda_decay,
                                     alpha_l=alpha_l,
                                     alpha_u=alpha_u,
                                     w_bridge=w_bridge,
                                     temporal_mode=temporal_mode)
        # Always enforce budget cap
        total_L = sum(self.doc_stats.get(c, {}).get('L', 0) for c in selected)
        while total_L > budget and len(selected) > 0:
            selected = selected[:-1]
            total_L = sum(self.doc_stats.get(c, {}).get('L', 0) for c in selected)
        # Cap at 15 chunks
        selected = selected[:15]
        return selected

    # ------------------------------------------------------------------
    # Benchmark evaluation helpers
    # ------------------------------------------------------------------
    def evaluate_budget_recall(self, query_text, qrels_set):
        selected = self.pack_context(query_text, budget=450)
        covered_docs = set()
        for cid in selected:
            parent_doc = self.chunk_doc_id.get(cid, cid.split('_')[0])
            covered_docs.add(parent_doc)
        covered_rels = covered_docs.intersection(qrels_set)
        return len(covered_rels) / max(len(qrels_set), 1)

if __name__ == '__main__':
    engine = SQLiteContextEngine()
    print(f"SQLiteContextEngine loaded. DB: {DB_PATH}")
    print(f"Vocabulary terms cached: {len(engine.N_i)}")
    # Quick sanity query
    selected = engine.pack_context("myeloid-derived suppressor cells", budget=450)
    print(f"Sample selected chunks ({len(selected)}): {selected[:5]}")
    total_L = sum(engine.doc_stats.get(c, {}).get('L', 0) for c in selected)
    print(f"Sample total L: {total_L}")
