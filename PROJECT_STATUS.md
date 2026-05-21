# RAG-Enhanced LLM Re-ranking — Project Snapshot

Path: `/Users/lijiayu.102/RAG-Enhanced-LLM-Re-ranking-for-E-commerce-Search`
13 Python files · ~2,500 LOC · MIT license · pinned `requirements.txt`

---

## Project Structure

```
.
├── config.py                       # All constants, paths, RANDOM_SEED, seed_everything()
│
├── bm25_baseline.py                # Stage 1: BM25 index + Recall/MRR baseline
├── bm25_export_topk.py             # Stage 2: top-20 candidates per query → JSON
│
├── query_attributes.py             # Stage 2a: LLM Query → 5-field attribute JSON (Day 5)
├── evidence_retrieval.py           # Stage 2b: v0 evidence (SBERT cosine + keyword) (Day 6)
├── evidence_verify.py              # Stage 2c: v1 LLM verifier on top of v0 (Day 6)
│
├── llm_rerank.py                   # Stage 3: RankGPT listwise, no RAG (baseline)
├── cross_encoder_rerank.py         # Hard baseline: cross-encoder/ms-marco-MiniLM-L-6-v2
├── rerank_listwise_evidence.py     # Stage 4a: listwise + evidence (Day 7)
├── rerank_pointwise_evidence.py    # Stage 4b: pointwise {score, reason} (Day 7)
│
├── evaluate.py                     # Reload-only metrics; --rerank-key flag for any reranker
├── significance.py                 # Paired-bootstrap CI + p on per-query MRR deltas
├── long_tail_analysis.py           # Per-BM25-rank-bucket lift breakdown
│
├── results/
│   ├── bm25_baseline_results.json      # Recall@10=0.699, Recall@100=0.862, MRR@10=0.365
│   └── bm25_top20_candidates.json      # 5.6 MB — 490 queries × 20 candidates
│
├── requirements.txt · .gitignore · .env.example · LICENSE · README.md
```

---

## What's Implemented

### Code — all 13 scripts written

| Stage | File | What it does |
|---|---|---|
| 1 | `bm25_baseline.py` | BM25 over MS MARCO v1.1 5K-slice. Computes Recall@10/100, MRR@10. |
| 2 | `bm25_export_topk.py` | Per-query top-20 candidates with relevance labels. |
| 2a | `query_attributes.py` | LLM Query→Attribute JSON. `--resume` supported. |
| 2b | `evidence_retrieval.py` | v0: SBERT cosine + keyword overlap. No LLM. |
| 2c | `evidence_verify.py` | v1: LLM verifier filters v0 by confidence. |
| 3 | `llm_rerank.py` | RankGPT listwise rerank, no RAG. |
| ref | `cross_encoder_rerank.py` | Off-the-shelf cross-encoder reranker. **Hard baseline.** |
| 4a | `rerank_listwise_evidence.py` | Listwise rerank with per-candidate evidence. |
| 4b | `rerank_pointwise_evidence.py` | Pointwise `{score, reason}` rerank. |
| eval | `evaluate.py` | Reload-only Recall/MRR. Configurable `--rerank-key`. |
| stat | `significance.py` | Paired bootstrap CI + p-value. Paired t-stat. |
| stat | `long_tail_analysis.py` | Lift breakdown by BM25 rank bucket. |

### Numbers actually on disk

Only the BM25 baseline has been executed:

- **BM25**: Recall@10 = 0.699 · Recall@100 = 0.862 · MRR@10 = 0.365 (490 queries)

`long_tail_analysis.py` against the BM25 candidates revealed:

- 107 queries (22%) already at **rank 1** → reranker ceiling
- 110 queries at rank 2-3, 135 at rank 4-10 → the addressable space
- 35 queries at rank 11-20
- 103 queries (21%) **not in top-20** → unrecoverable by any reranker

### Infrastructure

- Bilingual EN/中文 README with pipeline diagram, schemas, ablation grid.
- `RANDOM_SEED=42` seeds Python/NumPy/PYTHONHASHSEED via `config.seed_everything()`.
- Every result JSON logs its effective config.
- `cross_encoder_rerank.py` and the new reranker output schema persist `queries[]`, so `evaluate.py --use-llm-order --rerank-key …` reloads them. *(Note: `llm_rerank.py` has a latent bug — mutates `queries[]` but doesn't persist; reload-evaluation only works for the cross-encoder output as-is.)*

---

## What Still Needs Doing

### Paper-blockers

1. **Run the pipeline end-to-end.** All stages are coded; only BM25 has actual numbers. Needs `OPENAI_API_KEY` + a venv with `requirements.txt` installed (~2 GB: torch + transformers + datasets). ~$0.20 inference budget.
2. *(done)* ~~Cross-encoder hard baseline~~
3. *(done)* ~~Significance tests~~

### Should-haves

4. **Efficiency table** — token cost + latency per query, comparing pointwise vs listwise vs cross-encoder. Needs timing data from #1.
5. **Reasoning-quality eval** — LLM-as-judge on the pointwise `reason` strings. Needs API key.
6. **Evidence retriever v2** — learned retriever (Contriever / ColBERT) trained on `(attribute, evidence)` pairs. Bigger scope.

### Nice-to-haves

7. **E-commerce transfer** — Amazon-ESCI or similar (title promises e-commerce; testbed is MS MARCO).
8. **Review-aware evidence** — mine product reviews for attribute support.

### Polish

9. **CI** — at minimum lint/syntax check on push. None currently.
10. **Tests** — none. Smoke tests for `parse_ranking`, `paired_bootstrap`, `_first_relevant_rank` would be cheap.
11. **Fix `llm_rerank.py`** — persist `queries[]` so it's reload-evaluable like cross-encoder is.
12. **GitHub push auth** — last polish commit is local only.

---

## How to Run Each Step

### Setup (once)

```bash
cd /Users/lijiayu.102/RAG-Enhanced-LLM-Re-ranking-for-E-commerce-Search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # add OPENAI_API_KEY for stages 2a, 2c, 3, 4a, 4b
```

### Pipeline (run in order)

```bash
# Stage 1 — BM25 baseline, writes results/bm25_baseline_results.json
python bm25_baseline.py --sample-limit 5000 --num-eval 500

# Stage 2 — Export BM25 top-20 candidates per query
python bm25_export_topk.py --num-queries 500 --top-k 20

# Stage 2a — Query → Attribute decomposition (LLM, structured JSON)
python query_attributes.py --num-queries 100 --resume

# Stage 2b — Evidence retrieval v0 (SBERT + keyword, no LLM)
python evidence_retrieval.py --num-queries 50 --top-k-candidates 10

# Stage 2c — Evidence verifier v1 (LLM filter on top of v0)
python evidence_verify.py --min-confidence 3

# Stage 3 — LLM rerank (no-RAG baseline)
python llm_rerank.py --num-rerank 50 --model gpt-4o-mini

# Hard baseline — Cross-encoder rerank (no API, local MiniLM model)
python cross_encoder_rerank.py --num-rerank 500

# Stage 4a — Attribute-grounded listwise rerank
python rerank_listwise_evidence.py --num-rerank 50 --top-k 20

# Stage 4b — Attribute-grounded pointwise rerank
python rerank_pointwise_evidence.py --num-rerank 50 --top-k 10
# Use --evidence results/evidence_verified.json on 4a/4b to compare v0 vs v1.
```

### Analysis (reload-only — no API, no GPU)

```bash
# Recompute metrics from any saved results file
python evaluate.py results/bm25_top20_candidates.json
python evaluate.py results/cross_encoder_rerank_results.json \
    --use-llm-order --rerank-key cross_encoder_reranked_doc_ids --k 10

# Paired-bootstrap significance test
python significance.py \
    --baseline results/bm25_top20_candidates.json \
    --method   results/cross_encoder_rerank_results.json \
    --method-key cross_encoder_reranked_doc_ids \
    --output results/sig_ce_vs_bm25.json

# Long-tail analysis (which BM25-rank buckets does the lift come from?)
python long_tail_analysis.py \
    --baseline results/bm25_top20_candidates.json \
    --method   results/cross_encoder_rerank_results.json \
    --method-key cross_encoder_reranked_doc_ids \
    --method-label cross_encoder \
    --output results/long_tail_ce_vs_bm25.json
```

### One-command full ablation (after Stage 2 is done)

```bash
# Once you have BM25 candidates, this produces every comparison row in the grid
python cross_encoder_rerank.py &&                                                  # hard baseline
python llm_rerank.py &&                                                            # no-RAG LLM
python rerank_listwise_evidence.py --evidence results/evidence_retrieval.json &&   # 4a + v0
python rerank_listwise_evidence.py --evidence results/evidence_verified.json &&    # 4a + v1
python rerank_pointwise_evidence.py --evidence results/evidence_retrieval.json &&  # 4b + v0
python rerank_pointwise_evidence.py --evidence results/evidence_verified.json      # 4b + v1
```

### Recommended next action

Get a `venv` + `OPENAI_API_KEY` in place and run the pipeline. Every analysis script (`evaluate`, `significance`, `long_tail_analysis`) is ready to consume the outputs the moment they exist.
