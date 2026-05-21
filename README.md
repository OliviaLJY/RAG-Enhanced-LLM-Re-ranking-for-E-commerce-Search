# RAG-Enhanced LLM Re-ranking for E-commerce Search

> **Attribute-Grounded Retrieval-Augmented LLM Re-ranking** — a research scaffold
> exploring whether structured-attribute decomposition + per-attribute evidence
> retrieval improves LLM reranking over BM25 on MS MARCO, as a stepping stone
> toward e-commerce search.

[English](#english) · [中文](#中文)

---

## English

### Pipeline

```
Query
  │
  ├─▶ (1) Candidate Retrieval (BM25)            ── implemented
  │
  ├─▶ (2) Query → Attribute Decomposition (LLM) ── implemented
  │
  ├─▶ (3) Evidence Retrieval (RAG)              ── implemented (v0 + v1)
  │
  └─▶ (4) Attribute-grounded LLM Re-ranking     ── implemented (listwise + pointwise)
```

All four pipeline stages are now end-to-end runnable. The remaining work is
**experiments**: ablations and the e-commerce-dataset transfer. The hard
cross-encoder baseline (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is implemented
in `cross_encoder_rerank.py`, and a paired-bootstrap significance test is in
`significance.py`.

### Results (MS MARCO v1.1, 5K-query slice)

BM25 baseline on 490 evaluable queries:

| Metric      | BM25   |
|-------------|--------|
| Recall@10   | 0.6986 |
| Recall@100  | 0.8622 |
| MRR@10      | 0.3654 |

LLM rerank (`gpt-4o-mini`, listwise, BM25 top-20 → reorder) is logged into
`results/llm_rerank_results.json` when you run stage 4.

### Setup

```bash
git clone https://github.com/OliviaLJY/RAG-Enhanced-LLM-Re-ranking-for-E-commerce-Search.git
cd RAG-Enhanced-LLM-Re-ranking-for-E-commerce-Search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # add your OPENAI_API_KEY for stage 4
```

### Usage

Each stage is a standalone script with `argparse`. Defaults live in `config.py`.

```bash
# Stage 1 — BM25 baseline, writes results/bm25_baseline_results.json
python bm25_baseline.py --sample-limit 5000 --num-eval 500

# Stage 2 — Export BM25 top-20 candidates per query
python bm25_export_topk.py --num-queries 500 --top-k 20

# Stage 2a — Query → Attribute decomposition (LLM, structured JSON)
python query_attributes.py --num-queries 100 --resume

# Stage 2b — Evidence retrieval v0 (lexical + SBERT cosine, no LLM)
python evidence_retrieval.py --num-queries 50 --top-k-candidates 10

# Stage 2c — Evidence verifier v1 (LLM filter on top of v0); optional
python evidence_verify.py --min-confidence 3

# Stage 3 — LLM rerank (no-RAG baseline) — Requires OPENAI_API_KEY
python llm_rerank.py --num-rerank 50 --model gpt-4o-mini

# Hard baseline — Cross-encoder rerank (no API, local MiniLM model)
python cross_encoder_rerank.py --num-rerank 500

# Stage 4a — Attribute-grounded listwise rerank (RankGPT-style + evidence)
python rerank_listwise_evidence.py --num-rerank 50 --top-k 20

# Stage 4b — Attribute-grounded pointwise rerank (per-candidate score + reason)
python rerank_pointwise_evidence.py --num-rerank 50 --top-k 10

# Use --evidence results/evidence_verified.json on 4a/4b to compare v0 vs v1.

# Recompute metrics from any saved results file (no LLM calls)
python evaluate.py results/bm25_top20_candidates.json
python evaluate.py results/bm25_top20_candidates.json --use-llm-order --k 10
python evaluate.py results/cross_encoder_rerank_results.json \
    --use-llm-order --rerank-key cross_encoder_reranked_doc_ids --k 10

# Paired-bootstrap significance test on per-query MRR deltas (numpy-only)
python significance.py \
    --baseline results/bm25_top20_candidates.json \
    --method   results/cross_encoder_rerank_results.json \
    --method-key cross_encoder_reranked_doc_ids \
    --output results/sig_ce_vs_bm25.json

# Long-tail analysis: which BM25-rank buckets does the lift come from?
python long_tail_analysis.py \
    --baseline results/bm25_top20_candidates.json \
    --method   results/cross_encoder_rerank_results.json \
    --method-key cross_encoder_reranked_doc_ids \
    --method-label cross_encoder \
    --output results/long_tail_ce_vs_bm25.json
```

### Project layout

```
.
├── bm25_baseline.py             # Stage 1: BM25 index + Recall/MRR baseline
├── bm25_export_topk.py          # Stage 2: dump top-K candidates per query
├── query_attributes.py          # Stage 2a: Query → structured attribute JSON (Day 5)
├── evidence_retrieval.py        # Stage 2b: per-attribute evidence, v0 (Day 6)
├── evidence_verify.py           # Stage 2c: LLM verifier, v1 (Day 6)
├── llm_rerank.py                # Stage 3: RankGPT-style listwise rerank (no-RAG)
├── cross_encoder_rerank.py      # Hard baseline: cross-encoder/ms-marco-MiniLM-L-6-v2
├── significance.py              # Paired-bootstrap p-value + 95% CI on MRR deltas
├── long_tail_analysis.py        # Per-BM25-rank-bucket lift breakdown
├── rerank_listwise_evidence.py  # Stage 4a: attribute-grounded listwise (Day 7)
├── rerank_pointwise_evidence.py # Stage 4b: attribute-grounded pointwise + reason (Day 7)
├── evaluate.py             # Recompute metrics from saved JSON
├── config.py               # Shared constants + seed_everything()
├── requirements.txt        # Pinned dependencies
├── results/                # JSON outputs (large dumps are gitignored)
└── README.md
```

### Reproducibility

- A single `RANDOM_SEED` (default `42`) seeds Python, NumPy, and
  `PYTHONHASHSEED` via `config.seed_everything()`.
- Every script logs its full effective config into the output JSON.
- MS MARCO is loaded via `datasets.load_dataset(...)` and sliced
  deterministically (`dataset.select(range(limit))`), so two runs with the same
  `--sample-limit` see the same corpus.
- LLM calls use `temperature=0` and `max_tokens=300`; outputs are still subject
  to provider-side non-determinism.

### Roadmap (research contribution)

The current code is a baseline; the research delta lives in modules **not yet
written**:

1. ~~**Query → Attribute Decomposition** — LLM emits structured intent~~
   ✅ Implemented in `query_attributes.py` (Day 5). Schema is generalized
   beyond e-commerce (see [Schema](#query--attribute-schema-day-5) below) so
   MS MARCO's factual / how-to queries also fit.
2. ~~**Evidence Retrieval** — per-attribute span retrieval against
   title / description / reviews. A learned retriever (Contriever / ColBERT)
   would be more defensible than keyword match.~~
   ✅ Implemented as a two-stage ablation chain (Day 6):
   - **v0** (`evidence_retrieval.py`) — lexical keyword overlap + SBERT
     (`all-MiniLM-L6-v2`) cosine similarity, weighted by `--alpha`. No LLM
     calls; serves as the deliberately-simple lower bound.
   - **v1** (`evidence_verify.py`) — LLM verifier layered on v0. Per
     `(query, candidate)` it asks `gpt-4o-mini` "does this evidence support
     this attribute?", filters by `--min-confidence` (1-5). Decoupling lets
     you tune v0 freely without re-paying for LLM calls, and the v1 delta is
     measurable as a clean ablation.
   - A learned retriever (Contriever / ColBERT) is still the natural v2 if v1
     proves a viable signal.
3. ~~**Attribute-Grounded Re-ranking** — score each candidate against the
   attribute-evidence bundle, emit `{score, reason}` for explainability.~~
   ✅ Implemented as two complementary modes (Day 7) so each can be ablated:
   - **Listwise** (`rerank_listwise_evidence.py`) — RankGPT-style; one LLM
     call per query, all candidates plus their evidence in the prompt.
     Cheapest, directly comparable to the no-RAG `llm_rerank.py` baseline.
   - **Pointwise** (`rerank_pointwise_evidence.py`) — one LLM call per
     `(query, candidate)` returning `{"score": 1-5, "reason": "..."}`.
     Sorted by score with BM25 rank as tiebreak. The `reason` field is the
     paper's explainability artifact and supports per-attribute weighting in
     follow-up work.
   - Both consume the v0 *or* v1 evidence JSON via `--evidence`, so the
     `evidence_quality × rerank_mode` ablation is a 2×2 grid for free.
4. **Ablations** — `-attribute`, `-evidence` (v0 vs. v1), listwise vs.
   pointwise, full method. The hard cross-encoder baseline
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is in `cross_encoder_rerank.py`;
   paired-bootstrap significance testing is in `significance.py`.

### Query → Attribute schema (Day 5)

Each query is decomposed into a strict JSON object:

```json
{
  "intent_type":          "factual | product_search | how_to | comparison | navigational | other",
  "core_concepts":        ["main subject/entity nouns"],
  "constraints":          ["hard filters: price, time, location, audience"],
  "important_attributes": ["attributes that drive relevance"],
  "soft_preferences":     ["nice-to-haves"]
}
```

Example — product search:

```json
{
  "intent_type": "product_search",
  "core_concepts": ["headphones"],
  "constraints": ["under $100", "for gym use"],
  "important_attributes": ["sweat resistance", "secure fit", "battery life"],
  "soft_preferences": ["noise cancellation", "lightweight"]
}
```

Example — factual MS MARCO query:

```json
{
  "intent_type": "factual",
  "core_concepts": ["capital", "France"],
  "constraints": [],
  "important_attributes": [],
  "soft_preferences": []
}
```

Generation uses `response_format={"type": "json_object"}` and a normalizer
(`query_attributes.normalize_attributes`) that fills missing fields with safe
defaults so downstream stages can rely on the shape.

### Evidence retrieval schema (Day 6)

`evidence_retrieval.py` (v0) emits one evidence map per `(query, candidate)`:

```json
{
  "query_id": 0,
  "query": "best headphones for gym",
  "attributes": { ... Day 5 schema ... },
  "candidates": [
    {
      "doc_id": 1234,
      "rank": 1,
      "is_relevant": true,
      "evidence": {
        "sweat resistance": {
          "field": "important_attributes",
          "evidence": "IPX7 rating protects against sweat and rain.",
          "score":   0.81,
          "cosine":  0.74,
          "keyword": 0.66
        },
        "battery life": { ... }
      }
    }
  ]
}
```

`score = alpha * cosine + (1 - alpha) * keyword`. Default `alpha=0.6`.

`evidence_verify.py` (v1) preserves the same shape but each evidence entry
gains `verifier_confidence: 1-5`, and entries where the LLM said
`supports=false` or `confidence < --min-confidence` are dropped. The raw
per-attribute verdicts are kept under `candidates[i]._verifier_raw` for audit.

### Reranker output schema (Day 7)

Both rerankers emit the same evaluatable shape so `evaluate.py --use-llm-order`
works on either file:

```json
{
  "method": "listwise_with_evidence" | "pointwise_with_evidence",
  "evidence_source": "results/evidence_retrieval.json",
  "metrics": {
    "bm25_mrr10": 0.36,
    "llm_rerank_mrr10": 0.42,
    "delta_mrr10": 0.06,
    "relative_gain_pct": 16.6,
    "...": "..."
  },
  "queries": [
    {
      "query_id": 0,
      "query": "...",
      "relevant_doc_ids": [...],
      "candidates":           [{"rank": 1, "doc_id": ..., "is_relevant": ...}],
      "llm_reranked_doc_ids": [...],
      "rerank_success": true,
      "scored_candidates":    [               // pointwise only
        {"doc_id": ..., "score": 5, "reason": "...", "evidence_attrs": [...]}
      ]
    }
  ]
}
```

### Ablation grid (4 runs, all from the same upstream artifacts)

| Evidence source                  | Listwise (4a) | Pointwise (4b) |
|----------------------------------|---------------|----------------|
| `evidence_retrieval.json` (v0)   | run a         | run b          |
| `evidence_verified.json` (v1)    | run c         | run d          |

Plus the no-RAG `llm_rerank.py` (run 0), BM25 (run -), and the cross-encoder
hard baseline `cross_encoder_rerank.py` (run +) as references. The cross-encoder
is the off-the-shelf benchmark that the attribute-grounded methods must beat to
claim a real win — not just a lift over BM25.

See [中文](#中文) for the original day-by-day plan.

### License

[MIT](LICENSE)

---

## 中文

### 目标

> **基于结构化属性的检索增强 LLM 重排序** —— 一个研究脚手架。先用 MS MARCO
> 验证“属性分解 + 证据检索”是否能改进 LLM 重排序，再迁移到电商搜索场景。

### 流水线

```
Query
  │
  ├─▶ (1) 候选检索 BM25                  ── 已实现
  │
  ├─▶ (2) Query → 属性结构化 (LLM)       ── 计划中
  │
  ├─▶ (3) 证据检索 RAG                   ── 计划中
  │
  └─▶ (4) 属性证据驱动的 LLM 重排序      ── 部分实现（无 RAG 版本）
```

### 现有结果（MS MARCO v1.1，5000 query 切片）

| 指标        | BM25   |
|-------------|--------|
| Recall@10   | 0.6986 |
| Recall@100  | 0.8622 |
| MRR@10      | 0.3654 |

### 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # 填入 OPENAI_API_KEY，用于第 3 阶段
```

### 各阶段脚本

```bash
# 阶段 1：BM25 基线
python bm25_baseline.py --sample-limit 5000 --num-eval 500

# 阶段 2：导出每条 query 的 BM25 top-20 候选
python bm25_export_topk.py --num-queries 500 --top-k 20

# 阶段 2a：Query → 结构化属性分解（Day 5）
python query_attributes.py --num-queries 100 --resume

# 阶段 2b：证据检索 v0（关键词 + SBERT 余弦，无 LLM）
python evidence_retrieval.py --num-queries 50 --top-k-candidates 10

# 阶段 2c：证据验证 v1（LLM 验证器叠加 v0），可选
python evidence_verify.py --min-confidence 3

# 阶段 3：LLM 列表式重排序（无 RAG 基线）
python llm_rerank.py --num-rerank 50 --model gpt-4o-mini

# 阶段 4a：属性证据驱动的列表式重排
python rerank_listwise_evidence.py --num-rerank 50 --top-k 20

# 阶段 4b：属性证据驱动的点式重排（含 reason，可解释）
python rerank_pointwise_evidence.py --num-rerank 50 --top-k 10

# 把 4a/4b 的 --evidence 切到 results/evidence_verified.json 即可对比 v0 / v1。

# 离线复算指标（不再调用 LLM）
python evaluate.py results/bm25_top20_candidates.json
```

### 论文方向 / 命名建议

候选标题：

1. **Attribute-Grounded Retrieval-Augmented LLM Re-ranking for E-commerce Search**
2. **From Query to Attributes: Evidence-Grounded LLM Re-ranking with Retrieval-Augmented Intent Decomposition**
3. **Efficient Evidence-Augmented LLM Re-ranking for E-commerce Search**

### 方法模块（写论文时三段式）

#### Module 1：候选检索 BM25 ✅
已完成，BM25 top-100。

#### Module 2：Query → Attribute（创新点 1）✅
LLM 把 query 解构成结构化意图（已实现，`query_attributes.py`）。为了同时覆盖
MS MARCO 的事实型 query 与电商型 query，统一为以下 schema：

```json
{
  "intent_type":          "factual | product_search | how_to | comparison | navigational | other",
  "core_concepts":        ["主要主体/实体"],
  "constraints":          ["硬约束：价格、时间、地点、人群"],
  "important_attributes": ["决定相关性的属性"],
  "soft_preferences":     ["加分项"]
}
```

电商示例：

```json
{
  "intent_type": "product_search",
  "core_concepts": ["headphones"],
  "constraints": ["under $100", "for gym use"],
  "important_attributes": ["sweat resistance", "secure fit", "battery life"],
  "soft_preferences": ["noise cancellation", "lightweight"]
}
```

事实型 query 示例：

```json
{
  "intent_type": "factual",
  "core_concepts": ["capital", "France"],
  "constraints": [],
  "important_attributes": [],
  "soft_preferences": []
}
```

#### Module 3：Evidence Retrieval（创新点 2）✅
两阶段消融链（Day 6）：

- **v0** (`evidence_retrieval.py`) — 关键词重合 + SBERT (`all-MiniLM-L6-v2`)
  余弦相似度，按 `--alpha` 加权（默认 0.6）。**不调用 LLM**，是有意设计的简单下界。
- **v1** (`evidence_verify.py`) — 在 v0 之上叠加 LLM 验证器。按 (query, 候选)
  批量请求 `gpt-4o-mini` 判断「该证据是否真正支持该属性」，按
  `--min-confidence` (1–5) 过滤。两段拆开，v0 可自由调参，v1 增量可独立度量。

后续若 v1 验证为有效信号，自然 v2 是训练小型 evidence retriever（Contriever /
ColBERT），训练数据可由 LLM 生成 `(attribute, evidence)` 对。

证据 JSON 结构示例：

```json
{
  "sweat resistance": {
    "field":   "important_attributes",
    "evidence": "IPX7 rating protects against sweat and rain.",
    "score":   0.81,
    "cosine":  0.74,
    "keyword": 0.66
  }
}
```

`score = alpha * cosine + (1 - alpha) * keyword`。v1 额外加 `verifier_confidence: 1-5`，
被验证器判为不支持或低置信度的证据会被剔除。

#### Module 4：Attribute-grounded LLM Rerank ✅
两种互补模式（Day 7），各自独立 ablation：

- **Listwise** (`rerank_listwise_evidence.py`) — RankGPT 风格，一次 LLM 调用
  喂入全部候选 + 各自的属性证据。最便宜，与无 RAG 的 `llm_rerank.py` 直接对比。
- **Pointwise** (`rerank_pointwise_evidence.py`) — 每条 (query, candidate) 调用一次
  LLM，返回 `{"score": 1-5, "reason": "..."}`。按分数排序，BM25 原始名次作为
  并列时的次序。`reason` 字段是论文的可解释性叙事来源。
- 两者都通过 `--evidence` 接 v0 或 v1 的证据 JSON，所以
  `证据质量 × 重排模式` 直接是 2×2 消融网格。

Prompt 结构（pointwise）：

```
Search query: ...
Query intent: ...
Important attributes: ...
Constraints: ...

Candidate passage:
...

Attribute evidence found in passage:
- attribute_a: "..."
- attribute_b: "..."

Score 1-5 how well the passage satisfies the query.
Return JSON: {"score": <int 1-5>, "reason": "<one sentence>"}.
```

### 实验设计

**Baselines**

- BM25
- BM25 + LLM rerank（无 RAG）← 已实现
- BM25 + Attribute only
- BM25 + Evidence only
- Ours（完整方法）
- Hard baseline：`cross-encoder/ms-marco-MiniLM-L-6-v2`

**指标**：MRR@10, NDCG@10, Recall@10；附加 long-tail 提升、token 成本、latency。

### 三周执行计划

| 周次 | 重点 |
|------|------|
| Week 1 | Day 1–2 BM25 ✅ · Day 3 导出 top-K ✅ · Day 4 LLM rerank ✅ · Day 5 Query→Attribute ✅ · Day 6 Evidence Retrieval v0+v1 ✅ · Day 7 完整 pipeline ✅ |
| Week 2 | Day 8–9 full eval · Day 10 ablation · Day 11 long-tail · Day 12 efficiency |
| Week 3 | Day 13–14 Method/Experiment · Day 15 Intro · Day 16 Abstract |

### DS 建议（顶会方向）

**优先级最高 — Evidence Retrieval 不能 trivial**：
- 训练小型 evidence retriever（Contriever / ColBERT），训练数据可由 LLM 生成
  `(attribute, evidence)` 对。
- 多视角融合：从用户评论中挖掘属性证据（review-aware attribute grounding，
  创新点更明确）。
- 证据质量打分：一个轻量分类器判断 evidence 与 attribute 的相关性，过滤噪声。

**次高 — 证明 LLM 不可替代**：
- Human eval / GPT-4-as-judge 对比 attribute 分解的合理性。
- 在属性 trade-off（“便宜但不轻 vs 贵但超轻”）上对比 embedding 相似度。

**第三 — 硬核 baseline**：
- 强烈建议加 `cross-encoder/ms-marco-MiniLM-L-6-v2`，若你的方法
  NDCG@10 提升 < 5%，顶会很难过。需做显著性检验并报告 effect size。

**第四 — 可复现性**：
- 开源 query→attribute prompt、evidence retrieval 代码、评估脚本，加分项。

### 许可

[MIT](LICENSE)
