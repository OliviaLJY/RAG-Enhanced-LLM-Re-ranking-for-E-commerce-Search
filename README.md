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
  ├─▶ (2) Query → Attribute Decomposition (LLM) ── planned
  │
  ├─▶ (3) Evidence Retrieval (RAG)              ── planned
  │
  └─▶ (4) Attribute-grounded LLM Re-ranking     ── partial (no-RAG variant)
```

Currently implemented: **stages (1) and a no-RAG variant of (4)**. The
attribute / evidence modules are the research contribution and remain TBD.

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

# Stage 3 (LLM rerank, no RAG) — Requires OPENAI_API_KEY
python llm_rerank.py --num-rerank 50 --model gpt-4o-mini

# Recompute metrics from any saved results file (no LLM calls)
python evaluate.py results/bm25_top20_candidates.json
python evaluate.py results/bm25_top20_candidates.json --use-llm-order --k 10
```

### Project layout

```
.
├── bm25_baseline.py        # Stage 1: BM25 index + Recall/MRR baseline
├── bm25_export_topk.py     # Stage 2: dump top-K candidates per query
├── llm_rerank.py           # Stage 3: RankGPT-style listwise rerank
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

1. **Query → Attribute Decomposition** — LLM emits structured intent
   `{use_case, constraints, important_attributes, soft_preferences}`.
2. **Evidence Retrieval** — per-attribute span retrieval against
   title / description / reviews. A learned retriever (Contriever / ColBERT)
   would be more defensible than keyword match.
3. **Attribute-Grounded Re-ranking** — score each candidate against the
   attribute-evidence bundle, emit `{score, reason}` for explainability.
4. **Ablations** — `-attribute`, `-evidence`, full method, plus a hardened
   baseline (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and significance tests.

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

# 阶段 3：LLM 列表式重排序（无 RAG）
python llm_rerank.py --num-rerank 50 --model gpt-4o-mini

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

#### Module 2：Query → Attribute（创新点 1）
LLM 把 query 解构成结构化意图：

```
Query: "best headphones for gym under $100"
↓
{
  "use_case": "gym",
  "constraints": ["under $100"],
  "important_attributes": ["sweat resistance", "secure fit", "battery life", "durability"],
  "soft_preferences": ["noise cancellation", "lightweight"]
}
```

#### Module 3：Evidence Retrieval（创新点 2）
对每个 attribute，从 item 文本（title / description / reviews）中检索支持
证据，本质是 attribute → evidence 的对齐，而不是简单文本拼接。

```
{
  "sweat resistance": "IPX7 waterproof rating",
  "battery life":      "10 hours continuous playback",
  "secure fit":        "ergonomic ear hook design"
}
```

#### Module 4：Attribute-grounded LLM Rerank
Prompt 结构：

```
Query: ...
Attributes: ...
Candidate Item: ...
Evidence: ...

Task: Score how well the item satisfies the query based on the attributes and evidence.
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
| Week 1 | Day 1–2 BM25 ✅ · Day 3 导出 top-K ✅ · Day 4 LLM rerank ✅ · Day 5 Query→Attribute · Day 6 Evidence Retrieval · Day 7 完整 pipeline |
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
