"""
Shared configuration for the BM25 → LLM rerank pipeline.
共享配置：BM25 → LLM 重排序流水线

All hardcoded constants live here so scripts stay short and reproducible.
所有硬编码常量集中在这里，方便复现实验。
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"

BM25_BASELINE_RESULTS = RESULTS_DIR / "bm25_baseline_results.json"
BM25_CANDIDATES = RESULTS_DIR / "bm25_top20_candidates.json"
LLM_RERANK_RESULTS = RESULTS_DIR / "llm_rerank_results.json"
QUERY_ATTRIBUTES_RESULTS = RESULTS_DIR / "query_attributes.json"
EVIDENCE_RETRIEVAL_RESULTS = RESULTS_DIR / "evidence_retrieval.json"
EVIDENCE_VERIFIED_RESULTS = RESULTS_DIR / "evidence_verified.json"
LISTWISE_EVIDENCE_RESULTS = RESULTS_DIR / "rerank_listwise_evidence.json"
POINTWISE_EVIDENCE_RESULTS = RESULTS_DIR / "rerank_pointwise_evidence.json"

# ── Dataset ───────────────────────────────────────────────────────
DATASET_NAME = "microsoft/ms_marco"
DATASET_VERSION = "v1.1"
DATASET_SPLIT = "train"
SAMPLE_LIMIT = 5000     # subset size for tractable BM25 indexing

# ── Retrieval / evaluation ────────────────────────────────────────
NUM_EVAL = 500          # queries scored in BM25 baseline
TOP_K_RETRIEVE = 100    # BM25 retrieval depth (for Recall@100)
TOP_K_EXPORT = 20       # candidates persisted for LLM rerank
DEFAULT_K_METRIC = 10   # K for MRR@K / Recall@K reporting

# ── LLM rerank ────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
NUM_RERANK = 50         # queries to rerank (budget cap)
LLM_SLEEP_BETWEEN = 0.3
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 300
PASSAGE_TRUNCATE_CHARS = 400

# ── Query → Attribute decomposition (Day 5) ───────────────────────
NUM_QUERY_ATTRIBUTES = 100    # how many queries to decompose by default
ATTRIBUTE_MAX_TOKENS = 400
ATTRIBUTE_SCHEMA_FIELDS = (
    "intent_type",            # "factual" | "product_search" | "how_to" |
                              # "comparison" | "navigational" | "other"
    "core_concepts",          # list[str] — main subject/entity nouns
    "constraints",            # list[str] — hard filters (price, time, location)
    "important_attributes",   # list[str] — attributes that drive relevance
    "soft_preferences",       # list[str] — nice-to-haves
)

# ── Evidence retrieval (Day 6) ────────────────────────────────────
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EVIDENCE_TOP_K_CANDIDATES = 10   # candidates per query to ground evidence on
EVIDENCE_ALPHA = 0.6              # weight on cosine vs. keyword score
EVIDENCE_MIN_SCORE = 0.10         # below this, no evidence is recorded
EVIDENCE_GROUNDED_FIELDS = (      # which schema fields to retrieve evidence for
    "important_attributes",
    "constraints",
)
EVIDENCE_VERIFY_MAX_TOKENS = 600
EVIDENCE_VERIFY_MIN_CONFIDENCE = 3   # 1 (low) ─ 5 (high)

# ── Attribute-grounded reranking (Day 7) ──────────────────────────
LISTWISE_TOP_K = 20             # candidates fed into the listwise prompt
POINTWISE_TOP_K = 10            # per-(query,candidate) scoring is expensive
POINTWISE_MAX_TOKENS = 200      # {"score": 1-5, "reason": "..."} fits easily
POINTWISE_SCORE_MIN = 1
POINTWISE_SCORE_MAX = 5
POINTWISE_FALLBACK_SCORE = 3    # used when LLM call fails — preserves BM25 tie-break

# ── Reproducibility ───────────────────────────────────────────────
RANDOM_SEED = 42


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PYTHONHASHSEED-style sources."""
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
