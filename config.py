"""
Shared configuration for the BM25 → LLM rerank pipeline.
共享配置：BM25 → LLM 重排序流水线

All hardcoded constants live here so scripts stay short and reproducible.
所有硬编码常量集中在这里，方便复现实验。
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent


def _load_env_file() -> None:
    """Load project-root `.env` into os.environ (no python-dotenv dependency)."""
    env_path = ROOT_DIR / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


_load_env_file()
RESULTS_DIR = ROOT_DIR / "results"

BM25_BASELINE_RESULTS = RESULTS_DIR / "bm25_baseline_results.json"
BM25_CANDIDATES = RESULTS_DIR / "bm25_top20_candidates.json"
LLM_RERANK_RESULTS = RESULTS_DIR / "llm_rerank_results.json"
CROSS_ENCODER_RESULTS = RESULTS_DIR / "cross_encoder_rerank_results.json"
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
LLM_MODEL = "deepseek-v4-flash"
LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
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
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_BATCH_SIZE = 64
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

# ── Significance testing ──────────────────────────────────────────
BOOTSTRAP_ITERATIONS = 10_000   # paired bootstrap resamples
BOOTSTRAP_CI_LEVEL = 0.95       # two-sided confidence level


def make_llm_client():
    """OpenAI-compatible client (DeepSeek, OpenAI, etc.) from env / .env."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env.")
    return OpenAI(api_key=api_key, base_url=LLM_BASE_URL)


def llm_extra_body() -> dict | None:
    """
    DeepSeek V4 defaults to thinking mode, which can consume the entire
    max_tokens budget and return empty ``content`` (finish_reason=length).
    Disable thinking for short ranking / JSON outputs.
    """
    if "deepseek" in LLM_BASE_URL.lower():
        return {"thinking": {"type": "disabled"}}
    return None


def llm_chat_kwargs(**overrides) -> dict:
    """Kwargs merged into ``client.chat.completions.create``."""
    kwargs: dict = {}
    extra = llm_extra_body()
    if extra:
        kwargs["extra_body"] = extra
    kwargs.update(overrides)
    return kwargs


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PYTHONHASHSEED-style sources."""
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
