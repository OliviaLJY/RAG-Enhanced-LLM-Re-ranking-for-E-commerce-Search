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
