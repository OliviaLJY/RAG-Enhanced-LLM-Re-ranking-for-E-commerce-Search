"""
BM25 baseline for MS MARCO passage ranking.
MS MARCO 段落排序的 BM25 基线。

Stage 1 of the pipeline. Input: query. Output: top-K passages by BM25.
流水线第一步：输入 query，输出 BM25 top-K 段落。

Usage / 用法:
    python bm25_baseline.py --sample-limit 5000 --num-eval 500
"""

import argparse
import json
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import config


class BM25Retriever:
    """Thin wrapper around rank_bm25 with explicit tokenization. / rank_bm25 的轻量封装。"""

    def __init__(self) -> None:
        self.corpus: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def build_index(self, corpus: List[str]) -> None:
        self.corpus = corpus
        self.tokenized_corpus = [doc.lower().split() for doc in tqdm(corpus, desc="Tokenizing")]
        print(f"Corpus size: {len(self.corpus):,} passages")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 index ready.")

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[int, str, float]]:
        """Return ``[(doc_id, passage, score)]`` sorted by descending BM25 score."""
        assert self.bm25 is not None, "Call build_index() first."
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), self.corpus[i], float(scores[i])) for i in top_indices]


# ── Data loading ──────────────────────────────────────────────────

def load_msmarco_data(split: str = config.DATASET_SPLIT, limit: Optional[int] = None):
    """Load MS MARCO v1.1, optionally truncated to ``limit`` samples."""
    print(f"Loading MS MARCO {split} (limit={limit})...")
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_VERSION, split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    print(f"Loaded {len(dataset):,} queries.")
    return dataset


def build_corpus_from_dataset(dataset) -> Tuple[List[str], Dict[int, List[int]]]:
    """
    Flatten passages into a deduped corpus and record per-query relevance.
    去重所有段落，并记录每条 query 的相关文档下标。
    """
    corpus: List[str] = []
    passage_to_idx: Dict[str, int] = {}
    query_to_relevant: Dict[int, List[int]] = {}

    for query_idx, item in enumerate(tqdm(dataset, desc="Building corpus")):
        passages = item["passages"]["passage_text"]
        is_selected = item["passages"]["is_selected"]
        relevant: List[int] = []
        for passage, selected in zip(passages, is_selected):
            corpus_idx = passage_to_idx.get(passage)
            if corpus_idx is None:
                corpus_idx = len(corpus)
                corpus.append(passage)
                passage_to_idx[passage] = corpus_idx
            if selected == 1:
                relevant.append(corpus_idx)
        query_to_relevant[query_idx] = relevant

    print(f"Corpus built: {len(corpus):,} unique passages.")
    return corpus, query_to_relevant


# ── Metrics ───────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for d in retrieved_ids[:k] if d in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    retriever: BM25Retriever,
    dataset,
    query_to_relevant: Dict[int, List[int]],
    top_k: int = config.TOP_K_RETRIEVE,
    num_eval: Optional[int] = None,
) -> Dict[str, float]:
    recalls_10, recalls_100, mrrs_10 = [], [], []
    limit = min(num_eval, len(dataset)) if num_eval else len(dataset)

    for query_idx in tqdm(range(limit), desc="Evaluating"):
        relevant_ids = set(query_to_relevant.get(query_idx, []))
        if not relevant_ids:
            continue
        query = dataset[query_idx]["query"]
        retrieved_ids = [d for d, _, _ in retriever.retrieve(query, top_k=top_k)]
        recalls_10.append(recall_at_k(retrieved_ids, relevant_ids, k=10))
        recalls_100.append(recall_at_k(retrieved_ids, relevant_ids, k=100))
        mrrs_10.append(mrr(retrieved_ids[:10], relevant_ids))

    return {
        "Recall@10": float(np.mean(recalls_10)) if recalls_10 else 0.0,
        "Recall@100": float(np.mean(recalls_100)) if recalls_100 else 0.0,
        "MRR@10": float(np.mean(mrrs_10)) if mrrs_10 else 0.0,
        "Num_Queries": len(recalls_10),
    }


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BM25 baseline on MS MARCO v1.1")
    p.add_argument("--sample-limit", type=int, default=config.SAMPLE_LIMIT,
                   help="Truncate dataset to this many queries (None = full).")
    p.add_argument("--num-eval", type=int, default=config.NUM_EVAL,
                   help="Number of queries to evaluate.")
    p.add_argument("--top-k", type=int, default=config.TOP_K_RETRIEVE,
                   help="Retrieval depth.")
    p.add_argument("--output", type=str, default=str(config.BM25_BASELINE_RESULTS),
                   help="Path to write JSON results.")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.seed_everything(args.seed)

    print(f"BM25 baseline on MS MARCO {config.DATASET_VERSION}")
    dataset = load_msmarco_data(split=config.DATASET_SPLIT, limit=args.sample_limit)
    corpus, query_to_relevant = build_corpus_from_dataset(dataset)

    retriever = BM25Retriever()
    retriever.build_index(corpus)

    metrics = evaluate_retriever(
        retriever, dataset, query_to_relevant,
        top_k=args.top_k, num_eval=args.num_eval,
    )

    print("\nBM25 Baseline Results / BM25 基线结果")
    print(f"  Evaluated queries : {metrics['Num_Queries']}")
    print(f"  Recall@10         : {metrics['Recall@10']:.4f}")
    print(f"  Recall@100        : {metrics['Recall@100']:.4f}")
    print(f"  MRR@10            : {metrics['MRR@10']:.4f}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": "BM25 (rank_bm25)",
            "dataset": f"{config.DATASET_NAME} {config.DATASET_VERSION}",
            "split": config.DATASET_SPLIT,
            "corpus_size": len(corpus),
            "metrics": metrics,
            "config": {
                "sample_limit": args.sample_limit,
                "num_eval": args.num_eval,
                "top_k": args.top_k,
                "seed": args.seed,
            },
        }, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
