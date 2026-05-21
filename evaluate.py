"""
Reload a results file and recompute metrics — no LLM calls.
重新加载结果文件并计算指标，不调用 LLM。

Useful for comparing different K values, or auditing a saved rerank run.
适合在不同 K 下比较结果，或复核已经保存的重排序产物。

Usage / 用法:
    # Score the BM25 candidates dump (Stage 2 output)
    python evaluate.py results/bm25_top20_candidates.json

    # Score a stored LLM rerank run, at K=10
    python evaluate.py results/bm25_top20_candidates.json --use-llm-order --k 10
"""

import argparse
import json
from typing import Dict, List, Set

import numpy as np

import config


def mrr(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return sum(1 for d in retrieved_ids[:k] if d in relevant_ids) / len(relevant_ids)


def _doc_ids(q: Dict, use_llm_order: bool, rerank_key: str = "llm_reranked_doc_ids") -> List[int]:
    if use_llm_order and rerank_key in q:
        return q[rerank_key]
    return [c["doc_id"] for c in q["candidates"]]


def evaluate(path: str, use_llm_order: bool, k: int,
             rerank_key: str = "llm_reranked_doc_ids") -> Dict[str, float]:
    with open(path) as f:
        data = json.load(f)
    queries = data["queries"] if "queries" in data else data

    if use_llm_order and not any(rerank_key in q for q in queries):
        raise SystemExit(
            f"--use-llm-order was passed but no {rerank_key} found in {path}."
        )

    mrrs, recalls_k, recalls_100 = [], [], []
    for q in queries:
        rel = set(q["relevant_doc_ids"])
        if not rel:
            continue
        ids = _doc_ids(q, use_llm_order, rerank_key)
        mrrs.append(mrr(ids[:k], rel))
        recalls_k.append(recall_at_k(ids, rel, k=k))
        recalls_100.append(recall_at_k(ids, rel, k=100))

    return {
        f"MRR@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"Recall@{k}": float(np.mean(recalls_k)) if recalls_k else 0.0,
        "Recall@100": float(np.mean(recalls_100)) if recalls_100 else 0.0,
        "Num_Queries": len(mrrs),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute Recall/MRR from a results JSON.")
    p.add_argument("path", help="Path to a candidates or rerank JSON.")
    p.add_argument("--use-llm-order", action="store_true",
                   help="Score using a reranked doc-id list instead of BM25 order.")
    p.add_argument("--rerank-key", default="llm_reranked_doc_ids",
                   help="JSON field holding the reranked doc_id list "
                        "(e.g. cross_encoder_reranked_doc_ids).")
    p.add_argument("--k", type=int, default=config.DEFAULT_K_METRIC)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.path, args.use_llm_order, args.k, args.rerank_key)
    order = args.rerank_key if args.use_llm_order else "BM25"
    print(f"\nMetrics on {args.path} (order = {order}):")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<14}: {val:.4f}")
        else:
            print(f"  {key:<14}: {val}")


if __name__ == "__main__":
    main()
