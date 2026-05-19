"""
Export BM25 top-K candidates per query for downstream reranking.
为下游 LLM 重排序导出每条 query 的 BM25 top-K 候选。

Stage 2 of the pipeline. Reads MS MARCO with the same slicing as the
baseline, persists ranked candidates with relevance labels.
流水线第二步：与基线使用同一切片，导出带相关性标注的候选段落。

Usage / 用法:
    python bm25_export_topk.py --num-queries 500 --top-k 20
"""

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

import config
from bm25_baseline import (
    BM25Retriever,
    build_corpus_from_dataset,
    load_msmarco_data,
    mrr,
)


def export_top_k_candidates(
    retriever: BM25Retriever,
    dataset,
    query_to_relevant: Dict[int, List[int]],
    top_k: int = config.TOP_K_EXPORT,
    num_queries: Optional[int] = None,
) -> List[Dict]:
    """
    Retrieve BM25 top-K for each query, attach relevance labels, return
    JSON-ready dicts. Queries with no relevant docs are skipped.
    """
    queries_data: List[Dict] = []
    limit = min(num_queries, len(dataset)) if num_queries else len(dataset)
    skipped = 0

    for query_idx in tqdm(range(limit), desc=f"Exporting top-{top_k}"):
        relevant_ids = set(query_to_relevant.get(query_idx, []))
        if not relevant_ids:
            skipped += 1
            continue

        query = dataset[query_idx]["query"]
        results = retriever.retrieve(query, top_k=top_k)

        candidates = [
            {
                "rank": rank,
                "doc_id": doc_id,
                "passage": passage,
                "bm25_score": round(score, 6),
                "is_relevant": doc_id in relevant_ids,
            }
            for rank, (doc_id, passage, score) in enumerate(results, start=1)
        ]

        queries_data.append({
            "query_id": query_idx,
            "query": query,
            "relevant_doc_ids": sorted(relevant_ids),
            "candidates": candidates,
        })

    print(f"Exported {len(queries_data)} queries  |  Skipped {skipped} (no relevance label).")
    return queries_data


def compute_bm25_mrr_at_k(queries_data: List[Dict], k: int = config.DEFAULT_K_METRIC) -> float:
    scores = [
        mrr([c["doc_id"] for c in q["candidates"]][:k], set(q["relevant_doc_ids"]))
        for q in queries_data
    ]
    return float(np.mean(scores)) if scores else 0.0


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export BM25 top-K candidates per query.")
    p.add_argument("--sample-limit", type=int, default=config.SAMPLE_LIMIT)
    p.add_argument("--num-queries", type=int, default=config.NUM_EVAL,
                   help="Number of queries to export candidates for.")
    p.add_argument("--top-k", type=int, default=config.TOP_K_EXPORT)
    p.add_argument("--output", type=str, default=str(config.BM25_CANDIDATES))
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.seed_everything(args.seed)

    print(f"Exporting BM25 top-{args.top_k} candidates for up to {args.num_queries} queries.")

    dataset = load_msmarco_data(split=config.DATASET_SPLIT, limit=args.sample_limit)
    corpus, query_to_relevant = build_corpus_from_dataset(dataset)

    retriever = BM25Retriever()
    retriever.build_index(corpus)

    queries_data = export_top_k_candidates(
        retriever, dataset, query_to_relevant,
        top_k=args.top_k, num_queries=args.num_queries,
    )

    bm25_mrr10 = compute_bm25_mrr_at_k(queries_data, k=config.DEFAULT_K_METRIC)
    print(f"BM25 MRR@{config.DEFAULT_K_METRIC} on exported queries: {bm25_mrr10:.4f}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "BM25 top-K candidates per query with relevance labels.",
        "corpus_size": len(corpus),
        "top_k": args.top_k,
        "num_queries": len(queries_data),
        "bm25_mrr10": round(bm25_mrr10, 6),
        "config": {
            "sample_limit": args.sample_limit,
            "num_queries_requested": args.num_queries,
            "seed": args.seed,
        },
        "queries": queries_data,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved: {args.output}  ({len(queries_data)} queries × top-{args.top_k})")


if __name__ == "__main__":
    main()
