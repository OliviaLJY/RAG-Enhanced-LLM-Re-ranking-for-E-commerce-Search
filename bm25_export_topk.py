"""
Day 3: Export BM25 Top-K Candidates
=====================================
目标：为每个 query 保存 BM25 top-20 candidates，供 LLM reranking (Day 4) 使用

输入：MS MARCO v1.1 (same 5000-sample slice as Day 1-2)
输出：results/bm25_top20_candidates.json

格式：
{
  "metadata": { ... },
  "queries": [
    {
      "query_id": int,
      "query": str,
      "relevant_doc_ids": [int, ...],
      "candidates": [
        {"rank": int, "doc_id": int, "passage": str, "bm25_score": float, "is_relevant": bool},
        ...
      ]
    },
    ...
  ]
}
"""

import json
import os
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional

from bm25_baseline import (
    BM25Retriever,
    load_msmarco_data,
    build_corpus_from_dataset,
    mrr,
)


def export_top_k_candidates(
    retriever: BM25Retriever,
    dataset,
    query_to_relevant: Dict[int, List[int]],
    top_k: int = 20,
    num_queries: Optional[int] = None,
) -> List[Dict]:
    """
    Retrieve BM25 top-k passages for each query and attach relevance labels.

    Returns a list of query dicts ready for JSON serialization.
    """
    queries_data = []
    limit = min(num_queries, len(dataset)) if num_queries else len(dataset)
    skipped = 0

    for query_idx in tqdm(range(limit), desc=f"Exporting top-{top_k}"):
        relevant_ids = set(query_to_relevant.get(query_idx, []))

        if len(relevant_ids) == 0:
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

        queries_data.append(
            {
                "query_id": query_idx,
                "query": query,
                "relevant_doc_ids": list(relevant_ids),
                "candidates": candidates,
            }
        )

    print(f"  Exported: {len(queries_data)} queries  |  Skipped (no relevance): {skipped}")
    return queries_data


def compute_bm25_mrr_at_k(queries_data: List[Dict], k: int = 10) -> float:
    """Compute MRR@k on exported BM25 candidates (sanity-check vs Day 1-2)."""
    scores = [
        mrr([c["doc_id"] for c in q["candidates"]][:k], set(q["relevant_doc_ids"]))
        for q in queries_data
    ]
    return float(np.mean(scores)) if scores else 0.0


def main():
    print("=" * 60)
    print("📦 Day 3: Export BM25 Top-20 Candidates")
    print("=" * 60)

    SAMPLE_LIMIT = 5000   # same as Day 1-2
    NUM_QUERIES = 500     # how many queries to process
    TOP_K = 20            # passages per query

    # ── Load & index ──────────────────────────────────────────────
    dataset = load_msmarco_data(split="train", limit=SAMPLE_LIMIT)
    corpus, query_to_relevant = build_corpus_from_dataset(dataset)

    retriever = BM25Retriever()
    retriever.build_index(corpus)

    # ── Export ────────────────────────────────────────────────────
    print(f"\n🔍 Retrieving top-{TOP_K} candidates for up to {NUM_QUERIES} queries...")
    queries_data = export_top_k_candidates(
        retriever,
        dataset,
        query_to_relevant,
        top_k=TOP_K,
        num_queries=NUM_QUERIES,
    )

    # ── Sanity-check MRR ──────────────────────────────────────────
    bm25_mrr10 = compute_bm25_mrr_at_k(queries_data, k=10)
    print(f"\n📊 BM25 MRR@10 on exported queries: {bm25_mrr10:.4f}")
    print(f"   (Day 1-2 baseline was 0.3654 on 490 queries — should be close)")

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    output = {
        "metadata": {
            "day": "Day 3",
            "description": "BM25 top-20 candidates per query, with relevance labels",
            "corpus_size": len(corpus),
            "top_k": TOP_K,
            "num_queries": len(queries_data),
            "bm25_mrr10": round(bm25_mrr10, 6),
            "config": {
                "sample_limit": SAMPLE_LIMIT,
                "num_queries_requested": NUM_QUERIES,
            },
        },
        "queries": queries_data,
    }

    out_path = "results/bm25_top20_candidates.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved: {out_path}")
    print(f"   {len(queries_data)} queries  ×  top-{TOP_K} candidates")
    print("\n✅ Day 3 完成！→ Run llm_rerank.py for Day 4")


if __name__ == "__main__":
    main()
