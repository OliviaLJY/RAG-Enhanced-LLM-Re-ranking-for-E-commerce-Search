"""
Cross-encoder reranking (no LLM, no API) over BM25 top-K.
基于 BM25 top-K 的 Cross-Encoder 重排序（不调用 LLM / 不需要 API）。

Hard baseline reference for the ablation grid. Uses the standard
cross-encoder/ms-marco-MiniLM-L-6-v2 model — the off-the-shelf
benchmark that any LLM reranker must beat to claim a win.
作为消融实验的强基线参考。任何 LLM 重排序方法都必须超过它才能称之为有效。

Usage / 用法:
    python cross_encoder_rerank.py                    # full 500-query run
    python cross_encoder_rerank.py --num-rerank 5     # smoke test
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from sentence_transformers import CrossEncoder
from tqdm import tqdm

import config
from bm25_baseline import mrr


# ── Metrics ───────────────────────────────────────────────────────

def _doc_ids(q: Dict, use_reranked: bool) -> List[int]:
    if use_reranked:
        return q.get("cross_encoder_reranked_doc_ids", [c["doc_id"] for c in q["candidates"]])
    return [c["doc_id"] for c in q["candidates"]]


def compute_mrr_at_k(queries: List[Dict], use_reranked: bool, k: int) -> float:
    scores = [mrr(_doc_ids(q, use_reranked)[:k], set(q["relevant_doc_ids"])) for q in queries]
    return float(np.mean(scores)) if scores else 0.0


def compute_recall_at_k(queries: List[Dict], use_reranked: bool, k: int) -> float:
    scores = []
    for q in queries:
        rel = set(q["relevant_doc_ids"])
        if not rel:
            scores.append(0.0)
            continue
        hits = sum(1 for d in _doc_ids(q, use_reranked)[:k] if d in rel)
        scores.append(hits / len(rel))
    return float(np.mean(scores)) if scores else 0.0


# ── Reranking ─────────────────────────────────────────────────────

def rerank_query(
    model: CrossEncoder,
    query: str,
    candidates: List[Dict],
    batch_size: int,
) -> List[float]:
    """Return raw cross-encoder scores aligned with the input ``candidates`` order."""
    truncate = config.PASSAGE_TRUNCATE_CHARS
    pairs = [(query, c["passage"][:truncate]) for c in candidates]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(s) for s in scores]


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-encoder reranking on BM25 top-K candidates (hard baseline)."
    )
    p.add_argument("--candidates", type=str, default=str(config.BM25_CANDIDATES),
                   help="Input JSON produced by bm25_export_topk.py")
    p.add_argument("--output", type=str, default=str(config.CROSS_ENCODER_RESULTS))
    p.add_argument("--model", type=str, default=config.CROSS_ENCODER_MODEL)
    p.add_argument("--num-rerank", type=int, default=config.NUM_EVAL,
                   help="Number of queries to rerank.")
    p.add_argument("--top-k-rerank", type=int, default=config.TOP_K_EXPORT,
                   help="How many BM25 candidates to rescore per query.")
    p.add_argument("--batch-size", type=int, default=config.CROSS_ENCODER_BATCH_SIZE)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.seed_everything(args.seed)

    if not os.path.exists(args.candidates):
        raise SystemExit(
            f"{args.candidates} not found. Run bm25_export_topk.py first."
        )

    with open(args.candidates) as f:
        data = json.load(f)

    all_queries = data["queries"]
    queries = all_queries[: args.num_rerank]
    print(f"Loaded {len(all_queries)} queries; reranking first {len(queries)} with {args.model}.")

    k = config.DEFAULT_K_METRIC
    bm25_mrr10 = compute_mrr_at_k(queries, use_reranked=False, k=k)
    bm25_recall10 = compute_recall_at_k(queries, use_reranked=False, k=k)
    bm25_recall100 = compute_recall_at_k(queries, use_reranked=False, k=100)
    print(f"BM25 baseline on subset — MRR@{k}: {bm25_mrr10:.4f}, "
          f"Recall@{k}: {bm25_recall10:.4f}, Recall@100: {bm25_recall100:.4f}")

    print(f"Loading cross-encoder: {args.model} ...")
    model = CrossEncoder(args.model)

    for q in tqdm(queries, desc="Cross-encoder reranking"):
        candidates = q["candidates"][: args.top_k_rerank]
        scores = rerank_query(model, q["query"], candidates, batch_size=args.batch_size)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        q["cross_encoder_reranked_doc_ids"] = [candidates[i]["doc_id"] for i in order]
        q["cross_encoder_scores"] = [scores[i] for i in order]

    ce_mrr10 = compute_mrr_at_k(queries, use_reranked=True, k=k)
    ce_recall10 = compute_recall_at_k(queries, use_reranked=True, k=k)
    ce_recall100 = compute_recall_at_k(queries, use_reranked=True, k=100)

    deltas = []
    for q in queries:
        rel = set(q["relevant_doc_ids"])
        bm25_q = mrr([c["doc_id"] for c in q["candidates"]][:k], rel)
        ce_q = mrr(q["cross_encoder_reranked_doc_ids"][:k], rel)
        deltas.append(ce_q - bm25_q)

    delta_mrr = ce_mrr10 - bm25_mrr10
    rel_gain = (delta_mrr / bm25_mrr10 * 100.0) if bm25_mrr10 > 0 else 0.0
    arrow = "↑" if delta_mrr > 0 else ("↓" if delta_mrr < 0 else "→")

    print("\nResults: BM25 → Cross-Encoder Rerank")
    print(f"  Queries           : {len(queries)}")
    print(f"  MRR@{k}            : {bm25_mrr10:.4f} → {ce_mrr10:.4f}  ({delta_mrr:+.4f} {arrow})")
    print(f"  Recall@{k}         : {bm25_recall10:.4f} → {ce_recall10:.4f}")
    print(f"  Recall@100        : {bm25_recall100:.4f} → {ce_recall100:.4f}")
    print(f"  Relative MRR gain : {rel_gain:+.1f}%")
    print(f"  Improved / Tied / Degraded: "
          f"{sum(1 for d in deltas if d > 0)} / "
          f"{sum(1 for d in deltas if d == 0)} / "
          f"{sum(1 for d in deltas if d < 0)}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "num_queries": len(queries),
        "metrics": {
            "bm25_mrr10": round(bm25_mrr10, 6),
            "ce_mrr10": round(ce_mrr10, 6),
            "delta_mrr10": round(delta_mrr, 6),
            "relative_gain_pct": round(rel_gain, 2),
            "bm25_recall10": round(bm25_recall10, 6),
            "ce_recall10": round(ce_recall10, 6),
            "bm25_recall100": round(bm25_recall100, 6),
            "ce_recall100": round(ce_recall100, 6),
        },
        "per_query_improvement": {
            "mean_delta": round(float(np.mean(deltas)), 6),
            "queries_improved": int(sum(1 for d in deltas if d > 0)),
            "queries_degraded": int(sum(1 for d in deltas if d < 0)),
            "queries_neutral": int(sum(1 for d in deltas if d == 0)),
        },
        "config": {
            "top_k_rerank": args.top_k_rerank,
            "num_rerank": args.num_rerank,
            "batch_size": args.batch_size,
            "passage_truncate_chars": config.PASSAGE_TRUNCATE_CHARS,
            "seed": args.seed,
        },
        "queries": queries,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
