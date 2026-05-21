"""
Long-tail analysis: where does a reranker's lift over BM25 actually concentrate?
长尾分析：重排序方法相对 BM25 的提升究竟分布在哪些 query 桶？

Buckets queries by the BM25 rank of their first relevant document, then reports
per-bucket mean MRR for both methods. The "rank 11-20" and "not in top-K"
buckets are the long tail where BM25 fails and rerankers should shine; the
"rank 1" bucket is the ceiling where rerankers can only tie or hurt.
按 BM25 首条相关文档的排名分桶，分别比较两种方法在每个桶里的平均 MRR。

Usage / 用法:
    python long_tail_analysis.py \\
        --baseline results/bm25_top20_candidates.json \\
        --method   results/cross_encoder_rerank_results.json \\
        --method-key cross_encoder_reranked_doc_ids \\
        --method-label cross_encoder
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import config


# ── Helpers ───────────────────────────────────────────────────────

def _first_relevant_rank(retrieved: List[int], relevant: Set[int]) -> Optional[int]:
    """1-indexed rank of the first relevant doc, or None if absent."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return rank
    return None


def _mrr_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _order(q: Dict, rerank_key: str) -> List[int]:
    if rerank_key and rerank_key in q:
        return q[rerank_key]
    return [c["doc_id"] for c in q["candidates"]]


def load_queries(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    return data["queries"] if "queries" in data else data


def align(baseline: List[Dict], method: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    b_by_id = {q["query_id"]: q for q in baseline if q.get("relevant_doc_ids")}
    m_by_id = {q["query_id"]: q for q in method if q.get("relevant_doc_ids")}
    common = sorted(set(b_by_id) & set(m_by_id))
    return [b_by_id[i] for i in common], [m_by_id[i] for i in common]


# ── Bucketing ─────────────────────────────────────────────────────

BUCKETS: List[Tuple[str, Optional[int], Optional[int]]] = [
    # (label, lo_rank_inclusive, hi_rank_inclusive)  — None for unbounded high
    ("rank 1",         1,  1),
    ("rank 2-3",       2,  3),
    ("rank 4-10",      4,  10),
    ("rank 11-20",     11, 20),
    ("not in top-K",   None, None),   # special: bm25 missed entirely
]


def bucket_for_rank(rank: Optional[int]) -> str:
    if rank is None:
        return "not in top-K"
    for label, lo, hi in BUCKETS:
        if lo is None:
            continue
        if lo <= rank <= hi:
            return label
    return "not in top-K"   # rank > 20 collapsed in with the unrecoverable tail


# ── Main analysis ─────────────────────────────────────────────────

def analyse(
    baseline_q: List[Dict],
    method_q: List[Dict],
    baseline_key: str,
    method_key: str,
    k: int,
) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {label: {
        "n": 0,
        "baseline_mrr_sum": 0.0,
        "method_mrr_sum": 0.0,
        "delta_sum": 0.0,
        "wins": 0, "ties": 0, "losses": 0,
    } for label, _, _ in BUCKETS}

    for bq, mq in zip(baseline_q, method_q):
        rel = set(bq["relevant_doc_ids"])
        if not rel:
            continue
        bm25_order = _order(bq, baseline_key)
        method_order = _order(mq, method_key)
        first = _first_relevant_rank(bm25_order, rel)
        bucket = bucket_for_rank(first)

        bm25_mrr = _mrr_at_k(bm25_order, rel, k)
        method_mrr = _mrr_at_k(method_order, rel, k)
        delta = method_mrr - bm25_mrr

        r = rows[bucket]
        r["n"] += 1
        r["baseline_mrr_sum"] += bm25_mrr
        r["method_mrr_sum"] += method_mrr
        r["delta_sum"] += delta
        if delta > 0:
            r["wins"] += 1
        elif delta < 0:
            r["losses"] += 1
        else:
            r["ties"] += 1

    out: Dict[str, Dict] = {}
    for label, _, _ in BUCKETS:
        r = rows[label]
        n = r["n"]
        out[label] = {
            "n": n,
            "baseline_mean_mrr": (r["baseline_mrr_sum"] / n) if n else 0.0,
            "method_mean_mrr":   (r["method_mrr_sum"]   / n) if n else 0.0,
            "mean_delta":        (r["delta_sum"]        / n) if n else 0.0,
            "wins": r["wins"], "ties": r["ties"], "losses": r["losses"],
        }
    return out


# ── Output ────────────────────────────────────────────────────────

def print_table(buckets: Dict[str, Dict], method_label: str, k: int) -> None:
    headers = ["bucket", "n", f"bm25_mrr{k}", f"{method_label}_mrr{k}", "delta", "win/tie/loss"]
    rows = [headers]
    for label, _, _ in BUCKETS:
        b = buckets[label]
        rows.append([
            label,
            str(b["n"]),
            f"{b['baseline_mean_mrr']:.4f}",
            f"{b['method_mean_mrr']:.4f}",
            f"{b['mean_delta']:+.4f}",
            f"{b['wins']}/{b['ties']}/{b['losses']}",
        ])

    widths = [max(len(r[c]) for r in rows) for c in range(len(headers))]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    for i, row in enumerate(rows):
        print(fmt.format(*row))
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))


def identify_concentration(buckets: Dict[str, Dict]) -> str:
    contributions = []
    for label, _, _ in BUCKETS:
        b = buckets[label]
        if b["n"] == 0:
            continue
        # bucket's share of total absolute lift
        contributions.append((label, b["mean_delta"] * b["n"], b["n"], b["mean_delta"]))
    contributions.sort(key=lambda x: x[1], reverse=True)
    top = [(label, delta) for label, _, _, delta in contributions[:2] if delta > 1e-4]
    if not top:
        return "no positive bucket — method does not lift over BM25 on any bucket."
    parts = [f"{label} (mean Δ {delta:+.3f})" for label, delta in top]
    return "lift concentrates in " + " and ".join(parts) + "."


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-bucket lift analysis: where does a reranker beat BM25?"
    )
    p.add_argument("--baseline", required=True,
                   help="Baseline results JSON (e.g. bm25_top20_candidates.json).")
    p.add_argument("--method", required=True,
                   help="Method results JSON (e.g. cross_encoder_rerank_results.json).")
    p.add_argument("--baseline-key", default="",
                   help="Doc-id field in baseline JSON. Empty = use BM25 candidate order.")
    p.add_argument("--method-key", default="llm_reranked_doc_ids",
                   help="Doc-id field in method JSON (e.g. cross_encoder_reranked_doc_ids).")
    p.add_argument("--method-label", default="",
                   help="Short label for the method column. Default = derived from filename.")
    p.add_argument("--k", type=int, default=config.DEFAULT_K_METRIC)
    p.add_argument("--output", type=str, default="",
                   help="Optional JSON output path. Default = no file written.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    baseline_q, method_q = align(load_queries(args.baseline), load_queries(args.method))
    n = len(baseline_q)
    if n == 0:
        raise SystemExit(
            "No queries shared between baseline and method (matched by query_id)."
        )

    label = args.method_label or Path(args.method).stem.replace("_results", "")
    buckets = analyse(baseline_q, method_q, args.baseline_key, args.method_key, args.k)

    print(f"\nLong-tail analysis (MRR@{args.k}): {label} vs baseline")
    print(f"  baseline file: {args.baseline}")
    print(f"  method   file: {args.method}")
    print(f"  paired queries: {n}\n")
    print_table(buckets, label, args.k)
    print(f"\n  → {identify_concentration(buckets)}")
    print("  Note: 'not in top-K' is unrecoverable — rerankers cannot promote "
          "docs that aren't in the candidate list.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline_file": args.baseline,
            "method_file": args.method,
            "baseline_key": args.baseline_key or "<bm25_candidate_order>",
            "method_key": args.method_key,
            "method_label": label,
            "k": args.k,
            "num_paired_queries": n,
            "buckets": buckets,
            "bucket_definitions": [
                {"label": lbl, "lo_rank": lo, "hi_rank": hi} for lbl, lo, hi in BUCKETS
            ],
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
