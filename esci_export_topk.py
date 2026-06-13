"""
Export Amazon ESCI (Task 1) query-product pools as BM25-ranked candidates.

ESCI is a product-search re-ranking benchmark: every query already ships with
a pool of ~20-40 labeled products (Exact / Substitute / Complement /
Irrelevant). Unlike MS MARCO's mostly-factual queries, ESCI queries are
genuine shopping queries that carry product attributes — exactly the regime
the attribute-grounded evidence method is designed for.

This script emits the SAME JSON schema as ``bm25_export_topk.py`` so the whole
downstream pipeline (query_attributes → evidence_retrieval →
rerank_listwise/pointwise_evidence → evaluate → significance) runs unchanged.

Per query we build a small BM25 index over that query's candidate product
texts and sort by score, mirroring the "BM25 ordering of a candidate pool"
setup used on MS MARCO.

Relevance: ``is_relevant = (esci_label == "Exact")``. The raw label is kept on
each candidate so graded metrics (nDCG) can use S/C/I later.

Usage:
    python esci_export_topk.py --num-queries 500 --top-k 20
    python esci_export_topk.py --num-queries 500 --scan-rows 400000
"""

import argparse
import json
from typing import Dict, List

from rank_bm25 import BM25Okapi
from tqdm import tqdm

import config

ESCI_CANDIDATES = config.RESULTS_DIR / "esci_top20_candidates.json"

PASSAGE_CHARS = 500          # title + bullet point, truncated
MIN_CANDIDATES = 5           # drop queries with too-thin pools
RELEVANT_LABEL = "Exact"


def build_passage(row: Dict) -> str:
    """Compact product text: title + brand + truncated bullet points."""
    title = (row.get("product_title") or "").strip()
    brand = (row.get("product_brand") or "").strip()
    bullet = (row.get("product_bullet_point") or "").strip()
    parts = [p for p in (title, brand, bullet) if p]
    text = ". ".join(parts)
    return text[:PASSAGE_CHARS]


def collect_queries(scan_rows: int, target_queries: int) -> Dict[int, Dict]:
    """Stream ESCI, group us / small_version==1 rows by query_id."""
    from datasets import load_dataset

    ds = load_dataset("tasksource/esci", split="train", streaming=True)
    grouped: Dict[int, Dict] = {}

    for i, row in enumerate(tqdm(ds, total=scan_rows, desc="Scanning ESCI")):
        if i >= scan_rows:
            break
        if row.get("product_locale") != "us" or int(row.get("small_version", 0)) != 1:
            continue
        qid = int(row["query_id"])
        bucket = grouped.setdefault(qid, {"query": row["query"].strip(), "products": []})
        bucket["products"].append({
            "product_id": row["product_id"],
            "passage": build_passage(row),
            "esci_label": row["esci_label"],
        })

    # Drop the highest query_id seen (its group may be truncated by the scan cap)
    if grouped:
        truncated = max(grouped)
        grouped.pop(truncated, None)
    return grouped


def rank_candidates(query: str, products: List[Dict], top_k: int) -> List[Dict]:
    """BM25-rank a query's candidate pool; return top_k candidates with scores."""
    texts = [p["passage"].lower().split() for p in products]
    bm25 = BM25Okapi(texts)
    scores = bm25.get_scores(query.lower().split())
    order = sorted(range(len(products)), key=lambda j: scores[j], reverse=True)[:top_k]

    candidates = []
    for rank, j in enumerate(order, start=1):
        p = products[j]
        candidates.append({
            "rank": rank,
            "doc_id": j,                      # stable int id within this query pool
            "product_id": p["product_id"],
            "passage": p["passage"],
            "bm25_score": round(float(scores[j]), 6),
            "esci_label": p["esci_label"],
            "is_relevant": p["esci_label"] == RELEVANT_LABEL,
        })
    return candidates


def main() -> None:
    p = argparse.ArgumentParser(description="Export ESCI Task-1 query pools as BM25 candidates.")
    p.add_argument("--num-queries", type=int, default=500)
    p.add_argument("--top-k", type=int, default=config.TOP_K_EXPORT)
    p.add_argument("--scan-rows", type=int, default=400_000,
                   help="How many streamed rows to scan before grouping stops.")
    p.add_argument("--output", type=str, default=str(ESCI_CANDIDATES))
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = p.parse_args()
    config.seed_everything(args.seed)

    grouped = collect_queries(args.scan_rows, args.num_queries)
    print(f"Grouped {len(grouped)} candidate queries from scan.")

    queries_data: List[Dict] = []
    skipped = 0
    for qid in sorted(grouped):
        products = grouped[qid]["products"]
        if len(products) < MIN_CANDIDATES:
            skipped += 1
            continue
        candidates = rank_candidates(grouped[qid]["query"], products, args.top_k)
        relevant_ids = [c["doc_id"] for c in candidates if c["is_relevant"]]
        if not relevant_ids:
            skipped += 1
            continue
        queries_data.append({
            "query_id": qid,
            "query": grouped[qid]["query"],
            "relevant_doc_ids": sorted(relevant_ids),
            "candidates": candidates,
        })
        if len(queries_data) >= args.num_queries:
            break

    # ── Stats ─────────────────────────────────────────────────────
    if queries_data:
        avg_cands = sum(len(q["candidates"]) for q in queries_data) / len(queries_data)
        avg_rel = sum(len(q["relevant_doc_ids"]) for q in queries_data) / len(queries_data)
    else:
        avg_cands = avg_rel = 0.0
    print(f"Exported {len(queries_data)} queries  |  skipped {skipped} "
          f"(< {MIN_CANDIDATES} candidates or no Exact).")
    print(f"Avg candidates/query: {avg_cands:.1f}  |  Avg relevant/query: {avg_rel:.1f}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "ESCI Task-1 (us) query pools, BM25-ranked, MS MARCO-compatible schema.",
        "dataset": "tasksource/esci",
        "relevant_label": RELEVANT_LABEL,
        "top_k": args.top_k,
        "num_queries": len(queries_data),
        "config": {"scan_rows": args.scan_rows, "seed": args.seed},
        "queries": queries_data,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}  ({len(queries_data)} queries × top-{args.top_k})")


if __name__ == "__main__":
    main()
