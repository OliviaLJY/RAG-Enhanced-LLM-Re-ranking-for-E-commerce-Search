"""
LLM listwise reranking (no RAG) over BM25 top-K.
基于 BM25 top-K 的 LLM 列表式重排（不使用 RAG 证据）。

Stage 3 of the pipeline. RankGPT-style: shows the LLM all K passages at once
and parses the returned ordering "[3] > [1] > [2] > ...".
流水线第三步：RankGPT 风格的列表式重排序，解析 "[3] > [1] > [2] > ..." 输出。

Usage / 用法:
    export OPENAI_API_KEY=sk-...
    python llm_rerank.py --num-rerank 50 --model gpt-4o-mini
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm

import config
from bm25_baseline import mrr


# ── Prompt construction & parsing ─────────────────────────────────

def build_reranking_prompt(query: str, passages: List[str]) -> str:
    """RankGPT-style listwise prompt. Passages are truncated to keep tokens bounded."""
    n = len(passages)
    truncate = config.PASSAGE_TRUNCATE_CHARS
    block = "".join(
        f"[{i}] {(p[:truncate] + '...') if len(p) > truncate else p}\n\n"
        for i, p in enumerate(passages, 1)
    )
    return (
        f"I will provide you with {n} passages, each indicated by a numerical "
        f"identifier []. Rank the passages based on their relevance to the search "
        f"query: {query}\n\n"
        f"{block}"
        f"Search Query: {query}\n\n"
        f"Rank the {n} passages above based on their relevance to the search query. "
        f"The passages should be listed in descending order using identifiers. "
        f"The most relevant passages should be listed first. "
        f"The output format should be [] > [] > [], e.g., [2] > [1] > [3]. "
        f"Only respond with the ranking results, do not say any word or explain."
    )


def parse_ranking(response: str, num_passages: int) -> List[int]:
    """
    Parse "[2] > [1] > [3]" into 0-indexed positions.
    Any positions the LLM omitted are appended in their original BM25 order so
    the returned list always has length ``num_passages``.
    """
    numbers = re.findall(r"\[(\d+)\]", response)
    ranking: List[int] = []
    seen: set = set()

    for n_str in numbers:
        n = int(n_str)
        if 1 <= n <= num_passages and n not in seen:
            ranking.append(n - 1)
            seen.add(n)

    for i in range(num_passages):
        if i not in seen:
            ranking.append(i)
    return ranking


# ── LLM call ──────────────────────────────────────────────────────

def rerank_query(
    client: OpenAI,
    query: str,
    candidates: List[Dict],
    model: str = config.LLM_MODEL,
    max_retries: int = 2,
) -> Optional[List[int]]:
    """Return reranked 0-indexed positions into ``candidates``, or None on failure."""
    prompt = build_reranking_prompt(query, [c["passage"] for c in candidates])

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            raw = response.choices[0].message.content.strip()
            return parse_ranking(raw, len(candidates))
        except Exception as e:  # noqa: BLE001
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"\n  Attempt {attempt + 1} failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  All retries failed for query: {query[:60]}...")
                return None


# ── Metrics ───────────────────────────────────────────────────────

def _doc_ids(q: Dict, use_llm_order: bool) -> List[int]:
    if use_llm_order:
        return q.get("llm_reranked_doc_ids", [c["doc_id"] for c in q["candidates"]])
    return [c["doc_id"] for c in q["candidates"]]


def compute_mrr_at_k(queries: List[Dict], use_llm_order: bool, k: int) -> float:
    scores = [mrr(_doc_ids(q, use_llm_order)[:k], set(q["relevant_doc_ids"])) for q in queries]
    return float(np.mean(scores)) if scores else 0.0


def compute_recall_at_k(queries: List[Dict], use_llm_order: bool, k: int) -> float:
    scores = []
    for q in queries:
        rel = set(q["relevant_doc_ids"])
        if not rel:
            scores.append(0.0)
            continue
        hits = sum(1 for d in _doc_ids(q, use_llm_order)[:k] if d in rel)
        scores.append(hits / len(rel))
    return float(np.mean(scores)) if scores else 0.0


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM listwise reranking on BM25 top-K candidates.")
    p.add_argument("--candidates", type=str, default=str(config.BM25_CANDIDATES),
                   help="Input JSON produced by bm25_export_topk.py")
    p.add_argument("--output", type=str, default=str(config.LLM_RERANK_RESULTS))
    p.add_argument("--model", type=str, default=config.LLM_MODEL)
    p.add_argument("--num-rerank", type=int, default=config.NUM_RERANK,
                   help="Number of queries to rerank (budget cap).")
    p.add_argument("--top-k-rerank", type=int, default=config.TOP_K_EXPORT,
                   help="How many BM25 candidates to pass to the LLM.")
    p.add_argument("--sleep", type=float, default=config.LLM_SLEEP_BETWEEN)
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
    bm25_mrr10 = compute_mrr_at_k(queries, use_llm_order=False, k=k)
    bm25_recall5 = compute_recall_at_k(queries, use_llm_order=False, k=5)
    print(f"BM25 baseline on subset — MRR@{k}: {bm25_mrr10:.4f}, Recall@5: {bm25_recall5:.4f}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. See .env.example.")

    client = OpenAI(api_key=api_key)
    failed = 0
    for q in tqdm(queries, desc="LLM reranking"):
        candidates = q["candidates"][: args.top_k_rerank]
        reranked = rerank_query(client, q["query"], candidates, model=args.model)
        original_doc_ids = [c["doc_id"] for c in candidates]
        if reranked is not None:
            q["llm_reranked_doc_ids"] = [original_doc_ids[i] for i in reranked]
            q["llm_rerank_success"] = True
        else:
            q["llm_reranked_doc_ids"] = original_doc_ids
            q["llm_rerank_success"] = False
            failed += 1
        time.sleep(args.sleep)

    llm_mrr10 = compute_mrr_at_k(queries, use_llm_order=True, k=k)
    llm_recall5 = compute_recall_at_k(queries, use_llm_order=True, k=5)

    deltas = []
    for q in queries:
        rel = set(q["relevant_doc_ids"])
        bm25_q = mrr([c["doc_id"] for c in q["candidates"]][:k], rel)
        llm_q = mrr(q["llm_reranked_doc_ids"][:k], rel)
        deltas.append(llm_q - bm25_q)

    delta_mrr = llm_mrr10 - bm25_mrr10
    rel_gain = (delta_mrr / bm25_mrr10 * 100.0) if bm25_mrr10 > 0 else 0.0
    arrow = "↑" if delta_mrr > 0 else ("↓" if delta_mrr < 0 else "→")

    print("\nResults: BM25 → LLM Rerank")
    print(f"  Queries           : {len(queries)} ({failed} fell back to BM25 on failure)")
    print(f"  MRR@{k}            : {bm25_mrr10:.4f} → {llm_mrr10:.4f}  ({delta_mrr:+.4f} {arrow})")
    print(f"  Recall@5          : {bm25_recall5:.4f} → {llm_recall5:.4f}")
    print(f"  Relative MRR gain : {rel_gain:+.1f}%")
    print(f"  Improved / Tied / Degraded: "
          f"{sum(1 for d in deltas if d > 0)} / "
          f"{sum(1 for d in deltas if d == 0)} / "
          f"{sum(1 for d in deltas if d < 0)}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "num_queries": len(queries),
            "failed_reranks": failed,
            "metrics": {
                "bm25_mrr10": round(bm25_mrr10, 6),
                "llm_rerank_mrr10": round(llm_mrr10, 6),
                "delta_mrr10": round(delta_mrr, 6),
                "relative_gain_pct": round(rel_gain, 2),
                "bm25_recall5": round(bm25_recall5, 6),
                "llm_rerank_recall5": round(llm_recall5, 6),
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
                "sleep_between": args.sleep,
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
                "passage_truncate_chars": config.PASSAGE_TRUNCATE_CHARS,
                "seed": args.seed,
            },
        }, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
