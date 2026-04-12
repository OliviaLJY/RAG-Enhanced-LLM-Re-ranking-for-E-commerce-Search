"""
Day 4: LLM Reranking (no RAG)
=====================================
目标：用 LLM 对 BM25 top-20 进行重排序，验证 MRR@10 提升

Pipeline:  BM25 top-20  →  LLM listwise rerank  →  MRR@10
Milestone: BM25 MRR@10 < LLM Rerank MRR@10  ✅

输入：results/bm25_top20_candidates.json  (由 Day 3 生成)
输出：results/llm_rerank_results.json

Approach:
  RankGPT-style listwise reranking — give LLM all 20 passages at once,
  ask it to output a ranked order "[3] > [1] > [2] > ..."
  Model: gpt-4o-mini (cheap & fast)

Usage:
  export OPENAI_API_KEY=<your-key>
  python llm_rerank.py
"""

import json
import os
import re
import time
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional

from openai import OpenAI

from bm25_baseline import mrr


# ─────────────────────────────────────────────────────────────────
#  Prompt & Parsing
# ─────────────────────────────────────────────────────────────────

def build_reranking_prompt(query: str, passages: List[str]) -> str:
    """
    RankGPT-style listwise prompt.
    Passages are truncated to ~400 chars to keep token count manageable.
    Expected output: [2] > [5] > [1] > ...
    """
    n = len(passages)
    passage_block = ""
    for i, p in enumerate(passages, 1):
        truncated = p[:400] + "..." if len(p) > 400 else p
        passage_block += f"[{i}] {truncated}\n\n"

    return (
        f"I will provide you with {n} passages, each indicated by a numerical "
        f"identifier []. Rank the passages based on their relevance to the search "
        f"query: {query}\n\n"
        f"{passage_block}"
        f"Search Query: {query}\n\n"
        f"Rank the {n} passages above based on their relevance to the search query. "
        f"The passages should be listed in descending order using identifiers. "
        f"The most relevant passages should be listed first. "
        f"The output format should be [] > [] > [], e.g., [2] > [1] > [3]. "
        f"Only respond with the ranking results, do not say any word or explain."
    )


def parse_ranking(response: str, num_passages: int) -> List[int]:
    """
    Parse "[2] > [1] > [3]" → 0-indexed list [1, 0, 2].

    Any passage numbers missing from the LLM output are appended at the end
    in their original (BM25) order, so the full list always has length num_passages.
    """
    numbers = re.findall(r"\[(\d+)\]", response)
    ranking: List[int] = []
    seen: set = set()

    for n_str in numbers:
        n = int(n_str)
        if 1 <= n <= num_passages and n not in seen:
            ranking.append(n - 1)   # convert to 0-indexed
            seen.add(n)

    # Preserve BM25 order for any position the LLM skipped
    for i in range(num_passages):
        if i not in seen:
            ranking.append(i)

    return ranking


# ─────────────────────────────────────────────────────────────────
#  LLM Reranking
# ─────────────────────────────────────────────────────────────────

def rerank_query(
    client: OpenAI,
    query: str,
    candidates: List[Dict],
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
) -> Optional[List[int]]:
    """
    Rerank candidates for a single query.

    Returns 0-indexed reranked positions into `candidates`, or None on failure.
    """
    passages = [c["passage"] for c in candidates]
    prompt = build_reranking_prompt(query, passages)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            return parse_ranking(raw, len(passages))
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"\n  ⚠ Attempt {attempt + 1} failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  ✗ All retries failed for query: {query[:60]}...")
                return None


# ─────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────

def compute_mrr_at_k(queries: List[Dict], use_llm_order: bool, k: int = 10) -> float:
    scores = []
    for q in queries:
        relevant_ids = set(q["relevant_doc_ids"])
        if use_llm_order:
            doc_ids = q.get("llm_reranked_doc_ids", [c["doc_id"] for c in q["candidates"]])
        else:
            doc_ids = [c["doc_id"] for c in q["candidates"]]
        scores.append(mrr(doc_ids[:k], relevant_ids))
    return float(np.mean(scores)) if scores else 0.0


def compute_recall_at_k(queries: List[Dict], use_llm_order: bool, k: int) -> float:
    scores = []
    for q in queries:
        relevant_ids = set(q["relevant_doc_ids"])
        if use_llm_order:
            doc_ids = q.get("llm_reranked_doc_ids", [c["doc_id"] for c in q["candidates"]])
        else:
            doc_ids = [c["doc_id"] for c in q["candidates"]]
        hits = sum(1 for d in doc_ids[:k] if d in relevant_ids)
        scores.append(hits / len(relevant_ids) if relevant_ids else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🤖 Day 4: LLM Reranking (no RAG)")
    print("=" * 60)

    # ── Config ────────────────────────────────────────────────────
    MODEL = "gpt-4o-mini"
    NUM_RERANK = 50        # queries to rerank (budget control; ~$0.02 with gpt-4o-mini)
    TOP_K_RERANK = 20      # how many BM25 candidates to pass to LLM
    SLEEP_BETWEEN = 0.3    # seconds between API calls

    # ── Load Day 3 output ─────────────────────────────────────────
    candidates_path = "results/bm25_top20_candidates.json"
    if not os.path.exists(candidates_path):
        print(f"❌  {candidates_path} not found.")
        print("    Run Day 3 first:  python bm25_export_topk.py")
        return

    with open(candidates_path) as f:
        data = json.load(f)

    all_queries = data["queries"]
    queries = all_queries[:NUM_RERANK]
    print(f"📂 Loaded {len(all_queries)} queries → using first {len(queries)} for LLM reranking")

    # ── BM25 baseline MRR@10 (on this subset) ─────────────────────
    bm25_mrr10 = compute_mrr_at_k(queries, use_llm_order=False, k=10)
    bm25_recall5 = compute_recall_at_k(queries, use_llm_order=False, k=5)
    print(f"\n📊 BM25 baseline on {len(queries)} queries:")
    print(f"   MRR@10:    {bm25_mrr10:.4f}")
    print(f"   Recall@5:  {bm25_recall5:.4f}")

    # ── API key check ─────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n❌  OPENAI_API_KEY is not set.")
        print("    Run:  export OPENAI_API_KEY=sk-...")
        return

    client = OpenAI(api_key=api_key)
    print(f"\n🔄 Reranking {len(queries)} queries with {MODEL} (top-{TOP_K_RERANK} each)...")
    print(f"   Estimated cost: ~${len(queries) * TOP_K_RERANK * 400 / 1_000_000 * 0.15:.3f} USD\n")

    # ── Reranking loop ────────────────────────────────────────────
    failed = 0
    for q in tqdm(queries, desc="LLM reranking"):
        candidates = q["candidates"][:TOP_K_RERANK]
        reranked_order = rerank_query(client, q["query"], candidates, model=MODEL)

        original_doc_ids = [c["doc_id"] for c in candidates]
        if reranked_order is not None:
            q["llm_reranked_doc_ids"] = [original_doc_ids[i] for i in reranked_order]
            q["llm_rerank_success"] = True
        else:
            q["llm_reranked_doc_ids"] = original_doc_ids   # fallback: keep BM25 order
            q["llm_rerank_success"] = False
            failed += 1

        time.sleep(SLEEP_BETWEEN)

    # ── LLM rerank MRR@10 ─────────────────────────────────────────
    llm_mrr10 = compute_mrr_at_k(queries, use_llm_order=True, k=10)
    llm_recall5 = compute_recall_at_k(queries, use_llm_order=True, k=5)

    # ── Per-query delta ───────────────────────────────────────────
    deltas = []
    for q in queries:
        rel = set(q["relevant_doc_ids"])
        bm25_q = mrr([c["doc_id"] for c in q["candidates"]][:10], rel)
        llm_q = mrr(q["llm_reranked_doc_ids"][:10], rel)
        deltas.append(llm_q - bm25_q)

    delta_mrr = llm_mrr10 - bm25_mrr10
    arrow = "↑" if delta_mrr > 0 else ("↓" if delta_mrr < 0 else "→")

    # ── Print results ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📈 Results: BM25 → LLM Rerank")
    print("=" * 60)
    print(f"  Queries evaluated:    {len(queries)}")
    print(f"  Failed reranks:       {failed}  (fell back to BM25 order)")
    print(f"")
    print(f"  Metric        BM25      LLM       Δ")
    print(f"  ──────────────────────────────────────")
    print(f"  MRR@10        {bm25_mrr10:.4f}    {llm_mrr10:.4f}    {delta_mrr:+.4f} {arrow}")
    print(f"  Recall@5      {bm25_recall5:.4f}    {llm_recall5:.4f}    {llm_recall5 - bm25_recall5:+.4f}")
    print(f"")
    rel_gain = delta_mrr / bm25_mrr10 * 100 if bm25_mrr10 > 0 else 0
    print(f"  Relative MRR@10 gain: {rel_gain:+.1f}%")
    print(f"  Queries improved:     {sum(1 for d in deltas if d > 0)}/{len(queries)}")
    print(f"  Queries degraded:     {sum(1 for d in deltas if d < 0)}/{len(queries)}")
    print("=" * 60)

    # ── Save results ──────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    results = {
        "day": "Day 4",
        "model": MODEL,
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
            "top_k_rerank": TOP_K_RERANK,
            "num_rerank": NUM_RERANK,
            "sleep_between": SLEEP_BETWEEN,
        },
    }

    out_path = "results/llm_rerank_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved: {out_path}")

    if delta_mrr > 0:
        print(f"\n✅ Day 4 完成！Milestone: BM25 → LLM Rerank ↑  ({bm25_mrr10:.4f} → {llm_mrr10:.4f})")
    else:
        print(f"\n⚠  Day 4 完成，但 MRR 未提升 ({bm25_mrr10:.4f} → {llm_mrr10:.4f})")
        print("   Tips: try more queries, a stronger model (gpt-4o), or larger NUM_RERANK.")


if __name__ == "__main__":
    main()
