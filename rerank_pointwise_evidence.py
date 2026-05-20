"""
Day 7 (pointwise): Attribute-grounded LLM reranker with reasons.
Day 7（点式）：基于属性证据的 LLM 点式打分 + 解释。

For each (query, candidate) the LLM returns a JSON ``{"score": 1-5, "reason": str}``.
Candidates are sorted by score (BM25 rank breaks ties). The ``reason`` field
backs the paper's explainability story.
对每个 (query, 候选) 单独打分并给出一句理由；按分数排序（BM25 原始名次作为
并列时的次序）。``reason`` 字段支撑论文的可解释性叙事。

Cost note / 成本提示:
    50 queries × top-10 candidates = 500 calls.
    With gpt-4o-mini that's typically < $0.05.

Pipeline / 流水线:
    BM25 → query_attributes → evidence_*.json → THIS → reranked + reasons

Usage / 用法:
    export OPENAI_API_KEY=sk-...
    python rerank_pointwise_evidence.py
    python rerank_pointwise_evidence.py --evidence results/evidence_verified.json --top-k 10
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm

import config
from bm25_baseline import mrr
from rerank_listwise_evidence import (
    _fmt_list,
    compute_mrr_at_k,
    compute_recall_at_k,
    join_evidence_with_passages,
)


# ── Prompt ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are scoring a single search result for relevance to a query. "
    "Return strict JSON with two keys only: \"score\" (integer 1-5) and "
    "\"reason\" (a single concise sentence). "
    "1 = irrelevant, 3 = partially relevant, 5 = directly answers the query."
)


def build_pointwise_prompt(
    query: str,
    attributes: Dict,
    candidate: Dict,
    truncate: int = config.PASSAGE_TRUNCATE_CHARS,
) -> str:
    passage = candidate["passage"]
    if len(passage) > truncate:
        passage = passage[:truncate] + "..."

    ev = candidate.get("evidence", {}) or {}
    if ev:
        ev_lines = "\n".join(
            f'  - {attr}: "{info.get("evidence", "")}"'
            for attr, info in ev.items()
        )
        ev_block = f"Attribute evidence found in passage:\n{ev_lines}"
    else:
        ev_block = "Attribute evidence found in passage: (none)"

    return (
        f"Search query: {query}\n"
        f"Query intent: {attributes.get('intent_type', 'other')}\n"
        f"Important attributes: {_fmt_list(attributes.get('important_attributes', []))}\n"
        f"Constraints: {_fmt_list(attributes.get('constraints', []))}\n\n"
        f"Candidate passage:\n{passage}\n\n"
        f"{ev_block}\n\n"
        f"Score 1-5 how well the passage satisfies the query. "
        f"Reward passages that match the important attributes and constraints. "
        f"Penalize passages whose only match is keyword overlap without intent. "
        f"Return JSON: {{\"score\": <int 1-5>, \"reason\": \"<one sentence>\"}}."
    )


# ── LLM call ──────────────────────────────────────────────────────

def score_candidate(
    client: OpenAI,
    query: str,
    attributes: Dict,
    candidate: Dict,
    model: str,
    max_retries: int = 2,
) -> Optional[Dict]:
    """Return ``{"score": int, "reason": str}`` or None on failure."""
    prompt = build_pointwise_prompt(query, attributes, candidate)
    lo, hi = config.POINTWISE_SCORE_MIN, config.POINTWISE_SCORE_MAX

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.POINTWISE_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            score_raw = parsed.get("score")
            try:
                score = int(round(float(score_raw)))
            except (TypeError, ValueError):
                score = config.POINTWISE_FALLBACK_SCORE
            score = max(lo, min(hi, score))
            reason = str(parsed.get("reason", "")).strip()[:500]
            return {"score": score, "reason": reason}
        except json.JSONDecodeError:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception as e:  # noqa: BLE001
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"\n  Pointwise scoring failed: {e}")
                return None


# ── Sorting ───────────────────────────────────────────────────────

def sort_by_score(scored: List[Dict]) -> List[int]:
    """
    Return doc_ids sorted by descending score, with the original BM25 rank as
    tie-break (stable sort over ascending rank ⇒ lower rank wins on tie).
    """
    ordered = sorted(scored, key=lambda x: (-x["score"], x["rank"]))
    return [x["doc_id"] for x in ordered]


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 7 pointwise: per-candidate LLM scoring.")
    p.add_argument("--evidence", type=str, default=str(config.EVIDENCE_RETRIEVAL_RESULTS))
    p.add_argument("--candidates", type=str, default=str(config.BM25_CANDIDATES))
    p.add_argument("--output", type=str, default=str(config.POINTWISE_EVIDENCE_RESULTS))
    p.add_argument("--model", type=str, default=config.LLM_MODEL)
    p.add_argument("--num-rerank", type=int, default=config.NUM_RERANK)
    p.add_argument("--top-k", type=int, default=config.POINTWISE_TOP_K,
                   help="Candidates per query to score (each is a separate LLM call).")
    p.add_argument("--sleep", type=float, default=config.LLM_SLEEP_BETWEEN)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.seed_everything(args.seed)

    for path in (args.evidence, args.candidates):
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found — run the upstream stage first.")

    with open(args.evidence) as f:
        evidence_data = json.load(f)
    with open(args.candidates) as f:
        candidates_data = json.load(f)

    joined = join_evidence_with_passages(evidence_data, candidates_data)
    queries = joined[: args.num_rerank]
    print(f"Loaded {len(joined)} queries with evidence; scoring first {len(queries)} "
          f"× top-{args.top_k} candidates.")

    # Cap per-query candidates
    for q in queries:
        q["candidates"] = q["candidates"][: args.top_k]

    total_calls = sum(len(q["candidates"]) for q in queries)
    est_cost = total_calls * 250 / 1_000_000 * 0.15
    print(f"~{total_calls} LLM calls planned (~${est_cost:.3f} with gpt-4o-mini).\n")

    k = config.DEFAULT_K_METRIC
    bm25_mrr10 = compute_mrr_at_k(queries, use_llm=False, k=k)
    bm25_recall5 = compute_recall_at_k(queries, use_llm=False, k=5)
    print(f"BM25 baseline on subset — MRR@{k}: {bm25_mrr10:.4f}, Recall@5: {bm25_recall5:.4f}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. See .env.example.")
    client = OpenAI(api_key=api_key)

    failed_calls = 0
    pbar = tqdm(total=total_calls, desc="Pointwise scoring")
    for q in queries:
        scored: List[Dict] = []
        for c in q["candidates"]:
            verdict = score_candidate(client, q["query"], q["attributes"], c, args.model)
            if verdict is None:
                verdict = {
                    "score": config.POINTWISE_FALLBACK_SCORE,
                    "reason": "(LLM call failed; fell back to neutral score)",
                }
                failed_calls += 1
            scored.append({
                "rank": c["rank"],
                "doc_id": c["doc_id"],
                "is_relevant": c.get("is_relevant", False),
                "score": verdict["score"],
                "reason": verdict["reason"],
                "evidence_attrs": sorted((c.get("evidence") or {}).keys()),
            })
            pbar.update(1)
            time.sleep(args.sleep)

        q["scored_candidates"] = scored
        q["llm_reranked_doc_ids"] = sort_by_score(scored)
        q["rerank_success"] = failed_calls == 0  # per-query coarse flag
    pbar.close()

    llm_mrr10 = compute_mrr_at_k(queries, use_llm=True, k=k)
    llm_recall5 = compute_recall_at_k(queries, use_llm=True, k=5)

    deltas = [
        mrr(q["llm_reranked_doc_ids"][:k], set(q["relevant_doc_ids"]))
        - mrr([c["doc_id"] for c in q["candidates"]][:k], set(q["relevant_doc_ids"]))
        for q in queries
    ]
    delta_mrr = llm_mrr10 - bm25_mrr10
    rel_gain = (delta_mrr / bm25_mrr10 * 100.0) if bm25_mrr10 > 0 else 0.0
    arrow = "↑" if delta_mrr > 0 else ("↓" if delta_mrr < 0 else "→")

    # Score distribution sanity check
    flat_scores = [s["score"] for q in queries for s in q["scored_candidates"]]
    score_hist = {s: flat_scores.count(s) for s in range(
        config.POINTWISE_SCORE_MIN, config.POINTWISE_SCORE_MAX + 1
    )}

    print("\nResults: BM25 → Pointwise rerank with evidence")
    print(f"  Queries × top-K   : {len(queries)} × {args.top_k}")
    print(f"  Failed LLM calls  : {failed_calls}/{total_calls}")
    print(f"  Evidence source   : {args.evidence}")
    print(f"  MRR@{k}            : {bm25_mrr10:.4f} → {llm_mrr10:.4f}  ({delta_mrr:+.4f} {arrow})")
    print(f"  Recall@5          : {bm25_recall5:.4f} → {llm_recall5:.4f}")
    print(f"  Relative MRR gain : {rel_gain:+.1f}%")
    print(f"  Score histogram   : {score_hist}")
    print(f"  Improved / Tied / Degraded: "
          f"{sum(1 for d in deltas if d > 0)} / "
          f"{sum(1 for d in deltas if d == 0)} / "
          f"{sum(1 for d in deltas if d < 0)}")

    # Drop passage text to keep output readable
    for q in queries:
        for c in q["candidates"]:
            c.pop("passage", None)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "method": "pointwise_with_evidence",
        "model": args.model,
        "evidence_source": args.evidence,
        "num_queries": len(queries),
        "total_llm_calls": total_calls,
        "failed_llm_calls": failed_calls,
        "score_histogram": score_hist,
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
            "top_k": args.top_k,
            "num_rerank": args.num_rerank,
            "sleep_between": args.sleep,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.POINTWISE_MAX_TOKENS,
            "passage_truncate_chars": config.PASSAGE_TRUNCATE_CHARS,
            "score_range": [config.POINTWISE_SCORE_MIN, config.POINTWISE_SCORE_MAX],
            "seed": args.seed,
        },
        "queries": queries,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
