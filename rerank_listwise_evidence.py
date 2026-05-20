"""
Day 7 (listwise): Attribute-grounded LLM reranker — RankGPT-style with evidence.
Day 7（列表式）：基于属性证据的 LLM 列表式重排。

Extends ``llm_rerank.py`` by injecting the per-attribute evidence (from
``evidence_retrieval.py`` v0 or ``evidence_verify.py`` v1) into each candidate
block in the prompt. Output is compatible with ``evaluate.py --use-llm-order``.
在原 RankGPT 列表式 prompt 中，为每个候选段落额外注入按属性提取的证据，输出
格式兼容 ``evaluate.py --use-llm-order``。

Pipeline / 流水线:
    BM25 → query_attributes.json → evidence_*.json → THIS → reranked order

Usage / 用法:
    export OPENAI_API_KEY=sk-...
    python rerank_listwise_evidence.py
    python rerank_listwise_evidence.py --evidence results/evidence_verified.json
    python rerank_listwise_evidence.py --num-rerank 30 --top-k 10
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Set

import numpy as np
from openai import OpenAI
from tqdm import tqdm

import config
from bm25_baseline import mrr


# ── Prompt construction & parsing ─────────────────────────────────

def _fmt_list(values: List[str], empty: str = "(none)") -> str:
    return ", ".join(values) if values else empty


def build_listwise_prompt(
    query: str,
    attributes: Dict,
    candidates: List[Dict],
    truncate: int = config.PASSAGE_TRUNCATE_CHARS,
) -> str:
    """
    candidates: list of dicts with keys ``passage`` and ``evidence``
                (a possibly-empty {attr: {"evidence": str, ...}} mapping).
    """
    n = len(candidates)
    header = (
        f"Search query: {query}\n"
        f"Query intent: {attributes.get('intent_type', 'other')}\n"
        f"Important attributes: {_fmt_list(attributes.get('important_attributes', []))}\n"
        f"Constraints: {_fmt_list(attributes.get('constraints', []))}\n"
        f"Soft preferences: {_fmt_list(attributes.get('soft_preferences', []))}\n\n"
        f"I will provide you with {n} candidate passages, each indicated by a "
        f"numerical identifier []. Some passages have extracted per-attribute "
        f"evidence snippets attached; treat them as hints when judging relevance, "
        f"but trust the passage text as the source of truth."
    )

    blocks: List[str] = []
    for i, c in enumerate(candidates, start=1):
        passage = c["passage"]
        if len(passage) > truncate:
            passage = passage[:truncate] + "..."
        ev = c.get("evidence", {}) or {}
        if ev:
            ev_lines = "\n".join(
                f'    - {attr}: "{info.get("evidence", "")}"'
                for attr, info in ev.items()
            )
            ev_block = f"\n  Evidence:\n{ev_lines}"
        else:
            ev_block = "\n  Evidence: (none found)"
        blocks.append(f"[{i}] Passage: {passage}{ev_block}")

    instructions = (
        f"\n\nRank the {n} passages above based on their relevance to the "
        f"search query, weighing the attribute evidence. List them in descending "
        f"order using identifiers. Output format: [] > [] > [], e.g., [2] > [1] > [3]. "
        f"Only respond with the ranking, no explanation."
    )

    return header + "\n\n" + "\n\n".join(blocks) + instructions


def parse_ranking(response: str, num_passages: int) -> List[int]:
    """Same as in llm_rerank.py — parse "[2] > [1] > [3]" to 0-indexed list."""
    numbers = re.findall(r"\[(\d+)\]", response)
    ranking: List[int] = []
    seen: Set[int] = set()
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

def rerank_listwise(
    client: OpenAI,
    query: str,
    attributes: Dict,
    candidates: List[Dict],
    model: str,
    max_retries: int = 2,
) -> Optional[List[int]]:
    prompt = build_listwise_prompt(query, attributes, candidates)
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
                time.sleep(2 ** attempt)
            else:
                print(f"\n  Listwise rerank failed: {e}")
                return None


# ── Data joining ──────────────────────────────────────────────────

def join_evidence_with_passages(
    evidence_data: Dict,
    candidates_data: Dict,
) -> List[Dict]:
    """
    Build per-query records with the fields needed for reranking + evaluation.

    Each record has:
        query_id, query, attributes, relevant_doc_ids,
        candidates: [{rank, doc_id, passage, is_relevant, evidence}]
    """
    # Index BM25 candidates: query_id -> {doc_id -> candidate_with_passage}
    cand_index: Dict[int, Dict[int, Dict]] = {}
    relevant_index: Dict[int, List[int]] = {}
    for q in candidates_data["queries"]:
        cand_index[q["query_id"]] = {c["doc_id"]: c for c in q["candidates"]}
        relevant_index[q["query_id"]] = q["relevant_doc_ids"]

    joined: List[Dict] = []
    for r in evidence_data["results"]:
        qid = r["query_id"]
        cand_map = cand_index.get(qid)
        if cand_map is None:
            continue
        cands: List[Dict] = []
        for c in r["candidates"]:
            base = cand_map.get(c["doc_id"])
            if base is None:
                continue
            cands.append({
                "rank": c["rank"],
                "doc_id": c["doc_id"],
                "passage": base["passage"],
                "is_relevant": c.get("is_relevant", base.get("is_relevant", False)),
                "evidence": c.get("evidence", {}),
            })
        joined.append({
            "query_id": qid,
            "query": r["query"],
            "attributes": r["attributes"],
            "relevant_doc_ids": relevant_index.get(qid, []),
            "candidates": cands,
        })
    return joined


# ── Metrics ───────────────────────────────────────────────────────

def _doc_ids(q: Dict, use_llm: bool) -> List[int]:
    if use_llm:
        return q.get("llm_reranked_doc_ids", [c["doc_id"] for c in q["candidates"]])
    return [c["doc_id"] for c in q["candidates"]]


def compute_mrr_at_k(queries: List[Dict], use_llm: bool, k: int) -> float:
    scores = [mrr(_doc_ids(q, use_llm)[:k], set(q["relevant_doc_ids"])) for q in queries]
    return float(np.mean(scores)) if scores else 0.0


def compute_recall_at_k(queries: List[Dict], use_llm: bool, k: int) -> float:
    scores = []
    for q in queries:
        rel = set(q["relevant_doc_ids"])
        if not rel:
            continue
        hits = sum(1 for d in _doc_ids(q, use_llm)[:k] if d in rel)
        scores.append(hits / len(rel))
    return float(np.mean(scores)) if scores else 0.0


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 7 listwise: attribute-grounded LLM rerank.")
    p.add_argument("--evidence", type=str, default=str(config.EVIDENCE_RETRIEVAL_RESULTS),
                   help="Evidence JSON (v0 or v1) to use.")
    p.add_argument("--candidates", type=str, default=str(config.BM25_CANDIDATES),
                   help="BM25 candidates JSON (for passage text).")
    p.add_argument("--output", type=str, default=str(config.LISTWISE_EVIDENCE_RESULTS))
    p.add_argument("--model", type=str, default=config.LLM_MODEL)
    p.add_argument("--num-rerank", type=int, default=config.NUM_RERANK)
    p.add_argument("--top-k", type=int, default=config.LISTWISE_TOP_K,
                   help="Candidates fed into the prompt (and capped from evidence).")
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
    print(f"Loaded {len(joined)} queries with evidence; reranking first {len(queries)}.")

    # Cap each query's candidates to args.top_k for the prompt
    for q in queries:
        q["candidates"] = q["candidates"][: args.top_k]

    k = config.DEFAULT_K_METRIC
    bm25_mrr10 = compute_mrr_at_k(queries, use_llm=False, k=k)
    bm25_recall5 = compute_recall_at_k(queries, use_llm=False, k=5)
    print(f"BM25 baseline on subset — MRR@{k}: {bm25_mrr10:.4f}, Recall@5: {bm25_recall5:.4f}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. See .env.example.")
    client = OpenAI(api_key=api_key)

    failed = 0
    for q in tqdm(queries, desc="Listwise rerank"):
        cands = q["candidates"]
        ranking = rerank_listwise(client, q["query"], q["attributes"], cands, args.model)
        original_doc_ids = [c["doc_id"] for c in cands]
        if ranking is not None:
            q["llm_reranked_doc_ids"] = [original_doc_ids[i] for i in ranking]
            q["rerank_success"] = True
        else:
            q["llm_reranked_doc_ids"] = original_doc_ids
            q["rerank_success"] = False
            failed += 1
        time.sleep(args.sleep)

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

    print("\nResults: BM25 → Listwise rerank with evidence")
    print(f"  Queries           : {len(queries)} ({failed} fell back to BM25)")
    print(f"  Evidence source   : {args.evidence}")
    print(f"  MRR@{k}            : {bm25_mrr10:.4f} → {llm_mrr10:.4f}  ({delta_mrr:+.4f} {arrow})")
    print(f"  Recall@5          : {bm25_recall5:.4f} → {llm_recall5:.4f}")
    print(f"  Relative MRR gain : {rel_gain:+.1f}%")
    print(f"  Improved / Tied / Degraded: "
          f"{sum(1 for d in deltas if d > 0)} / "
          f"{sum(1 for d in deltas if d == 0)} / "
          f"{sum(1 for d in deltas if d < 0)}")

    # Strip heavy passage text from the saved output (keep doc_ids/evidence only)
    # so the file stays readable. Pass --keep-passages to retain if you want.
    for q in queries:
        for c in q["candidates"]:
            c.pop("passage", None)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "method": "listwise_with_evidence",
        "model": args.model,
        "evidence_source": args.evidence,
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
            "top_k": args.top_k,
            "num_rerank": args.num_rerank,
            "sleep_between": args.sleep,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
            "passage_truncate_chars": config.PASSAGE_TRUNCATE_CHARS,
            "seed": args.seed,
        },
        "queries": queries,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
