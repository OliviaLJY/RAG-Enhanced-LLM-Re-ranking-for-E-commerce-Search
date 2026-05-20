"""
Day 6 (v0): Evidence Retrieval — keyword + embedding similarity.
Day 6 v0：证据检索 —— 关键词重合 + 句向量余弦相似度。

For each (query, candidate-passage, attribute) triple, find the single
sentence in the passage that best supports the attribute. Pure lexical +
SBERT cosine; no LLM calls in this stage. Output JSON is consumed by
``evidence_verify.py`` (v1, optional) and Day 7's reranker.
对每个 (query, 候选段落, 属性) 三元组，从段落中挑选最支持该属性的句子。
完全不调用 LLM；输出可被 v1 验证器和 Day 7 重排序器消费。

This is the deliberately-simple ablation. The README's DS-suggestion section
flags pure keyword retrieval as too trivial for a top venue, hence v1
(``evidence_verify.py``) layers an LLM verifier on top.
这是有意设计的简单基线 / ablation。仅靠关键词重合不够强，所以 v1 在此基础上
叠加 LLM 验证器。

Usage / 用法:
    python evidence_retrieval.py
    python evidence_retrieval.py --num-queries 50 --top-k-candidates 10
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import config


# ── Sentence / token utilities ────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_TOKEN_RE = re.compile(r"\w+")


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter. Good enough for short MS MARCO passages."""
    chunks = _SENTENCE_SPLIT_RE.split(text.strip())
    return [c.strip() for c in chunks if c.strip()]


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def keyword_score(attribute: str, sentence: str) -> float:
    """Token overlap normalized by attribute length (Jaccard-on-attribute)."""
    attr_tokens = set(tokenize(attribute))
    if not attr_tokens:
        return 0.0
    sent_tokens = set(tokenize(sentence))
    overlap = attr_tokens & sent_tokens
    return len(overlap) / len(attr_tokens)


# ── Core scoring ──────────────────────────────────────────────────

def score_evidence(
    sent_embs: np.ndarray,           # (n_sents, d), normalized
    attr_emb: np.ndarray,            # (d,), normalized
    sentences: List[str],
    attribute: str,
    alpha: float,
) -> Tuple[int, float, float, float]:
    """
    Return ``(best_idx, combined, cosine, keyword)`` for the top-1 sentence.
    Embeddings are assumed L2-normalized so cosine == dot product.
    """
    cosine = sent_embs @ attr_emb
    kw = np.array([keyword_score(attribute, s) for s in sentences])
    combined = alpha * cosine + (1.0 - alpha) * kw
    best = int(np.argmax(combined))
    return best, float(combined[best]), float(cosine[best]), float(kw[best])


# ── Main pipeline ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 6 v0: lexical+embedding evidence retrieval.")
    p.add_argument("--candidates", type=str, default=str(config.BM25_CANDIDATES))
    p.add_argument("--attributes", type=str, default=str(config.QUERY_ATTRIBUTES_RESULTS))
    p.add_argument("--output", type=str, default=str(config.EVIDENCE_RETRIEVAL_RESULTS))
    p.add_argument("--model", type=str, default=config.SBERT_MODEL)
    p.add_argument("--num-queries", type=int, default=config.NUM_QUERY_ATTRIBUTES,
                   help="How many queries to ground (must be <= queries with attributes).")
    p.add_argument("--top-k-candidates", type=int, default=config.EVIDENCE_TOP_K_CANDIDATES,
                   help="Top-N BM25 candidates per query to extract evidence for.")
    p.add_argument("--alpha", type=float, default=config.EVIDENCE_ALPHA,
                   help="Weight on cosine vs. keyword score (1.0 = cosine only).")
    p.add_argument("--min-score", type=float, default=config.EVIDENCE_MIN_SCORE,
                   help="Drop evidence if combined score is below this.")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.seed_everything(args.seed)

    if not os.path.exists(args.attributes):
        raise SystemExit(
            f"{args.attributes} not found. Run query_attributes.py first."
        )
    if not os.path.exists(args.candidates):
        raise SystemExit(
            f"{args.candidates} not found. Run bm25_export_topk.py first."
        )

    with open(args.attributes) as f:
        attr_data = json.load(f)
    with open(args.candidates) as f:
        cand_data = json.load(f)

    attr_by_qid: Dict[int, Dict] = {r["query_id"]: r for r in attr_data["results"]}

    # Process queries that have both candidates and attributes
    queries_to_process: List[Dict] = [
        q for q in cand_data["queries"]
        if q["query_id"] in attr_by_qid
    ][: args.num_queries]
    print(f"Processing {len(queries_to_process)} queries  ×  top-{args.top_k_candidates} candidates.")

    # Lazy import — keeps the rest of the repo usable without torch
    from sentence_transformers import SentenceTransformer
    print(f"Loading {args.model} ...")
    model = SentenceTransformer(args.model)

    # ── Pass 1: collect every (query, candidate) sentence list ────
    flat_sentences: List[str] = []
    sentence_index: Dict[Tuple[int, int], Tuple[int, int, List[str]]] = {}
    for q in queries_to_process:
        for c in q["candidates"][: args.top_k_candidates]:
            sents = split_sentences(c["passage"])
            start = len(flat_sentences)
            flat_sentences.extend(sents)
            sentence_index[(q["query_id"], c["doc_id"])] = (start, len(flat_sentences), sents)

    print(f"Encoding {len(flat_sentences):,} sentences...")
    sent_embs = model.encode(
        flat_sentences,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # ── Pass 2: collect unique attribute strings ──────────────────
    unique_attrs: set = set()
    for q in queries_to_process:
        attrs = attr_by_qid[q["query_id"]]["attributes"]
        for field in config.EVIDENCE_GROUNDED_FIELDS:
            unique_attrs.update(attrs.get(field, []))
    attr_list = sorted(unique_attrs)
    print(f"Encoding {len(attr_list)} unique attribute strings...")
    attr_emb_matrix = (
        model.encode(attr_list, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        if attr_list else np.zeros((0, sent_embs.shape[1]), dtype=np.float32)
    )
    attr_emb_map = {a: emb for a, emb in zip(attr_list, attr_emb_matrix)}

    # ── Pass 3: per-(query, candidate, attribute) evidence ────────
    results: List[Dict] = []
    cov_numerators: List[int] = []
    cov_denominators: List[int] = []

    for q in tqdm(queries_to_process, desc="Grounding evidence"):
        attrs = attr_by_qid[q["query_id"]]["attributes"]
        attrs_to_ground: List[Tuple[str, str]] = []
        for field in config.EVIDENCE_GROUNDED_FIELDS:
            for a in attrs.get(field, []):
                attrs_to_ground.append((a, field))

        candidate_evidence: List[Dict] = []
        for c in q["candidates"][: args.top_k_candidates]:
            start, end, sents = sentence_index[(q["query_id"], c["doc_id"])]
            evidence: Dict[str, Dict] = {}
            if sents:
                sub_embs = sent_embs[start:end]
                for attr, field in attrs_to_ground:
                    attr_emb = attr_emb_map[attr]
                    best_idx, combined, cos, kw = score_evidence(
                        sub_embs, attr_emb, sents, attr, args.alpha
                    )
                    if combined >= args.min_score:
                        evidence[attr] = {
                            "field": field,
                            "evidence": sents[best_idx],
                            "score": round(combined, 4),
                            "cosine": round(cos, 4),
                            "keyword": round(kw, 4),
                        }

            candidate_evidence.append({
                "doc_id": c["doc_id"],
                "rank": c["rank"],
                "is_relevant": c["is_relevant"],
                "evidence": evidence,
            })

            if attrs_to_ground:
                cov_numerators.append(len(evidence))
                cov_denominators.append(len(attrs_to_ground))

        results.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "attributes": attrs,
            "candidates": candidate_evidence,
        })

    # ── Stats ─────────────────────────────────────────────────────
    avg_coverage = (sum(cov_numerators) / sum(cov_denominators)) if cov_denominators else 0.0
    queries_with_no_attrs = sum(
        1 for r in results
        if not any(r["attributes"].get(f) for f in config.EVIDENCE_GROUNDED_FIELDS)
    )

    print("\nEvidence Retrieval (v0) stats / 统计:")
    print(f"  Queries processed         : {len(results)}")
    print(f"  Queries with 0 groundable : {queries_with_no_attrs}  (factual etc.)")
    print(f"  Avg attribute coverage    : {avg_coverage:.3f}")
    print(f"  alpha                     : {args.alpha}")
    print(f"  min_score                 : {args.min_score}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "method": "v0_lexical_emb",
        "embedding_model": args.model,
        "num_queries": len(results),
        "queries_with_no_groundable_attrs": queries_with_no_attrs,
        "avg_attribute_coverage": round(avg_coverage, 4),
        "config": {
            "top_k_candidates": args.top_k_candidates,
            "alpha": args.alpha,
            "min_score": args.min_score,
            "grounded_fields": list(config.EVIDENCE_GROUNDED_FIELDS),
            "seed": args.seed,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
