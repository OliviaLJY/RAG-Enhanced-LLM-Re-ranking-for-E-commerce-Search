"""
Paired significance test on per-query MRR deltas between two result files.
配对显著性检验：在两份结果 JSON 之间，对每条 query 的 MRR 差值做配对自助法。

For IR ablations the standard test is paired bootstrap: take per-query MRR
differences (method − baseline), resample with replacement B times, and report
the 95% CI on the mean delta plus a two-sided p-value. We also report the
paired t-statistic for readers expecting the parametric flavor.
IR 消融实验的标准做法：取每条 query 的 MRR 差值，重采样 B 次，给出 95% 置信区间和
双侧 p 值，并附上配对 t 统计量。

Usage / 用法:
    # Cross-encoder vs BM25 baseline
    python significance.py \\
        --baseline results/bm25_top20_candidates.json \\
        --method   results/cross_encoder_rerank_results.json \\
        --method-key cross_encoder_reranked_doc_ids

    # LLM rerank vs BM25 (uses the default --method-key)
    python significance.py \\
        --baseline results/bm25_top20_candidates.json \\
        --method   results/llm_rerank_results.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

import config


# ── Per-query metric helpers ──────────────────────────────────────

def _mrr(retrieved: List[int], relevant: Set[int]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _order(q: Dict, rerank_key: str) -> List[int]:
    """Return the doc-id ordering: rerank_key if present, else BM25 candidate order."""
    if rerank_key and rerank_key in q:
        return q[rerank_key]
    return [c["doc_id"] for c in q["candidates"]]


def load_queries(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    return data["queries"] if "queries" in data else data


def per_query_mrr(queries: List[Dict], rerank_key: str, k: int) -> List[float]:
    out = []
    for q in queries:
        rel = set(q["relevant_doc_ids"])
        if not rel:
            continue
        out.append(_mrr(_order(q, rerank_key)[:k], rel))
    return out


def align_queries(
    baseline_queries: List[Dict],
    method_queries: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Pair queries by query_id; drop queries missing from either side or with no qrels."""
    b_by_id = {q["query_id"]: q for q in baseline_queries if q.get("relevant_doc_ids")}
    m_by_id = {q["query_id"]: q for q in method_queries if q.get("relevant_doc_ids")}
    common = sorted(set(b_by_id) & set(m_by_id))
    return [b_by_id[i] for i in common], [m_by_id[i] for i in common]


# ── Statistics ────────────────────────────────────────────────────

def paired_bootstrap(
    deltas: np.ndarray,
    iterations: int,
    ci_level: float,
    seed: int,
) -> Dict[str, float]:
    """
    Bootstrap the mean of a paired delta array.

    Returns observed mean, lower/upper CI bounds, and a two-sided p-value.
    The p-value is the fraction of resamples whose mean delta has the opposite
    sign of the observed mean (×2, clipped at 1.0); the standard IR convention.
    """
    n = len(deltas)
    if n == 0:
        raise SystemExit("No paired queries to bootstrap on.")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iterations, n))
    resample_means = deltas[idx].mean(axis=1)

    observed = float(deltas.mean())
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(resample_means, alpha))
    hi = float(np.quantile(resample_means, 1.0 - alpha))

    if observed > 0:
        p = 2.0 * float((resample_means <= 0).mean())
    elif observed < 0:
        p = 2.0 * float((resample_means >= 0).mean())
    else:
        p = 1.0
    p = min(p, 1.0)

    return {
        "mean_delta": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
        "iterations": iterations,
        "ci_level": ci_level,
    }


def paired_t_stat(deltas: np.ndarray) -> Dict[str, float]:
    """Paired t-statistic + Welch's df. p-value derived from a normal-approx CDF (n ≥ 30)."""
    n = len(deltas)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        return {"t": 0.0, "df": n - 1, "p_value_normal_approx": 1.0}
    se = sd / math.sqrt(n)
    t = mean / se
    # Two-sided p via Φ — fine for n ≥ 30, which IR ablations usually satisfy.
    p = 2.0 * (1.0 - _standard_normal_cdf(abs(t)))
    return {"t": t, "df": n - 1, "p_value_normal_approx": p}


def _standard_normal_cdf(x: float) -> float:
    """Φ(x) via erf, stdlib only."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired bootstrap significance test on per-query MRR deltas."
    )
    p.add_argument("--baseline", required=True,
                   help="Baseline results JSON (e.g. bm25_top20_candidates.json).")
    p.add_argument("--method", required=True,
                   help="Method results JSON (e.g. cross_encoder_rerank_results.json).")
    p.add_argument("--baseline-key", default="",
                   help="Doc-id field in baseline JSON. Empty = use BM25 candidate order.")
    p.add_argument("--method-key", default="llm_reranked_doc_ids",
                   help="Doc-id field in method JSON. Use cross_encoder_reranked_doc_ids "
                        "for the cross-encoder file.")
    p.add_argument("--k", type=int, default=config.DEFAULT_K_METRIC,
                   help="K for MRR@K (default from config).")
    p.add_argument("--iterations", type=int, default=config.BOOTSTRAP_ITERATIONS)
    p.add_argument("--ci-level", type=float, default=config.BOOTSTRAP_CI_LEVEL)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--output", type=str, default="",
                   help="Optional JSON output path. Default = no file written.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.seed_everything(args.seed)

    baseline = load_queries(args.baseline)
    method = load_queries(args.method)
    b_aligned, m_aligned = align_queries(baseline, method)
    n = len(b_aligned)
    if n == 0:
        raise SystemExit(
            "No queries shared between baseline and method (matched by query_id)."
        )

    bm = np.array(per_query_mrr(b_aligned, args.baseline_key, args.k))
    mm = np.array(per_query_mrr(m_aligned, args.method_key, args.k))
    if len(bm) != len(mm):
        raise SystemExit(
            f"Per-query arrays disagree in length after alignment: "
            f"{len(bm)} vs {len(mm)}."
        )
    deltas = mm - bm

    boot = paired_bootstrap(deltas, args.iterations, args.ci_level, args.seed)
    t_stat = paired_t_stat(deltas)

    baseline_mean = float(bm.mean())
    method_mean = float(mm.mean())
    win = int((deltas > 0).sum())
    lose = int((deltas < 0).sum())
    tie = int((deltas == 0).sum())

    sig_marker = "***" if boot["p_value"] < 0.001 else (
        "**" if boot["p_value"] < 0.01 else (
            "*" if boot["p_value"] < 0.05 else "n.s."
        )
    )

    print(f"\nPaired bootstrap on MRR@{args.k}")
    print(f"  Baseline file     : {args.baseline}")
    print(f"  Method   file     : {args.method}")
    print(f"  Paired queries    : {n}")
    print(f"  Baseline MRR@{args.k:<3}: {baseline_mean:.4f}")
    print(f"  Method   MRR@{args.k:<3}: {method_mean:.4f}")
    print(f"  Mean delta        : {boot['mean_delta']:+.4f}  "
          f"[{int(boot['ci_level']*100)}% CI: {boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")
    print(f"  Bootstrap p       : {boot['p_value']:.4f}   {sig_marker}")
    print(f"  Paired t          : t={t_stat['t']:+.3f}, df={t_stat['df']}, "
          f"p≈{t_stat['p_value_normal_approx']:.4f} (normal approx)")
    print(f"  Win / Tie / Lose  : {win} / {tie} / {lose}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline_file": args.baseline,
            "method_file": args.method,
            "baseline_key": args.baseline_key or "<bm25_candidate_order>",
            "method_key": args.method_key,
            "k": args.k,
            "num_paired_queries": n,
            "baseline_mean_mrr": baseline_mean,
            "method_mean_mrr": method_mean,
            "bootstrap": boot,
            "paired_t": t_stat,
            "wins": win,
            "ties": tie,
            "losses": lose,
            "seed": args.seed,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
