"""
Day 6 (v1): LLM verifier layered on top of v0 evidence retrieval.
Day 6 v1：在 v0 证据检索基础上叠加 LLM 验证器。

Reads the JSON produced by ``evidence_retrieval.py`` and asks an LLM, batched
per (query, candidate), whether each evidence sentence actually supports its
attribute. Outputs the same JSON shape with ``verified`` and ``confidence``
fields added; evidence below ``--min-confidence`` is dropped from the final
``evidence`` map (the raw verifier output is kept under ``_verifier_raw``).
读取 v0 输出，按 (query, candidate) 批量请求 LLM 判断每条证据是否真正支持其
属性，并加上置信度。低于阈值的证据被剔除。

Why a separate stage:
- v0 → v1 → Day 7 is a clean ablation chain.
- v0 can be tuned (alpha / min_score) without paying LLM cost each time.
- v1's verifier delta is measurable in isolation.
分离动机：v0 / v1 各自可独立调参，方便消融与成本控制。

Usage / 用法:
    export OPENAI_API_KEY=sk-...
    python evidence_verify.py
    python evidence_verify.py --min-confidence 4 --num-queries 30
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

import config


# ── Prompt ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are evaluating whether short evidence sentences support attributes "
    "of a search query. For each numbered (attribute, evidence) pair, return "
    "a strict JSON object whose keys are the pair numbers (as strings) and "
    "whose values are objects with two keys: \"supports\" (boolean) and "
    "\"confidence\" (integer 1-5). Do not include any other text."
)


def build_verifier_prompt(query: str, items: List[Dict]) -> str:
    """``items`` is a list of {'attribute': str, 'evidence': str}."""
    lines = [f'Search query: "{query}"', "", "Pairs:"]
    for i, it in enumerate(items, start=1):
        lines.append(f'[{i}] attribute: "{it["attribute"]}"')
        lines.append(f'    evidence:  "{it["evidence"]}"')
    lines += [
        "",
        'Return JSON like: {"1": {"supports": true, "confidence": 5}, '
        '"2": {"supports": false, "confidence": 4}}',
        "confidence: 1 (very unsure) to 5 (very confident).",
    ]
    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────

def verify_candidate(
    client: OpenAI,
    query: str,
    items: List[Dict],
    model: str,
    max_retries: int = 2,
) -> Optional[Dict[int, Dict]]:
    """Return ``{1-indexed_position: {'supports': bool, 'confidence': int}}``."""
    if not items:
        return {}

    prompt = build_verifier_prompt(query, items)
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.EVIDENCE_VERIFY_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            out: Dict[int, Dict] = {}
            for k, v in parsed.items():
                try:
                    idx = int(str(k).strip().lstrip("[").rstrip("]"))
                except ValueError:
                    continue
                if not isinstance(v, dict):
                    continue
                supports = bool(v.get("supports", False))
                conf_raw = v.get("confidence", 0)
                try:
                    conf = int(conf_raw)
                except (TypeError, ValueError):
                    conf = 0
                conf = max(1, min(5, conf)) if supports else max(1, min(5, conf or 1))
                out[idx] = {"supports": supports, "confidence": conf}
            return out
        except json.JSONDecodeError:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception as e:  # noqa: BLE001
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"\n  Verifier failed: {e}")
                return None


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 6 v1: LLM verifier on top of v0 evidence.")
    p.add_argument("--input", type=str, default=str(config.EVIDENCE_RETRIEVAL_RESULTS),
                   help="JSON produced by evidence_retrieval.py.")
    p.add_argument("--output", type=str, default=str(config.EVIDENCE_VERIFIED_RESULTS))
    p.add_argument("--model", type=str, default=config.LLM_MODEL)
    p.add_argument("--num-queries", type=int, default=None,
                   help="Cap the number of queries verified (default: all).")
    p.add_argument("--min-confidence", type=int,
                   default=config.EVIDENCE_VERIFY_MIN_CONFIDENCE,
                   help="Drop evidence whose verifier confidence is below this.")
    p.add_argument("--sleep", type=float, default=config.LLM_SLEEP_BETWEEN)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.seed_everything(args.seed)

    if not os.path.exists(args.input):
        raise SystemExit(
            f"{args.input} not found. Run evidence_retrieval.py first."
        )
    with open(args.input) as f:
        v0 = json.load(f)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. See .env.example.")
    client = OpenAI(api_key=api_key)

    queries = v0["results"]
    if args.num_queries:
        queries = queries[: args.num_queries]

    # Count total candidates needing a call (those with ≥1 evidence)
    total_calls = sum(1 for q in queries for c in q["candidates"] if c["evidence"])
    print(f"Verifying {total_calls} (query, candidate) bundles with {args.model}.")

    n_supported = 0
    n_evaluated = 0
    n_dropped = 0
    failed = 0

    pbar = tqdm(total=total_calls, desc="LLM verify")
    for q in queries:
        for cand in q["candidates"]:
            ev_map: Dict[str, Dict] = cand["evidence"]
            if not ev_map:
                continue
            attrs = list(ev_map.keys())
            items = [{"attribute": a, "evidence": ev_map[a]["evidence"]} for a in attrs]

            verdicts = verify_candidate(client, q["query"], items, model=args.model)
            if verdicts is None:
                failed += 1
                cand["_verifier_raw"] = None
                pbar.update(1)
                time.sleep(args.sleep)
                continue

            kept: Dict[str, Dict] = {}
            raw_block: Dict[str, Dict] = {}
            for i, attr in enumerate(attrs, start=1):
                v = verdicts.get(i, {"supports": False, "confidence": 1})
                raw_block[attr] = v
                if v["supports"] and v["confidence"] >= args.min_confidence:
                    entry = dict(ev_map[attr])
                    entry["verifier_confidence"] = v["confidence"]
                    kept[attr] = entry
                    n_supported += 1
                else:
                    n_dropped += 1
                n_evaluated += 1

            cand["evidence"] = kept
            cand["_verifier_raw"] = raw_block

            pbar.update(1)
            time.sleep(args.sleep)
    pbar.close()

    # Recompute coverage on the verified set
    cov_num, cov_den = 0, 0
    for q in queries:
        groundable = sum(
            len(q["attributes"].get(f, []))
            for f in config.EVIDENCE_GROUNDED_FIELDS
        )
        if not groundable:
            continue
        for c in q["candidates"]:
            cov_num += len(c["evidence"])
            cov_den += groundable
    avg_cov = cov_num / cov_den if cov_den else 0.0

    print("\nVerifier stats / 验证器统计:")
    print(f"  Bundles verified         : {total_calls - failed}/{total_calls}")
    print(f"  Evidence evaluated       : {n_evaluated}")
    print(f"  Kept (supports & conf≥{args.min_confidence}) : {n_supported}")
    print(f"  Dropped                  : {n_dropped}")
    print(f"  Coverage after verifier  : {avg_cov:.3f}  "
          f"(v0 was {v0.get('avg_attribute_coverage', 'n/a')})")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "method": "v1_lexical_emb_plus_llm_verifier",
        "embedding_model": v0.get("embedding_model"),
        "verifier_model": args.model,
        "num_queries": len(queries),
        "avg_attribute_coverage": round(avg_cov, 4),
        "v0_avg_attribute_coverage": v0.get("avg_attribute_coverage"),
        "verifier": {
            "evaluated": n_evaluated,
            "kept": n_supported,
            "dropped": n_dropped,
            "failed_bundles": failed,
            "min_confidence": args.min_confidence,
        },
        "config": {
            "v0_input": args.input,
            "min_confidence": args.min_confidence,
            "sleep_between": args.sleep,
            "max_tokens": config.EVIDENCE_VERIFY_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            "seed": args.seed,
        },
        "results": queries,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
