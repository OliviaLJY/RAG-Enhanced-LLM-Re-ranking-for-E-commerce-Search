"""
Day 5: Query → Attribute decomposition.
Day 5：Query → 结构化属性分解。

Stage 2a of the pipeline. The LLM transforms a free-text query into a
structured intent representation. The downstream Evidence Retrieval (Day 6)
and Attribute-Grounded Re-ranking (Day 7) modules consume this JSON.
流水线第 2a 步：LLM 将自然语言 query 转为结构化意图表示，供后续证据检索与
属性证据驱动的重排序使用。

Schema / 模式:
    intent_type           — factual | product_search | how_to | comparison |
                            navigational | other
    core_concepts         — main subject/entity nouns (list[str])
    constraints           — hard filters: price, time, location (list[str])
    important_attributes  — attributes that drive relevance (list[str])
    soft_preferences      — nice-to-haves (list[str])

The schema generalizes the e-commerce shape from the README so that factual
MS MARCO queries (where there's nothing to filter on) also fit — those simply
yield empty constraint / attribute / preference lists.

Usage / 用法:
    export OPENAI_API_KEY=sk-...
    python query_attributes.py --num-queries 50
    python query_attributes.py --num-queries 200 --resume   # incremental
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
    "You are an expert at understanding search queries. "
    "Given a search query, output a JSON object that decomposes it into a "
    "structured intent. Always return valid JSON with exactly these keys: "
    "intent_type, core_concepts, constraints, important_attributes, "
    "soft_preferences. Return only the JSON object, no commentary."
)

_USER_TEMPLATE = """Decompose the following search query.

Schema:
- intent_type: one of "factual", "product_search", "how_to", "comparison", "navigational", "other"
- core_concepts: list of strings — the main subject/entity nouns (1-4 items)
- constraints: list of strings — hard filters such as price, time, location, audience (empty list if none)
- important_attributes: list of strings — attributes that determine relevance (empty list if not applicable, e.g. for factual queries)
- soft_preferences: list of strings — nice-to-have attributes (empty list if none)

Examples:

Query: best headphones for gym under $100
{{"intent_type": "product_search", "core_concepts": ["headphones"], "constraints": ["under $100", "for gym use"], "important_attributes": ["sweat resistance", "secure fit", "battery life", "durability"], "soft_preferences": ["noise cancellation", "lightweight"]}}

Query: what is the capital of france
{{"intent_type": "factual", "core_concepts": ["capital", "France"], "constraints": [], "important_attributes": [], "soft_preferences": []}}

Query: how to bake sourdough bread
{{"intent_type": "how_to", "core_concepts": ["sourdough bread", "baking"], "constraints": [], "important_attributes": ["step-by-step instructions", "ingredient list"], "soft_preferences": ["beginner-friendly"]}}

Query: {query}
"""


def build_user_prompt(query: str) -> str:
    return _USER_TEMPLATE.format(query=query)


# ── Schema validation ─────────────────────────────────────────────

_ALLOWED_INTENTS = {
    "factual", "product_search", "how_to",
    "comparison", "navigational", "other",
}


def _coerce_list_of_str(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_attributes(raw: Dict) -> Dict:
    """
    Validate / fill missing fields so downstream stages can rely on the shape.
    """
    intent = raw.get("intent_type", "other")
    if not isinstance(intent, str) or intent not in _ALLOWED_INTENTS:
        intent = "other"
    return {
        "intent_type": intent,
        "core_concepts": _coerce_list_of_str(raw.get("core_concepts")),
        "constraints": _coerce_list_of_str(raw.get("constraints")),
        "important_attributes": _coerce_list_of_str(raw.get("important_attributes")),
        "soft_preferences": _coerce_list_of_str(raw.get("soft_preferences")),
    }


# ── LLM call ──────────────────────────────────────────────────────

def decompose_query(
    client: OpenAI,
    query: str,
    model: str = config.LLM_MODEL,
    max_retries: int = 2,
) -> Optional[Dict]:
    """
    Decompose a single query into the attribute schema.
    Returns the normalized dict, or None on persistent failure.
    """
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(query)},
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.ATTRIBUTE_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content.strip()
            parsed = json.loads(raw_text)
            return normalize_attributes(parsed)
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"\n  JSON parse failed for query: {query[:60]}... ({e})")
                return None
        except Exception as e:  # noqa: BLE001
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"\n  Attempt {attempt + 1} failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  All retries failed for: {query[:60]}...")
                return None


# ── Resume support ────────────────────────────────────────────────

def load_existing(path: str) -> Dict[int, Dict]:
    """Return {query_id: result_dict} from a prior run, or {} if none."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        prev = json.load(f)
    return {r["query_id"]: r for r in prev.get("results", [])}


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 5: Query → Attribute decomposition.")
    p.add_argument("--candidates", type=str, default=str(config.BM25_CANDIDATES),
                   help="BM25 candidates JSON (provides the queries to decompose).")
    p.add_argument("--output", type=str, default=str(config.QUERY_ATTRIBUTES_RESULTS))
    p.add_argument("--model", type=str, default=config.LLM_MODEL)
    p.add_argument("--num-queries", type=int, default=config.NUM_QUERY_ATTRIBUTES,
                   help="Number of queries to decompose (budget cap).")
    p.add_argument("--resume", action="store_true",
                   help="Skip queries already present in the output file.")
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
        candidates_data = json.load(f)
    all_queries = candidates_data["queries"][: args.num_queries]
    print(f"Loaded {len(all_queries)} queries from {args.candidates}.")

    existing = load_existing(args.output) if args.resume else {}
    if existing:
        print(f"Resuming: {len(existing)} queries already decomposed.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. See .env.example.")
    client = OpenAI(api_key=api_key)

    results: List[Dict] = list(existing.values())
    failed = 0

    todo = [q for q in all_queries if q["query_id"] not in existing]
    print(f"Decomposing {len(todo)} new queries with {args.model}.")

    for q in tqdm(todo, desc="Query → Attributes"):
        attrs = decompose_query(client, q["query"], model=args.model)
        if attrs is None:
            failed += 1
            attrs = normalize_attributes({})   # safe default
        results.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "attributes": attrs,
        })
        time.sleep(args.sleep)

    # Sort by query_id for stable diffs
    results.sort(key=lambda r: r["query_id"])

    # ── Quick stats ───────────────────────────────────────────────
    intent_counts: Dict[str, int] = {}
    avg_attrs = 0.0
    for r in results:
        intent_counts[r["attributes"]["intent_type"]] = \
            intent_counts.get(r["attributes"]["intent_type"], 0) + 1
        avg_attrs += len(r["attributes"]["important_attributes"])
    avg_attrs = avg_attrs / len(results) if results else 0.0

    print("\nIntent distribution / 意图分布:")
    for intent, count in sorted(intent_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {intent:<18}: {count}")
    print(f"Avg important_attributes per query: {avg_attrs:.2f}")
    print(f"Failed decompositions: {failed}")

    # ── Save ──────────────────────────────────────────────────────
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model,
        "num_queries": len(results),
        "failed": failed,
        "intent_distribution": intent_counts,
        "avg_important_attributes_per_query": round(avg_attrs, 3),
        "schema_fields": list(config.ATTRIBUTE_SCHEMA_FIELDS),
        "config": {
            "num_queries_requested": args.num_queries,
            "sleep_between": args.sleep,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.ATTRIBUTE_MAX_TOKENS,
            "seed": args.seed,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}  ({len(results)} queries)")


if __name__ == "__main__":
    main()
