"""
Build an attribute-bearing query subset for the Path-C ablation.

MS MARCO is dominated by factual queries with empty attributes, so the
evidence-grounding method has nothing to act on for ~92% of queries. This
script selects only the queries whose decomposition has non-empty
``important_attributes`` or ``constraints`` and writes filtered copies of
the attribute + BM25-candidate files, so the rest of the pipeline can run
on the subset where the method is actually expected to help.

Usage:
    python make_attr_subset.py
    python make_attr_subset.py --min-queries 5
"""

import argparse
import json

import config


def has_attributes(attrs: dict) -> bool:
    return bool(attrs.get("important_attributes") or attrs.get("constraints"))


def main() -> None:
    p = argparse.ArgumentParser(description="Filter to attribute-bearing queries.")
    p.add_argument("--attributes", type=str, default=str(config.QUERY_ATTRIBUTES_RESULTS))
    p.add_argument("--candidates", type=str, default=str(config.BM25_CANDIDATES))
    p.add_argument("--attr-out", type=str,
                   default=str(config.RESULTS_DIR / "query_attributes_attr.json"))
    p.add_argument("--cand-out", type=str,
                   default=str(config.RESULTS_DIR / "bm25_candidates_attr.json"))
    args = p.parse_args()

    with open(args.attributes) as f:
        attr_data = json.load(f)
    with open(args.candidates) as f:
        cand_data = json.load(f)

    attr_results = [r for r in attr_data["results"] if has_attributes(r["attributes"])]
    keep_ids = {r["query_id"] for r in attr_results}
    cand_queries = [q for q in cand_data["queries"] if q["query_id"] in keep_ids]

    print(f"Attribute-bearing queries: {len(keep_ids)} / {len(attr_data['results'])}")
    print(f"Matched in candidates:     {len(cand_queries)}")

    attr_out = dict(attr_data)
    attr_out["results"] = attr_results
    attr_out["num_queries"] = len(attr_results)
    with open(args.attr_out, "w") as f:
        json.dump(attr_out, f, indent=2, ensure_ascii=False)

    cand_out = dict(cand_data)
    cand_out["queries"] = cand_queries
    cand_out["num_queries"] = len(cand_queries)
    with open(args.cand_out, "w") as f:
        json.dump(cand_out, f, indent=2, ensure_ascii=False)

    print(f"Saved: {args.attr_out}")
    print(f"Saved: {args.cand_out}")
    print("\nQuery list:")
    for r in attr_results:
        a = r["attributes"]
        print(f"  [{r['query_id']:>3}] {r['query'][:50]:50} "
              f"attrs={a.get('important_attributes', [])} "
              f"constraints={a.get('constraints', [])}")


if __name__ == "__main__":
    main()
