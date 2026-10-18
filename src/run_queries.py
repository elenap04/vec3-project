
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.adapters.chroma_adapter import ChromaAdapter
from src.adapters.pgvector_adapter import PgvectorAdapter


DATASET = Path("datasets/synthetic_10k_d128.npz")
K = 10
SELECTIVITIES = (100, 50, 10, 1)


def recall_at_k(returned, exact):
    return len(set(returned) & set(exact)) / K


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        required=True,
        choices=["chroma", "pgvector"],
    )

    parser.add_argument(
        "--ef-search",
        type=int,
        default=80,
    )

    args = parser.parse_args()

    with np.load(DATASET) as data:
        queries = data["queries"].copy()
        ground_truths = {
            s: data[f"gt_{s}"].copy()
            for s in SELECTIVITIES
        }

    if args.db == "chroma":
        adapter = ChromaAdapter()
    else:
        adapter = PgvectorAdapter(
            ef_search=args.ef_search
        )

    results = {
        "mode": "development_pilot",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "database": args.db,
        "dataset": DATASET.name,
        "queries": len(queries),
        "k": K,
        "ef_search": (
            args.ef_search
            if args.db == "pgvector"
            else None
        ),
        "filters": {},
    }

    try:
        for selectivity in SELECTIVITIES:
            recalls = []
            short_results = 0

            for i, query in enumerate(queries):
                ids = adapter.search(
                    query,
                    k=K,
                    selectivity=selectivity,
                )

                if len(ids) != len(set(ids)):
                    raise ValueError(
                        "Duplicate IDs in search results"
                    )

                if len(ids) < K:
                    short_results += 1

                exact = ground_truths[selectivity][i]

                recalls.append(
                    recall_at_k(ids, exact)
                )

            mean_recall = float(np.mean(recalls))

            results["filters"][str(selectivity)] = {
                "mean_recall_at_10": mean_recall,
                "min_recall_at_10": float(np.min(recalls)),
                "max_recall_at_10": float(np.max(recalls)),
                "queries_with_less_than_k": short_results,
            }

            print(
                f"{args.db} | filter={selectivity}% "
                f"| recall@10={mean_recall:.4f} "
                f"| short={short_results}"
            )

    finally:
        if args.db == "pgvector":
            adapter.close()

    output_dir = Path("results/pilot")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (
        f"recall_{args.db}.json"
    )

    output_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_file}")
    print("QUERY RUNNER COMPLETED")


if __name__ == "__main__":
    main()

