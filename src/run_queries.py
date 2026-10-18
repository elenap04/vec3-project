
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

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

    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if args.ef_search < 1:
        parser.error("--ef-search must be positive")

    if args.warmup_queries < 0:
        parser.error("--warmup-queries cannot be negative")

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
        "metric": "L2",
        "clients": 1,
        "client_model": "closed_loop",
        "warmup_queries_per_filter": args.warmup_queries,
        "ef_search": (
            args.ef_search
            if args.db == "pgvector"
            else None
        ),
        "filters": {},
    }

    try:
        for selectivity in SELECTIVITIES:

            latencies_ms = []
            returned_batches = []

            # Warmup: excluded from measurements
            for i in range(args.warmup_queries):
                query = queries[i % len(queries)]

                adapter.search(
                    query,
                    k=K,
                    selectivity=selectivity,
                )

            # Timed search workload
            workload_start = perf_counter()

            for query in queries:
                request_start = perf_counter()

                ids = adapter.search(
                    query,
                    k=K,
                    selectivity=selectivity,
                )

                request_end = perf_counter()

                latencies_ms.append(
                    (request_end - request_start) * 1000
                )

                returned_batches.append(ids)

            workload_elapsed = (
                perf_counter() - workload_start
            )

            # Validation happens AFTER timing
            recalls = []
            short_results = 0

            for i, ids in enumerate(returned_batches):
                if len(ids) != len(set(ids)):
                    raise ValueError(
                        "Duplicate IDs in search results"
                    )

                if len(ids) > K:
                    raise ValueError(
                        "More than K results returned"
                    )

                if len(ids) < K:
                    short_results += 1

                exact = ground_truths[selectivity][i]

                recalls.append(
                    recall_at_k(ids, exact)
                )

            mean_recall = float(np.mean(recalls))
            p50 = float(np.percentile(latencies_ms, 50))
            p95 = float(np.percentile(latencies_ms, 95))
            p99 = float(np.percentile(latencies_ms, 99))

            qps = len(queries) / workload_elapsed

            results["filters"][str(selectivity)] = {
                "mean_recall_at_10": mean_recall,
                "min_recall_at_10": float(np.min(recalls)),
                "max_recall_at_10": float(np.max(recalls)),
                "queries_with_less_than_k": short_results,
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "latency_p99_ms": p99,
                "qps": float(qps),
                "measured_duration_seconds": workload_elapsed,
            }

            print(
                f"{args.db} | filter={selectivity}% "
                f"| recall@10={mean_recall:.4f} "
                f"| p50={p50:.2f} ms "
                f"| p95={p95:.2f} ms "
                f"| p99={p99:.2f} ms "
                f"| QPS={qps:.2f} "
                f"| short={short_results}"
            )

    finally:
        if args.db == "pgvector":
            adapter.close()

    output_dir = Path("results/pilot")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    if args.db == "pgvector":
        setting = f"ef{args.ef_search}"
    else:
        setting = "default"

    output_file = output_dir / (
        f"query_{args.db}_{setting}_{run_id}.json"
    )

    output_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_file}")
    print("QUERY RUNNER COMPLETED")


if __name__ == "__main__":
    main()

