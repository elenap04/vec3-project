
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
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


def create_adapter(args):
    if args.db == "chroma":
        return ChromaAdapter()

    return PgvectorAdapter(ef_search=args.ef_search)


def warmup_worker(adapter, queries, indices, selectivity, count):
    for j in range(count):
        i = indices[j % len(indices)]
        adapter.search(
            queries[i],
            k=K,
            selectivity=selectivity,
        )


def search_worker(adapter, queries, indices, selectivity):
    results = []

    for i in indices:
        start = perf_counter()

        ids = adapter.search(
            queries[i],
            k=K,
            selectivity=selectivity,
        )

        elapsed_ms = (perf_counter() - start) * 1000
        results.append((i, ids, elapsed_ms))

    return results


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
        "--clients",
        type=int,
        choices=[1, 8],
        default=1,
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

    if len(queries) < args.clients:
        parser.error("Not enough queries for the client count")

    # Each client gets a disjoint group of query indices.
    groups = [
        list(range(client_id, len(queries), args.clients))
        for client_id in range(args.clients)
    ]

    # Separate adapter / connection per client.
    adapters = [create_adapter(args) for _ in range(args.clients)]

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
        "clients": args.clients,
        "client_model": "closed_loop",
        "warmup_queries_per_client_per_filter": args.warmup_queries,
        "ef_search": (
            args.ef_search
            if args.db == "pgvector"
            else None
        ),
        "filters": {},
    }

    try:
        with ThreadPoolExecutor(
            max_workers=args.clients
        ) as executor:

            for selectivity in SELECTIVITIES:

                # Warmup is not timed.
                warmup_futures = [
                    executor.submit(
                        warmup_worker,
                        adapters[j],
                        queries,
                        groups[j],
                        selectivity,
                        args.warmup_queries,
                    )
                    for j in range(args.clients)
                ]

                for future in warmup_futures:
                    future.result()

                # The timed workload starts after warmup.
                workload_start = perf_counter()

                futures = [
                    executor.submit(
                        search_worker,
                        adapters[j],
                        queries,
                        groups[j],
                        selectivity,
                    )
                    for j in range(args.clients)
                ]

                batches = [
                    future.result()
                    for future in futures
                ]

                workload_elapsed = (
                    perf_counter() - workload_start
                )

                # All recall validation happens after timing.
                returned = [None] * len(queries)
                latencies_ms = []

                for batch in batches:
                    for i, ids, latency in batch:
                        if returned[i] is not None:
                            raise ValueError(
                                "Query executed more than once"
                            )

                        returned[i] = ids
                        latencies_ms.append(latency)

                if any(ids is None for ids in returned):
                    raise ValueError("Missing query results")

                recalls = []
                short_results = 0

                for i, ids in enumerate(returned):
                    if len(ids) != len(set(ids)):
                        raise ValueError("Duplicate IDs")

                    if len(ids) > K:
                        raise ValueError("More than K results")

                    if len(ids) < K:
                        short_results += 1

                    recalls.append(
                        recall_at_k(
                            ids,
                            ground_truths[selectivity][i],
                        )
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
                    f"{args.db} | clients={args.clients} "
                    f"| filter={selectivity}% "
                    f"| recall@10={mean_recall:.4f} "
                    f"| p50={p50:.2f} ms "
                    f"| p95={p95:.2f} ms "
                    f"| p99={p99:.2f} ms "
                    f"| QPS={qps:.2f} "
                    f"| short={short_results}"
                )

    finally:
        if args.db == "pgvector":
            for adapter in adapters:
                adapter.close()

    output_dir = Path("results/pilot")
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    setting = (
        f"ef{args.ef_search}"
        if args.db == "pgvector"
        else "default"
    )

    output_file = output_dir / (
        f"query_{args.db}_{setting}_"
        f"c{args.clients}_{stamp}.json"
    )

    output_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_file}")
    print("QUERY RUNNER COMPLETED")


if __name__ == "__main__":
    main()

