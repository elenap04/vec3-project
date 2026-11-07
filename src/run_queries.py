import argparse
import json

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from src.adapters.chroma_adapter import ChromaAdapter
from src.adapters.pgvector_adapter import PgvectorAdapter
from src.resource_monitor import ResourceMonitor


K = 10
SELECTIVITIES = (100, 50, 10, 1)

DATASETS = {
    "10k": {
        "path": Path("datasets/synthetic_10k_d128.npz"),
        "ground_truth_path": None,
        "collection": "vec3_pilot_10k",
        "table": "vec3_pilot_10k",
        "expected_vectors": 10000,
        "dimensions": 128,
    },
    "100k": {
        "path": Path("datasets/synthetic_100k_d128.npz"),
        "ground_truth_path": None,
        "collection": "vec3_bench_100k_d128",
        "table": "vec3_bench_100k_d128",
        "expected_vectors": 100000,
        "dimensions": 128,
    },
    "1m": {
        "path": Path("datasets/synthetic_1m_d128.npz"),
        "ground_truth_path": None,
        "collection": "vec3_bench_1m_d128",
        "table": "vec3_bench_1m_d128",
        "expected_vectors": 1000000,
        "dimensions": 128,
        "expected_queries": 200,
    },
    "agnews100k": {
        "path": Path("datasets/agnews_100k_d384.npz"),
        "ground_truth_path": Path(
            "datasets/agnews_100k_d384_ground_truth.npz"
        ),
        "collection": "vec3_agnews_100k_d384",
        "table": "vec3_agnews_100k_d384",
        "expected_vectors": 100000,
        "dimensions": 384,
        "expected_queries": 200,
    },
}


def recall_at_k(returned, exact):
    return len(set(returned) & set(exact)) / K


def create_adapter(args, dataset_config):
    if args.db == "chroma":
        return ChromaAdapter(
            collection_name=dataset_config["collection"]
        )

    return PgvectorAdapter(
        table_name=dataset_config["table"],
        ef_search=args.ef_search,
    )


def warmup_worker(
    adapter,
    queries,
    indices,
    selectivity,
    count,
):
    for j in range(count):
        i = indices[j % len(indices)]

        adapter.search(
            queries[i],
            k=K,
            selectivity=selectivity,
        )


def search_worker(
    adapter,
    queries,
    indices,
    selectivity,
):
    results = []

    for i in indices:
        started = perf_counter()

        ids = adapter.search(
            queries[i],
            k=K,
            selectivity=selectivity,
        )

        latency_ms = (
            perf_counter() - started
        ) * 1000

        results.append((i, ids, latency_ms))

    return results


def read_ground_truth(data):
    return {
        selectivity: data[f"gt_{selectivity}"].copy()
        for selectivity in SELECTIVITIES
    }


def load_dataset(dataset_config):
    dataset_path = dataset_config["path"]
    ground_truth_path = dataset_config["ground_truth_path"]

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset file does not exist: {dataset_path}"
        )

    with np.load(
        dataset_path,
        allow_pickle=False,
    ) as data:
        queries = data["queries"].copy()
        vectors_shape = data["vectors"].shape

        source_metadata = (
            str(data["metadata_json"])
            if "metadata_json" in data.files
            else None
        )

        if ground_truth_path is None:
            ground_truths = read_ground_truth(data)

    expected_shape = (
        dataset_config["expected_vectors"],
        dataset_config["dimensions"],
    )

    if vectors_shape != expected_shape:
        raise ValueError(
            f"Unexpected dataset shape: "
            f"{vectors_shape}, expected {expected_shape}"
        )

    if (
        queries.ndim != 2
        or queries.shape[1] != expected_shape[1]
    ):
        raise ValueError(
            f"Unexpected queries shape: {queries.shape}"
        )

    if queries.dtype != np.float32:
        raise ValueError(
            f"Queries must be float32, got {queries.dtype}"
        )

    if not np.isfinite(queries).all():
        raise ValueError("Non-finite query vectors")

    expected_queries = dataset_config.get("expected_queries")

    if (
        expected_queries is not None
        and len(queries) != expected_queries
    ):
        raise ValueError(
            f"Expected {expected_queries} queries, "
            f"found {len(queries)}"
        )

    if ground_truth_path is not None:
        if not ground_truth_path.is_file():
            raise FileNotFoundError(
                "Ground truth is not ready: "
                f"{ground_truth_path}\n"
                "Generate exact ground truth before "
                "running query benchmarks."
            )

        with np.load(
            ground_truth_path,
            allow_pickle=False,
        ) as gt_data:
            gt_metric = str(gt_data["metric"])
            gt_k = int(gt_data["k"])

            if gt_metric.lower() != "l2":
                raise ValueError(
                    f"Unexpected ground truth metric: {gt_metric}"
                )

            if gt_k != K:
                raise ValueError(
                    f"Ground truth K={gt_k}, expected {K}"
                )

            gt_metadata = str(gt_data["metadata_json"])

            if gt_metadata != source_metadata:
                raise ValueError(
                    "Ground truth metadata does not match "
                    "the source dataset"
                )

            ground_truths = read_ground_truth(gt_data)

    for selectivity, ground_truth in ground_truths.items():
        if ground_truth.shape != (len(queries), K):
            raise ValueError(
                f"Incorrect ground truth shape "
                f"for {selectivity}%: {ground_truth.shape}"
            )

        for row in ground_truth:
            if len(set(row.tolist())) != K:
                raise ValueError(
                    f"Duplicate ground truth IDs "
                    f"for filter {selectivity}%"
                )

    return queries, ground_truths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Development query benchmark for "
            "Chroma and PostgreSQL/pgvector"
        )
    )

    parser.add_argument(
        "--db",
        required=True,
        choices=["chroma", "pgvector"],
    )

    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="10k",
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

    dataset_config = DATASETS[args.dataset]

    queries, ground_truths = load_dataset(
        dataset_config
    )

    if len(queries) < args.clients:
        parser.error("Not enough queries for client count")

    groups = [
        list(range(j, len(queries), args.clients))
        for j in range(args.clients)
    ]

    target_name = (
        dataset_config["collection"]
        if args.db == "chroma"
        else dataset_config["table"]
    )

    container_name = (
        "vec3-chroma"
        if args.db == "chroma"
        else "vec3-postgres"
    )

    print("\n=== QUERY RUN CONFIGURATION ===")
    print(f"Database: {args.db}")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset file: {dataset_config['path']}")
    print(
        "Ground truth file: "
        f"{dataset_config['ground_truth_path'] or dataset_config['path']}"
    )
    print(f"Target: {target_name}")
    print(
        f"Base vectors: "
        f"{dataset_config['expected_vectors']}"
    )
    print(f"Dimensions: {dataset_config['dimensions']}")
    print(f"Queries: {len(queries)}")
    print(f"Clients: {args.clients}")
    print(f"Warmup per client: {args.warmup_queries}")

    if args.db == "pgvector":
        print(f"ef_search: {args.ef_search}")

    results = {
        "mode": "development_pilot",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "database": args.db,
        "dataset_key": args.dataset,
        "dataset": dataset_config["path"].name,
        "ground_truth_file": (
            dataset_config["ground_truth_path"].name
            if dataset_config["ground_truth_path"]
            else dataset_config["path"].name
        ),
        "target": target_name,
        "base_vectors": dataset_config["expected_vectors"],
        "dimensions": dataset_config["dimensions"],
        "queries": len(queries),
        "k": K,
        "metric": "L2",
        "clients": args.clients,
        "client_model": "closed_loop",
        "warmup_queries_per_client_per_filter": (
            args.warmup_queries
        ),
        "ef_search": (
            args.ef_search
            if args.db == "pgvector"
            else None
        ),
        "filters": {},
    }

    adapters = []
    monitor = ResourceMonitor(container_name)
    monitor_started = False
    resource_summary = None

    try:
        adapters = [
            create_adapter(args, dataset_config)
            for _ in range(args.clients)
        ]

        monitor.start()
        monitor_started = True

        with ThreadPoolExecutor(
            max_workers=args.clients
        ) as executor:
            for selectivity in SELECTIVITIES:
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

                workload_started = perf_counter()

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
                    perf_counter() - workload_started
                )

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
                        raise ValueError(
                            "Duplicate IDs in search results"
                        )

                    if len(ids) > K:
                        raise ValueError(
                            "More than K results returned"
                        )

                    if len(ids) < K:
                        short_results += 1

                    recalls.append(
                        recall_at_k(
                            ids,
                            ground_truths[selectivity][i],
                        )
                    )

                mean_recall = float(np.mean(recalls))
                p50 = float(
                    np.percentile(latencies_ms, 50)
                )
                p95 = float(
                    np.percentile(latencies_ms, 95)
                )
                p99 = float(
                    np.percentile(latencies_ms, 99)
                )
                qps = len(queries) / workload_elapsed

                results["filters"][str(selectivity)] = {
                    "mean_recall_at_10": mean_recall,
                    "min_recall_at_10": float(
                        np.min(recalls)
                    ),
                    "max_recall_at_10": float(
                        np.max(recalls)
                    ),
                    "queries_with_less_than_k": short_results,
                    "latency_p50_ms": p50,
                    "latency_p95_ms": p95,
                    "latency_p99_ms": p99,
                    "qps": float(qps),
                    "measured_duration_seconds": (
                        workload_elapsed
                    ),
                }

                print(
                    f"{args.db} "
                    f"| dataset={args.dataset} "
                    f"| clients={args.clients} "
                    f"| filter={selectivity}% "
                    f"| recall@10={mean_recall:.4f} "
                    f"| p50={p50:.2f} ms "
                    f"| p95={p95:.2f} ms "
                    f"| p99={p99:.2f} ms "
                    f"| QPS={qps:.2f} "
                    f"| short={short_results}",
                    flush=True,
                )

    finally:
        try:
            if monitor_started:
                resource_summary = monitor.stop()
        finally:
            for adapter in adapters:
                if args.db == "pgvector":
                    adapter.close()

    results["resources"] = resource_summary
    results["resources_scope"] = (
        "entire_run_including_warmup"
    )

    print("\n=== RESOURCE MONITOR ===")
    print(
        f"Samples: {resource_summary['sample_count']}"
    )
    print(
        f"Status: {resource_summary['status']}"
    )
    print(
        f"Peak RAM: {resource_summary['peak_memory_mib']}"
    )
    print(
        f"Mean CPU: {resource_summary['mean_cpu_percent']}"
    )

    output_dir = Path("results/pilot")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    setting = (
        f"ef{args.ef_search}"
        if args.db == "pgvector"
        else "default"
    )

    output_file = output_dir / (
        f"query_{args.db}_"
        f"{args.dataset}_"
        f"{setting}_"
        f"c{args.clients}_"
        f"{stamp}.json"
    )

    output_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_file}")
    print("QUERY RUNNER COMPLETED")


if __name__ == "__main__":
    main()
