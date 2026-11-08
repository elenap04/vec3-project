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


def create_adapter(args, config):
    if args.db == "chroma":
        return ChromaAdapter(
            collection_name=config["collection"],
            candidate_count=args.chroma_candidates,
        )

    return PgvectorAdapter(
        table_name=config["table"],
        ef_search=args.ef_search,
    )


def warmup_worker(adapter, queries, indices, selectivity, count):
    for j in range(count):
        i = indices[j % len(indices)]

        adapter.search(
            queries[i],
            k=K,
            selectivity=selectivity,
        )


def search_worker(adapter, queries, indices, selectivity, worker_id):
    results = []
    progress_every = max(1, min(25, (len(indices) + 3) // 4))

    for position, i in enumerate(indices, start=1):
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

        # Progress is printed after recording this query's latency.
        if position % progress_every == 0 or position == len(indices):
            print(
                f"Progress filter={selectivity}% "
                f"worker={worker_id}: {position}/{len(indices)}",
                flush=True,
            )

    return results


def read_ground_truth(data):
    return {
        selectivity: data[f"gt_{selectivity}"].copy()
        for selectivity in SELECTIVITIES
    }


def load_dataset(config):
    dataset_path = config["path"]
    ground_truth_path = config["ground_truth_path"]

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset file does not exist: {dataset_path}"
        )

    with np.load(dataset_path, allow_pickle=False) as data:
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
        config["expected_vectors"],
        config["dimensions"],
    )

    if vectors_shape != expected_shape:
        raise ValueError(
            f"Unexpected dataset shape: {vectors_shape}, "
            f"expected {expected_shape}"
        )

    if (
        queries.ndim != 2
        or queries.shape[1] != expected_shape[1]
    ):
        raise ValueError(
            f"Unexpected query shape: {queries.shape}"
        )

    if queries.dtype != np.float32:
        raise ValueError(
            f"Queries must be float32, got {queries.dtype}"
        )

    if not np.isfinite(queries).all():
        raise ValueError("Non-finite query vectors")

    expected_queries = config.get("expected_queries")

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
                f"Ground truth is not ready: {ground_truth_path}"
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
                f"Incorrect ground truth shape for "
                f"{selectivity}%: {ground_truth.shape}"
            )

        for row in ground_truth:
            if len(set(row.tolist())) != K:
                raise ValueError(
                    f"Duplicate ground truth IDs "
                    f"for filter {selectivity}%"
                )

    return queries, ground_truths


def execute_workload(
    executor,
    adapters,
    queries,
    groups,
    selectivity,
):
    # The timed interval includes the concurrent workload,
    # but excludes warmup and recall calculations.
    started_at = perf_counter()

    futures = [
        executor.submit(
            search_worker,
            adapters[j],
            queries,
            groups[j],
            selectivity,
            j + 1,
        )
        for j in range(len(adapters))
    ]

    batches = [
        future.result()
        for future in futures
    ]

    ended_at = perf_counter()

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

    return (
        returned,
        latencies_ms,
        started_at,
        ended_at,
    )


def calculate_query_metrics(
    returned,
    latencies_ms,
    ground_truth,
    duration,
):
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
            recall_at_k(ids, ground_truth[i])
        )

    return {
        "mean_recall_at_10": float(np.mean(recalls)),
        "min_recall_at_10": float(np.min(recalls)),
        "max_recall_at_10": float(np.max(recalls)),
        "queries_with_less_than_k": short_results,
        "latency_p50_ms": float(
            np.percentile(latencies_ms, 50)
        ),
        "latency_p95_ms": float(
            np.percentile(latencies_ms, 95)
        ),
        "latency_p99_ms": float(
            np.percentile(latencies_ms, 99)
        ),
        "qps": float(len(returned) / duration),
        "measured_duration_seconds": float(duration),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Query benchmark for "
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
        "--chroma-candidates",
        type=int,
        default=K,
        help=(
            "Chroma n_results to request before retaining top 10; "
            "default 10 (no oversampling)"
        ),
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

    parser.add_argument(
        "--mode",
        choices=["development_pilot", "official"],
        default="development_pilot",
        help="Label results and select the matching output directory",
    )

    args = parser.parse_args()

    if args.ef_search < 1:
        parser.error("--ef-search must be positive")

    if args.warmup_queries < 0:
        parser.error(
            "--warmup-queries cannot be negative"
        )

    if args.chroma_candidates < K:
        parser.error(
            f"--chroma-candidates must be at least {K}"
        )

    if args.db != "chroma" and args.chroma_candidates != K:
        parser.error(
            "--chroma-candidates applies only to Chroma"
        )

    config = DATASETS[args.dataset]

    # This runner runs all four filters. Reject settings that could
    # request more neighbors than the 1% filter contains.
    if args.db == "chroma":
        smallest_filter = min(SELECTIVITIES)
        minimum_eligible = (
            config["expected_vectors"] * smallest_filter // 100
        )
        if args.chroma_candidates > minimum_eligible:
            parser.error(
                f"--chroma-candidates={args.chroma_candidates} "
                f"exceeds {minimum_eligible} eligible vectors "
                f"for the {smallest_filter}% filter"
            )

    queries, ground_truths = load_dataset(config)

    if len(queries) < args.clients:
        parser.error(
            "Not enough queries for client count"
        )

    groups = [
        list(range(j, len(queries), args.clients))
        for j in range(args.clients)
    ]

    target_name = (
        config["collection"]
        if args.db == "chroma"
        else config["table"]
    )

    container_name = (
        "vec3-chroma"
        if args.db == "chroma"
        else "vec3-postgres"
    )

    print("\n=== QUERY RUN CONFIGURATION ===")
    print(f"Database: {args.db}")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset file: {config['path']}")
    print(
        "Ground truth file: "
        f"{config['ground_truth_path'] or config['path']}"
    )
    print(f"Target: {target_name}")
    print(f"Base vectors: {config['expected_vectors']}")
    print(f"Dimensions: {config['dimensions']}")
    print(f"Queries: {len(queries)}")
    print(f"Clients: {args.clients}")
    print(f"Warmup per client: {args.warmup_queries}")
    print(f"Mode: {args.mode}")

    if args.db == "pgvector":
        print(f"ef_search: {args.ef_search}")
    else:
        print(f"Chroma n_results: {args.chroma_candidates}")
        print("Recall evaluated on returned top 10")

    results = {
        "mode": args.mode,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "database": args.db,
        "dataset_key": args.dataset,
        "dataset": config["path"].name,
        "ground_truth_file": (
            config["ground_truth_path"].name
            if config["ground_truth_path"]
            else config["path"].name
        ),
        "target": target_name,
        "base_vectors": config["expected_vectors"],
        "dimensions": config["dimensions"],
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
        "chroma_candidates": (
            args.chroma_candidates
            if args.db == "chroma"
            else None
        ),
        "search_strategy": (
            "chroma_oversampling_top10"
            if args.db == "chroma" and args.chroma_candidates > K
            else "standard_top10"
        ),
        "filters": {},
    }

    adapters = []
    monitor = ResourceMonitor(container_name)
    monitor_started = False
    resource_summary = None

    # Record the exact monotonic window of each filter.
    workload_windows = {}

    try:
        adapters = [
            create_adapter(args, config)
            for _ in range(args.clients)
        ]

        if args.db == "chroma":
            # Record actual collection configuration, rather than treating
            # requested n_results as a substitute for the HNSW parameters.
            hnsw_config = adapters[0].collection.configuration["hnsw"]
            if hnsw_config["space"] != "l2":
                raise ValueError(
                    f"Expected L2 Chroma index, got {hnsw_config['space']!r}"
                )
            results["chroma_hnsw_configuration"] = hnsw_config
            print(f"Chroma HNSW: {hnsw_config}", flush=True)

        monitor.start()
        monitor_started = True

        with ThreadPoolExecutor(
            max_workers=args.clients
        ) as executor:

            for selectivity in SELECTIVITIES:
                print(
                    f"\nSTART filter={selectivity}% "
                    f"queries={len(queries)} clients={args.clients}",
                    flush=True,
                )
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

                (
                    returned,
                    latencies_ms,
                    started_at,
                    ended_at,
                ) = execute_workload(
                    executor,
                    adapters,
                    queries,
                    groups,
                    selectivity,
                )

                workload_windows[selectivity] = (
                    started_at,
                    ended_at,
                )

                metrics = calculate_query_metrics(
                    returned,
                    latencies_ms,
                    ground_truths[selectivity],
                    ended_at - started_at,
                )

                results["filters"][
                    str(selectivity)
                ] = metrics

                print(
                    f"{args.db} "
                    f"| dataset={args.dataset} "
                    f"| clients={args.clients} "
                    f"| filter={selectivity}% "
                    f"| recall@10="
                    f"{metrics['mean_recall_at_10']:.4f} "
                    f"| p50="
                    f"{metrics['latency_p50_ms']:.2f} ms "
                    f"| p95="
                    f"{metrics['latency_p95_ms']:.2f} ms "
                    f"| p99="
                    f"{metrics['latency_p99_ms']:.2f} ms "
                    f"| QPS={metrics['qps']:.2f} "
                    f"| short="
                    f"{metrics['queries_with_less_than_k']}",
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

    # The monitor has now stopped. All collected samples
    # are available for per-filter summaries.
    for selectivity, window in workload_windows.items():
        started_at, ended_at = window

        filter_resources = monitor.summarize_interval(
            started_at,
            ended_at,
        )

        results["filters"][
            str(selectivity)
        ]["resources"] = filter_resources

        print(
            f"Resources | filter={selectivity}% "
            f"| samples={filter_resources['sample_count']} "
            f"| status={filter_resources['status']} "
            f"| peak_ram="
            f"{filter_resources['peak_memory_mib']} MiB",
            flush=True,
        )

    results["resources"] = resource_summary
    results["resources_scope"] = (
        "entire_run_including_warmup"
    )

    print("\n=== OVERALL RESOURCE MONITOR ===")
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

    output_dir = Path(
        "results/official" if args.mode == "official"
        else "results/pilot"
    )
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
        else f"candidates{args.chroma_candidates}"
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
