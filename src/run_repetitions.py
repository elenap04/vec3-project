import argparse
import json
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

from src.run_queries import DATASETS


FILTERS = ("100", "50", "10", "1")
DATASET_KEYS = tuple(sorted(DATASETS))

MIN_RESOURCE_SAMPLES = 3


def numeric_summary(values):
    values = [float(value) for value in values]

    if not values:
        return {
            "median": None,
            "mean": None,
            "std": None,
        }

    return {
        "median": float(median(values)),
        "mean": float(mean(values)),
        "std": (
            float(stdev(values))
            if len(values) >= 2
            else None
        ),
    }


def summarize_resource_field(resources, field):
    """
    Use only measurements marked as valid.

    Preserve missing values as null, so the summary
    does not suggest that every run produced a
    reliable measurement.
    """

    per_run = []

    for resource in resources:
        if (
            resource.get("status") == "ok"
            and resource.get("sample_count", 0)
            >= MIN_RESOURCE_SAMPLES
            and resource.get(field) is not None
        ):
            per_run.append(float(resource[field]))
        else:
            per_run.append(None)

    valid_values = [
        value
        for value in per_run
        if value is not None
    ]

    return {
        "per_run": per_run,
        "valid_run_count": len(valid_values),
        **numeric_summary(valid_values),
    }


def summarize_resources(metrics):
    resources = [
        item.get("resources") or {}
        for item in metrics
    ]

    statuses = [
        resource.get("status", "missing")
        for resource in resources
    ]

    sample_counts = [
        int(resource.get("sample_count", 0))
        for resource in resources
    ]

    valid_runs = sum(
        status == "ok"
        and samples >= MIN_RESOURCE_SAMPLES
        for status, samples in zip(
            statuses,
            sample_counts,
        )
    )

    return {
        "status": (
            "all_runs_valid"
            if valid_runs == len(metrics)
            else "partial_or_insufficient"
        ),
        "valid_runs": valid_runs,
        "total_runs": len(metrics),
        "status_per_run": statuses,
        "sample_count_per_run": sample_counts,
        "peak_memory_mib": summarize_resource_field(
            resources,
            "peak_memory_mib",
        ),
        "mean_cpu_percent": summarize_resource_field(
            resources,
            "mean_cpu_percent",
        ),
        "peak_cpu_percent": summarize_resource_field(
            resources,
            "peak_cpu_percent",
        ),
    }


def execute_run(args, number):
    print(
        f"\n===== RUN {number}/{args.runs} "
        f"| db={args.db} "
        f"| dataset={args.dataset} "
        f"| clients={args.clients} =====",
        flush=True,
    )

    command = [
        sys.executable,
        "-m",
        "src.run_queries",
        "--db",
        args.db,
        "--dataset",
        args.dataset,
        "--warmup-queries",
        str(args.warmup_queries),
        "--clients",
        str(args.clients),
    ]

    if args.db == "pgvector":
        command.extend([
            "--ef-search",
            str(args.ef_search),
        ])

    result_paths = []
    prefix = "Results saved to: "

    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:

        if process.stdout is None:
            raise RuntimeError(
                "Could not capture benchmark output"
            )

        for line in process.stdout:
            print(line, end="", flush=True)

            if line.startswith(prefix):
                result_paths.append(
                    line[len(prefix):].strip()
                )

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Run {number}/{args.runs} failed "
            f"with exit code {return_code}"
        )

    if len(result_paths) != 1:
        raise RuntimeError(
            "Expected exactly one result file, "
            f"found {len(result_paths)}"
        )

    path = Path(result_paths[0])

    if not path.is_file():
        raise FileNotFoundError(path)

    result = json.loads(
        path.read_text(encoding="utf-8")
    )

    if result.get("database") != args.db:
        raise ValueError(
            f"Run {number}: unexpected database"
        )

    if result.get("dataset_key") != args.dataset:
        raise ValueError(
            f"Run {number}: unexpected dataset"
        )

    if result.get("clients") != args.clients:
        raise ValueError(
            f"Run {number}: unexpected client count"
        )

    if result.get("mode") != "development_pilot":
        raise ValueError(
            f"Run {number}: unexpected benchmark mode"
        )

    if args.db == "pgvector":
        if result.get("ef_search") != args.ef_search:
            raise ValueError(
                f"Run {number}: unexpected ef_search"
            )

    for selectivity in FILTERS:
        filter_metrics = result.get(
            "filters", {}
        ).get(selectivity)

        if not isinstance(filter_metrics, dict):
            raise ValueError(
                f"Run {number}: missing filter "
                f"{selectivity}%"
            )

        for field in (
            "qps",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "mean_recall_at_10",
        ):
            if field not in filter_metrics:
                raise ValueError(
                    f"Run {number}: missing {field} "
                    f"for filter {selectivity}%"
                )

    print(
        f"RUN {number}/{args.runs} VALIDATED",
        flush=True,
    )

    return str(path), result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Repeated development query benchmarks "
            "for Chroma and PostgreSQL/pgvector"
        )
    )

    parser.add_argument(
        "--db",
        required=True,
        choices=["chroma", "pgvector"],
    )

    parser.add_argument(
        "--dataset",
        choices=DATASET_KEYS,
        default="10k",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
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

    parser.add_argument(
        "--clients",
        type=int,
        choices=[1, 8],
        default=1,
    )

    args = parser.parse_args()

    if args.runs < 2:
        parser.error(
            "--runs must be at least 2"
        )

    if args.ef_search < 1:
        parser.error(
            "--ef-search must be positive"
        )

    if args.warmup_queries < 0:
        parser.error(
            "--warmup-queries cannot be negative"
        )

    print("\n=== REPETITIONS CONFIGURATION ===")
    print(f"Database: {args.db}")
    print(f"Dataset: {args.dataset}")
    print(f"Clients: {args.clients}")
    print(f"Repetitions: {args.runs}")
    print(
        "Warmup queries per client per filter: "
        f"{args.warmup_queries}"
    )

    if args.db == "pgvector":
        print(f"ef_search: {args.ef_search}")

    run_files = []
    run_results = []

    for number in range(1, args.runs + 1):
        path, result = execute_run(
            args,
            number,
        )

        if run_results:
            reference = run_results[0]

            for field in (
                "dataset",
                "ground_truth_file",
                "target",
                "base_vectors",
                "dimensions",
                "queries",
                "k",
                "metric",
                "client_model",
                "warmup_queries_per_client_per_filter",
            ):
                if result.get(field) != reference.get(field):
                    raise ValueError(
                        f"Run {number}: inconsistent {field}"
                    )

        run_files.append(path)
        run_results.append(result)

    reference = run_results[0]

    summary = {
        "mode": "development_pilot",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "database": args.db,
        "dataset_key": args.dataset,
        "dataset": reference["dataset"],
        "ground_truth_file": reference.get(
            "ground_truth_file"
        ),
        "target": reference["target"],
        "base_vectors": reference["base_vectors"],
        "dimensions": reference["dimensions"],
        "queries_per_run": reference["queries"],
        "k": reference["k"],
        "metric": reference["metric"],
        "clients": args.clients,
        "client_model": "closed_loop",
        "repetitions": args.runs,
        "ef_search": (
            args.ef_search
            if args.db == "pgvector"
            else None
        ),
        "warmup_queries_per_client_per_filter": (
            args.warmup_queries
        ),
        "run_files": run_files,
        "filters": {},
        "warnings": [],
    }

    if reference["queries"] < 1000:
        summary["warnings"].append(
            "Fewer than 1000 measured queries per "
            "filter per run. P99 is exploratory and "
            "needs a larger workload for reliable "
            "official reporting."
        )

    if args.runs < 5:
        summary["warnings"].append(
            "Fewer than five repetitions; "
            "not sufficient for the official protocol."
        )

    print("\n===== REPETITION SUMMARY =====")

    for selectivity in FILTERS:
        metrics = [
            result["filters"][selectivity]
            for result in run_results
        ]

        metric_fields = {
            "qps": "qps",
            "p50_ms": "latency_p50_ms",
            "p95_ms": "latency_p95_ms",
            "p99_ms": "latency_p99_ms",
            "recall_at_10": "mean_recall_at_10",
        }

        values = {}

        for name, source_field in metric_fields.items():
            per_run = [
                float(item[source_field])
                for item in metrics
            ]

            statistics = numeric_summary(per_run)

            values[f"{name}_per_run"] = per_run
            values[f"median_{name}"] = statistics["median"]
            values[f"mean_{name}"] = statistics["mean"]
            values[f"std_{name}"] = statistics["std"]

        values["resources"] = summarize_resources(
            metrics
        )

        summary["filters"][selectivity] = values

        print(
            f"Filter {selectivity:>3}% | "
            f"Recall@10="
            f"{values['mean_recall_at_10']:.4f} | "
            f"median QPS="
            f"{values['median_qps']:.2f} | "
            f"median p50="
            f"{values['median_p50_ms']:.2f} ms | "
            f"median p95="
            f"{values['median_p95_ms']:.2f} ms | "
            f"median p99="
            f"{values['median_p99_ms']:.2f} ms | "
            f"valid resource runs="
            f"{values['resources']['valid_runs']}"
            f"/{args.runs}",
            flush=True,
        )

        if (
            values["resources"]["valid_runs"]
            != args.runs
        ):
            summary["warnings"].append(
                f"Filter {selectivity}%: some "
                "resource measurements have "
                "insufficient samples or are missing."
            )

    summary["resource_measurements"] = (
        "Per-filter resource summaries are stored in "
        "filters[selectivity].resources. "
        "Only measurements with status='ok' and "
        "at least three samples are aggregated. "
        "Overall container measurements, including "
        "warmup, remain in the individual run files."
    )

    output_dir = Path(
        "results/pilot/repetitions"
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
        else "default"
    )

    output_path = output_dir / (
        f"summary_{args.db}_"
        f"{args.dataset}_"
        f"{setting}_"
        f"c{args.clients}_"
        f"{stamp}.json"
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSummary saved to: {output_path}"
    )

    if summary["warnings"]:
        print("\n===== WARNINGS =====")

        for warning in summary["warnings"]:
            print(f"- {warning}")

    print("REPETITIONS COMPLETED")


if __name__ == "__main__":
    main()
