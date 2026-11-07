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

    # Stream output so long runs show live progress.
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
        if selectivity not in result.get("filters", {}):
            raise ValueError(
                f"Run {number}: missing filter "
                f"{selectivity}%"
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
        help="Dataset selection (default: 10k)",
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
        f"Warmup queries per client per filter: "
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
    }

    print("\n===== REPETITION SUMMARY =====")

    for selectivity in FILTERS:
        metrics = [
            result["filters"][selectivity]
            for result in run_results
        ]

        qps = [
            item["qps"]
            for item in metrics
        ]

        p95 = [
            item["latency_p95_ms"]
            for item in metrics
        ]

        recall = [
            item["mean_recall_at_10"]
            for item in metrics
        ]

        values = {
            "median_qps": float(median(qps)),
            "mean_qps": float(mean(qps)),
            "std_qps": float(stdev(qps)),
            "median_p95_ms": float(median(p95)),
            "mean_recall_at_10": float(mean(recall)),
            "qps_per_run": qps,
            "p95_ms_per_run": p95,
            "recall_at_10_per_run": recall,
        }

        summary["filters"][selectivity] = values

        print(
            f"Filter {selectivity:>3}% | "
            f"Recall@10="
            f"{values['mean_recall_at_10']:.4f} | "
            f"median QPS="
            f"{values['median_qps']:.2f} | "
            f"std QPS="
            f"{values['std_qps']:.2f} | "
            f"median p95="
            f"{values['median_p95_ms']:.2f} ms",
            flush=True,
        )

    # Resource measurements remain in individual JSON files.
    # Each monitor covers the whole run, including warmup.
    summary["resource_measurements"] = (
        "See individual run_files. "
        "Resource monitoring covers the entire run, "
        "including warmup."
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
    print("REPETITIONS COMPLETED")


if __name__ == "__main__":
    main()
